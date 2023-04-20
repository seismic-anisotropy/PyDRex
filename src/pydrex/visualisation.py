"""> PyDRex: Visualisation functions for texture data and test outputs."""
import functools as ft

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import projections as mproj
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg as la

from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import stats as _stats
from pydrex import axes as _axes

# Always show XY grid by default.
plt.rcParams["axes.grid"] = True
# Make sure we have the required matplotlib "projections" (really just Axes subclasses).
if "pydrex.polefigure" not in mproj.get_projection_names():
    _log.warning(
        "failed to find pydrex.polefigure projection; it should be registered in %s",
        _axes,
    )


def polefigures(
    datafile,
    i_range=None,
    postfix=None,
    density=False,
    ref_axes="xz",
    savefile="polefigures.png",
    **kwargs,
):
    """Plot [100], [010] and [001] pole figures for CPO data.

    The data is read from fields ending with the optional `postfix` in the NPZ file
    `datafile`. Use `i_range` to specify the indices of the timesteps to be plotted,
    which can be any valid Python range object, e.g. `range(0, 12, 2)` with a step of 2.
    By default (`i_range=None`), a maximum of 25 timesteps are plotted.
    If the number would exceed this, a warning is printed,
    which signals the complete number of timesteps found in the file.

    Use `density=True` to plot contoured pole figures instead of raw points.
    In this case, any additional keyword arguments are passed to
    `pydrex.stats.point_density`.

    See also: `pydrex.minerals.Mineral.save`, `pydrex.axes.PoleFigureAxes.polefigure`.

    """
    mineral = _minerals.Mineral.from_file(datafile, postfix=postfix)
    if i_range is None:
        i_range = range(0, len(mineral.orientations))
        if len(i_range) > 25:
            _log.warning("truncating to 25 timesteps (out of %s total)", len(i_range))
            i_range = range(0, 25)

    orientations_resampled = [
        _stats.resample_orientations(mineral.orientations[i], mineral.fractions[i])[0]
        for i in np.arange(i_range.start, i_range.stop, i_range.step, dtype=int)
    ]
    n_orientations = len(orientations_resampled)

    fig = plt.figure(figsize=(n_orientations, 4), dpi=600)
    grid = fig.add_gridspec(3, n_orientations, hspace=0, wspace=0.2)
    fig100 = fig.add_subfigure(grid[0, :], frameon=False)
    fig100.suptitle("[100]")
    fig010 = fig.add_subfigure(grid[1, :], frameon=False)
    fig010.suptitle("[010]")
    fig001 = fig.add_subfigure(grid[2, :], frameon=False)
    fig001.suptitle("[001]")
    for n, orientations in enumerate(orientations_resampled):
        ax100 = fig100.add_subplot(
            1, n_orientations, n + 1, projection="pydrex.polefigure"
        )
        ax100.polefigure(
            orientations, hkl=[1, 0, 0], density=density, density_kwargs=kwargs
        )
        ax010 = fig010.add_subplot(
            1, n_orientations, n + 1, projection="pydrex.polefigure"
        )
        ax010.polefigure(
            orientations, hkl=[0, 1, 0], density=density, density_kwargs=kwargs
        )
        ax001 = fig001.add_subplot(
            1, n_orientations, n + 1, projection="pydrex.polefigure"
        )
        ax001.polefigure(
            orientations, hkl=[0, 0, 1], density=density, density_kwargs=kwargs
        )

    fig.savefig(_io.resolve_path(savefile), bbox_inches="tight")


def poles(orientations, ref_axes="xz", hkl=[1, 0, 0]):
    """Calculate stereographic poles from 3D orientation matrices.

    Expects `orientations` to be an array with shape (N, 3, 3).
    The optional arguments `ref_axes` and `hkl` can be used to specify
    the stereograph axes and the crystallographic axis respectively.
    The stereograph axes should be given as a string of two letters,
    e.g. "xz" (default), and the third letter in the set "xyz" is used
    as the 'northward' pointing axis for the Lambert equal area projection.

    See also: `lambert_equal_area`.

    """
    upward_axes = next((set("xyz") - set(ref_axes)).__iter__())
    axes_map = {"x": 0, "y": 1, "z": 2}
    directions = np.tensordot(orientations.transpose([0, 2, 1]), hkl, axes=(2, 0))
    directions_norm = la.norm(directions, axis=1)
    directions[:, 0] /= directions_norm
    directions[:, 1] /= directions_norm
    directions[:, 2] /= directions_norm

    zvals = directions[:, axes_map[upward_axes]]
    yvals = directions[:, axes_map[ref_axes[1]]]
    xvals = directions[:, axes_map[ref_axes[0]]]
    return lambert_equal_area(xvals, yvals, zvals)


def lambert_equal_area(xvals, yvals, zvals):
    """Project axial data from a 3D sphere onto a 2D disk.

    Project points from a 3D sphere, given in Cartesian coordinates,
    to points on a 2D disk using a Lambert equal area azimuthal projection.
    Returns arrays of the X and Y coordinates in the unit disk.

    This implementation first maps all points onto the same hemisphere,
    and then projects that hemisphere onto the disk.

    """
    # One reference for the equation is Mardia & Jupp 2009 (Directional Statistics),
    # where it appears as eq. 9.1.1 in spherical coordinates,
    #   [sinθcosφ, sinθsinφ, cosθ].
    # They project onto a disk of radius 2, but this is not necessary
    # if we are only projecting poionts from one hemisphere.

    # First we move all points into the upper hemisphere.
    # See e.g. page 186 of Snyder 1987 (Map Projections— A Working Manual).
    # This is done by taking θ' = π - θ where θ is the colatitude (inclination).
    # Equivalently, in Cartesian coords we just take abs of the z values.
    zvals = abs(zvals)
    # When x and y are both 0, we would have a zero-division.
    # These points are always projected onto [0, 0] (the centre of the disk).
    condition = np.logical_and(np.abs(xvals) < 1e-16, np.abs(yvals) < 1e-16)
    x_masked = np.ma.masked_where(condition, xvals)
    y_masked = np.ma.masked_where(condition, yvals)
    x_masked.fill_value = np.nan
    y_masked.fill_value = np.nan
    # The equations in Mardia & Jupp 2009 and Snyder 1987 both project the hemisphere
    # onto a disk of radius sqrt(2), so we drop the sqrt(2) factor that appears
    # after converting to Cartesian coords to get a radius of 1.
    prefactor = np.sqrt((1 - zvals) / (x_masked**2 + y_masked**2))
    prefactor.fill_value = 0.0
    return prefactor.filled() * xvals, prefactor.filled() * yvals


def check_marker_seq(func):
    """Raises a `ValueError` if number of markers and data series don't match.

    The decorated function is expected to take the data as the first positional
    argument, and a keyword argument called `markers`.

    """

    @ft.wraps(func)
    def wrapper(data, *args, **kwargs):
        markers = kwargs["markers"]
        if len(data) % len(markers) != 0:
            raise ValueError(
                "Number of data series must be divisible by number of markers."
                + f" You've supplied {len(data)} data series and {len(markers)} markers."
            )
        func(data, *args, **kwargs)

    return wrapper


def _get_marker_and_label(data, seq_index, markers, labels=None):
    marker = markers[int(seq_index / (len(data) / len(markers)))]
    label = None
    if labels is not None:
        label = labels[int(seq_index / (len(data) / len(labels)))]
    return marker, label


@check_marker_seq
def simple_shear_stationary_2d(
    angles,
    indices,
    timestop,
    savefile="pydrex_simple_shear_stationary_2d.png",
    markers=("."),
    labels=None,
    refval=None,
):
    """Plot diagnostics for stationary A-type olivine 2D simple shear box tests."""
    fig = plt.figure(figsize=(5, 8), dpi=300)
    grid = fig.add_gridspec(2, 1, hspace=0.05)
    ax_mean = fig.add_subplot(grid[0])
    ax_mean.set_ylabel("Mean angle ∈ [0, 90]°")
    ax_mean.axhline(0, color=mpl.rcParams["axes.edgecolor"])
    ax_mean.tick_params(labelbottom=False)
    ax_strength = fig.add_subplot(grid[1], sharex=ax_mean)
    ax_strength.set_ylabel("Texture strength (M-index)")
    ax_strength.set_xlabel(r"Strain-time ($\dot{ε}_0 t$)")

    for i, (misorient_angles, misorient_indices) in enumerate(zip(angles, indices)):
        timestamps = np.linspace(0, timestop, len(misorient_angles))
        marker, label = _get_marker_and_label(angles, i, markers, labels)

        ax_mean.plot(
            timestamps,
            misorient_angles,
            marker,
            markersize=5,
            alpha=0.33,
            label=label,
        )
        ax_strength.plot(
            timestamps,
            misorient_indices,
            marker,
            markersize=5,
            alpha=0.33,
            label=label,
        )

    if refval is not None:
        ax_mean.plot(
            timestamps,
            refval * np.exp(timestamps * (np.cos(np.deg2rad(refval * 2)) - 1)),
            "r--",
            label=r"$θ ⋅ \exp[t ⋅ (\cos2θ - 1)]$,"
            + "\n\t"
            + rf"$\theta = {refval:.1f}$",
        )

    if labels is not None:
        ax_mean.legend()

    fig.savefig(_io.resolve_path(savefile), bbox_inches="tight")


def _lag_2d_corner_flow(θ):
    # Predicted grain orientation lag for 2D corner flow, eq. 11 in Kaminski 2002.
    _θ = np.ma.masked_less(θ, 1e-15)
    return (_θ * (_θ**2 + np.cos(_θ) ** 2)) / (
        np.tan(_θ) * (_θ**2 + np.cos(_θ) ** 2 - _θ * np.sin(2 * _θ))
    )


@check_marker_seq
def corner_flow_2d(
    x_paths,
    z_paths,
    angles,
    indices,
    directions,
    timestamps,
    xlabel,
    savefile="pydrex_corner_2d.png",
    markers=("."),
    labels=None,
    xlims=None,
    zlims=None,
    cpo_threshold=0.33,
    Π_levels=(0.1, 0.5, 1, 2, 3),
    tick_formatter=lambda x, pos: f"{x/1e3:.1f} km",
):
    """Plot diagnostics for prescribed path 2D corner flow tests."""
    fig = plt.figure(figsize=(12, 8), dpi=300)
    grid = fig.add_gridspec(2, 2, hspace=-0.2, wspace=0.025)
    ax_domain = fig.add_subplot(grid[0, :])
    ax_domain.set_ylabel("z")
    ax_domain.set_xlabel(xlabel)
    ax_domain.xaxis.set_ticks_position("top")
    ax_domain.xaxis.set_label_position("top")
    ax_domain.set_aspect("equal")
    ax_domain.xaxis.set_major_formatter(tick_formatter)
    ax_domain.yaxis.set_major_formatter(tick_formatter)
    ax_strength = fig.add_subplot(grid[1, 0])
    ax_strength.set_ylabel("Texture strength (M-index)")
    ax_strength.set_xlabel("Time (s)")
    ax_mean = fig.add_subplot(grid[1, 1])
    ax_mean.set_ylabel("Mean angle from horizontal (°)")
    ax_mean.set_xlabel("Time (s)")
    ax_mean.set_ylim((0, 90))
    ax_mean.yaxis.set_ticks_position("right")
    ax_mean.yaxis.set_label_position("right")

    for i, (
        misorient_angles,
        misorient_indices,
        x_series,
        z_series,
        bingham_vectors,
        t_series,
    ) in enumerate(zip(angles, indices, x_paths, z_paths, directions, timestamps)):
        x_series = np.array(x_series)
        z_series = np.array(z_series)
        # Π := grain orientation lag, dimensionless, see Kaminski 2002.
        Π_series = _lag_2d_corner_flow(np.arctan2(x_series, -z_series))
        Π_max_step = np.argmax(Π_series)
        # TODO: Fix the rest.
        # Index of timestamp where Π ≈ 1 (corner).
        corner_step = Π_max_step + np.argmin(np.abs(Π_series[Π_max_step:] - 1.0))

        marker, label = _get_marker_and_label(angles, i, markers, labels)

        mean_angles = ax_mean.plot(
            t_series,
            misorient_angles,
            marker,
            markersize=5,
            alpha=0.33,
            label=label,
        )
        color = mean_angles[0].get_color()  # `mean_angles` has one element.
        ax_mean.plot(
            t_series[corner_step],
            misorient_angles[corner_step],
            marker,
            markersize=5,
            color=mpl.rcParams["axes.edgecolor"],
            zorder=11,
            alpha=0.33,
        )
        ax_strength.plot(
            t_series[corner_step],
            misorient_indices[corner_step],
            marker,
            markersize=5,
            color=mpl.rcParams["axes.edgecolor"],
            zorder=11,
            alpha=0.33,
        )
        ax_strength.plot(
            t_series,
            misorient_indices,
            marker,
            markersize=5,
            color=color,
            alpha=0.33,
            label=label,
        )

        # Plot the prescribed pathlines, indicate CPO and location of Π ≈ 1.
        mask_cpo = misorient_indices > cpo_threshold
        ax_domain.plot(
            x_series[~mask_cpo],
            z_series[~mask_cpo],
            marker,
            markersize=5,
            alpha=0.33,
            label=label,
            zorder=10,
        )
        if np.any(mask_cpo):
            # TODO: Scale bingham_vectors by a meaningful length; FSE long axis?
            ax_domain.quiver(
                x_series[mask_cpo],
                z_series[mask_cpo],
                bingham_vectors[mask_cpo, 0],
                bingham_vectors[mask_cpo, 2],
                color=color,
                pivot="mid",
                width=3e-3,
                headaxislength=0,
                headlength=0,
                zorder=10,
            )
        ax_domain.plot(
            x_series[corner_step],
            z_series[corner_step],
            marker,
            markersize=5,
            color=mpl.rcParams["axes.edgecolor"],
            zorder=11,
        )
        if xlims is not None:
            ax_domain.set_xlim(*xlims)
        if zlims is not None:
            ax_domain.set_ylim(*zlims)

    # Plot grain orientation lag contours.
    x_ticks = np.linspace(
        *ax_domain.get_xlim(), len(ax_domain.xaxis.get_major_ticks()) * 10 + 1
    )
    z_ticks = np.linspace(
        *ax_domain.get_ylim(), len(ax_domain.yaxis.get_major_ticks()) * 10 + 1
    )
    x_grid, z_grid = np.meshgrid(x_ticks, z_ticks)
    θ_grid = np.arctan2(x_grid, -z_grid)
    Π_contours = ax_domain.contourf(
        x_ticks,
        z_ticks,
        _lag_2d_corner_flow(θ_grid),
        levels=Π_levels,
        extend="min",
        cmap="copper_r",
        alpha=0.33,
    )
    # Some hacky workaround to have better edges between the contours.
    for c in Π_contours.collections:
        c.set_edgecolor("face")
    plt.rcParams["axes.grid"] = False
    divider = make_axes_locatable(ax_domain)
    fig.colorbar(
        Π_contours,
        label="grain orientation lag, Π",
        cax=divider.append_axes("right", size="2%", pad=0.05),
    ).ax.invert_xaxis()
    plt.rcParams["axes.grid"] = True

    # Lines to show texture threshold and shear direction.
    ax_strength.axhline(
        cpo_threshold, color=mpl.rcParams["axes.edgecolor"], linestyle="--"
    )
    if labels is not None:
        ax_mean.legend()

    fig.savefig(_io.resolve_path(savefile), bbox_inches="tight")
