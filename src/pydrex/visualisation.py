"""> PyDRex: Visualisation helpers for tests/examples."""
import functools as ft

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg as la

from pydrex import io as _io
from pydrex import minerals as _minerals
from pydrex import stats as _stats

# Always show XY grid by default.
plt.rcParams["axes.grid"] = True


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
    ax_mean.axhline(0, color="k")
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

    fig.savefig(_io.resolve_path(), bbox_inches="tight")


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
            color="k",
            zorder=11,
            alpha=0.33,
        )
        ax_strength.plot(
            t_series[corner_step],
            misorient_indices[corner_step],
            marker,
            markersize=5,
            color="k",
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
            color="k",
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
    ax_strength.axhline(cpo_threshold, color="k", linestyle="--")
    if labels is not None:
        ax_mean.legend()

    fig.savefig(_io.resolve_path(), bbox_inches="tight")


def set_polefig_axis(ax, ref_axes="xz"):
    # NOTE: We could subclass matplotlib's Axes like Joe does in mplstereonet,
    # but this turns out to be a lot of effort for not much gain...
    ax.set_axis_off()
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_aspect("equal")
    _circle_points = np.linspace(0, np.pi * 2, 100)
    ax.plot(np.cos(_circle_points), np.sin(_circle_points), linewidth=1, color="k")
    ax.axhline(0, color="k", alpha=0.5)
    ax.text(1.05, 0.5, ref_axes[0], verticalalignment="center", transform=ax.transAxes)
    ax.axvline(0, color="k", alpha=0.5)
    ax.text(
        0.5, 1.05, ref_axes[1], horizontalalignment="center", transform=ax.transAxes
    )


def poles(orientations, ref_axes="xz", hkl=[1, 0, 0]):
    """Calculate stereographic poles from 3D orientation matrices.

    Expects `orientations` to be an array with shape (N, 3, 3).
    The optional arguments `ref_axes` and `hkl` can be used to specify
    the stereograph axes and the crystallographic axis respectively.
    The stereograph axes should be given as a string of two letters,
    e.g. "xz" (default), and the third letter in the set "xyz" is used
    as the upward pointing axis for the lower hemisphere Lambert equal area projection.

    """
    upward_axes = next((set("xyz") - set(ref_axes)).__iter__())
    axes_map = {"x": 0, "y": 1, "z": 2}
    directions = np.tensordot(orientations.transpose([0, 2, 1]), hkl, axes=(2, 0))
    directions_norm = la.norm(directions, axis=1)
    directions[:, 0] /= directions_norm
    directions[:, 1] /= directions_norm
    directions[:, 2] /= directions_norm

    _directions = directions
    # NOTE: Use this to mask directions that point to the upper hemisphere.
    # _directions = np.ma.mask_rows(
    #     np.ma.masked_where(
    #         np.zeros(directions.shape, bool) | (directions[:, 1] >= 0)[:, None],
    #         directions,
    #     )
    # )

    # Lambert equal-area projection, in Cartesian coords from 3D to the circle.
    # TODO: Find a good reference or derive the equations to check.
    # This stuff just comes from wikipedia:
    # https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection#cite_ref-borradaile2003_6-0
    # Also from the implementation of Fraters & Billen 2021.
    stereograph_xvals = _directions[:, axes_map[ref_axes[0]]] / (
        1 + np.abs(directions[:, axes_map[upward_axes]])
    )
    stereograph_yvals = _directions[:, axes_map[ref_axes[1]]] / (
        1 + np.abs(directions[:, axes_map[upward_axes]])
    )
    return stereograph_xvals, stereograph_yvals


def polefigures(datafile, step=1, postfix=None, savefile="polefigures.png"):
    """Plot pole figures for CPO data.

    The data is read from fields ending with the optional `postfix` in the NPZ file
    `datafile`. Pole figures are plotted at every `step` number of timesteps.

    """
    mineral = _minerals.Mineral.from_file(datafile, postfix=postfix)
    orientations_resampled = [
        _stats.resample_orientations(_orientations, _fractions)[0]
        for _orientations, _fractions in zip(mineral.orientations, mineral.fractions)
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
        ax100 = fig100.add_subplot(1, n_orientations, n + 1)
        set_polefig_axis(ax100)
        ax100.scatter(*poles(orientations), s=0.3, alpha=0.33, zorder=11)
        # ax100.contourf(*point_density(*poles(orientations)))

        ax010 = fig010.add_subplot(1, n_orientations, n + 1)
        set_polefig_axis(ax010)
        ax010.scatter(*poles(orientations, hkl=[0, 1, 0]), s=0.3, alpha=0.33, zorder=11)

        ax001 = fig001.add_subplot(1, n_orientations, n + 1)
        set_polefig_axis(ax001)
        ax001.scatter(*poles(orientations, hkl=[0, 0, 1]), s=0.3, alpha=0.33, zorder=11)

    fig.savefig(_io.resolve_path(), bbox_inches="tight")


# TODO: The contouring stuff below is mostly copied/adapted from mplstereonet, but I
# don't really know what I'm doing and it doesn't work yet.


def _kamb_radius(n, σ):
    """Radius of kernel for Kamb-style smoothing."""
    a = σ**2 / (float(n) + σ**2)
    return 1 - a


def _kamb_units(n, radius):
    """Normalization function for Kamb-style counting."""
    return np.sqrt(n * radius * (1 - radius))


# All of the following kernel functions return an _unsummed_ distribution and
# a normalization factor
def _exponential_kamb(cos_dist, σ=3):
    """Kernel function from Vollmer for exponential smoothing."""
    n = float(cos_dist.size)
    f = 2 * (1.0 + n / σ**2)
    count = np.exp(f * (cos_dist - 1))
    units = np.sqrt(n * (f / 2.0 - 1) / f**2)
    return count, units


def _linear_inverse_kamb(cos_dist, σ=3):
    """Kernel function from Vollmer for linear smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, σ)
    f = 2 / (1 - radius)
    cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius)
    return count, _kamb_units(n, radius)


def _square_inverse_kamb(cos_dist, σ=3):
    """Kernel function from Vollemer for inverse square smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, σ)
    f = 3 / (1 - radius) ** 2
    cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius) ** 2
    return count, _kamb_units(n, radius)


def _kamb_count(cos_dist, σ=3):
    """Original Kamb kernel function (raw count within radius)."""
    n = float(cos_dist.size)
    dist = _kamb_radius(n, σ)
    count = (cos_dist >= dist).astype(float)
    return count, _kamb_units(n, dist)


def _schmidt_count(cos_dist, σ=None):
    """Schmidt (a.k.a. 1%) counting kernel function."""
    radius = 0.01
    count = ((1 - cos_dist) <= radius).astype(float)
    # To offset the count.sum() - 0.5 required for the kamb methods...
    count = 0.5 / count.size + count
    return count, (cos_dist.size * radius)


def point_density(
    x_data, y_data, kernel=_kamb_count, σ=3, gridsize=(100, 100), weights=None
):
    """Calculate point density of spherical data projected onto a circle.

    .. warning:: This method is currently broken.

    """
    if weights is None:
        weights = 1

    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()  # TODO: mplstereonet uses .mean()?

    # Generate a regular grid of "counters" to measure on.
    x_counters, y_counters = np.mgrid[
        -1 : 1 : gridsize[0] * 1j, -1 : 1 : gridsize[1] * 1j
    ]
    # Mask to remove any counters beyond the unit circle.
    mask = np.zeros(x_counters.shape, bool) | (
        np.sqrt(x_counters**2 + y_counters**2) > 1
    )
    x_counters = np.ma.array(x_counters, mask=mask, fill_value=np.nan)
    y_counters = np.ma.array(y_counters, mask=mask, fill_value=np.nan)

    # Basically, we can't model this as a convolution as we're not in Cartesian
    # space, so we have to iterate through and call the kernel function at
    # each "counter".
    data = np.vstack([x_data, y_data])
    totals = np.zeros(x_counters.shape, dtype=np.float64)
    for i in range(x_counters.shape[0]):
        for j in range(x_counters.shape[1]):
            cos_dist = np.abs(
                np.dot(np.array([x_counters[i, j], y_counters[i, j]]), data)
            )
            density, scale = kernel(cos_dist, σ)
            density *= weights
            totals[i, j] = (density.sum() - 0.5) / scale

    # Traditionally, the negative values (while valid, as they represent areas
    # with less than expected point-density) are not returned.
    # totals[totals < 0] = 0
    # print(np.nanmax(totals))
    # print(np.nanmin(totals))
    return x_counters, y_counters, totals
