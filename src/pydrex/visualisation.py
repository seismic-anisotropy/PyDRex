"""> PyDRex: Visualisation functions for texture data and test outputs."""
import functools as ft

import matplotlib as mpl
import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg as la

from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import stats as _stats
from pydrex import axes as _axes

# Always show XY grid by default.
plt.rcParams["axes.grid"] = True


def polefigures(
    datafile,
    i_range=None,
    postfix=None,
    savefile="polefigures.png",
    **kwargs,
):
    """Plot pole figures for CPO data.

    The data is read from fields ending with the optional `postfix` in the NPZ file
    `datafile`. Use `i_range` to specify the indices of the timesteps to be plotted,
    which can be any valid Python range object, e.g. `range(0, 12, 2)` with a step of 2.
    By default (`i_range=None`), a maximum of 25 timesteps are plotted.
    If the number would exceed this, a warning is printed,
    which signals the complete number of timesteps found in the file.

    Any additional keyword arguments are passed to `poles`.

    See also: `pydrex.minerals.Mineral.save`.

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
        ax100 = fig100.add_subplot(1, n_orientations, n + 1)
        set_polefig_axis(ax100)
        ax100.scatter(*poles(orientations, **kwargs), s=0.3, alpha=0.33, zorder=11)
        ax010 = fig010.add_subplot(1, n_orientations, n + 1)
        set_polefig_axis(ax010)
        ax010.scatter(
            *poles(orientations, hkl=[0, 1, 0], **kwargs), s=0.3, alpha=0.33, zorder=11
        )
        ax001 = fig001.add_subplot(1, n_orientations, n + 1)
        set_polefig_axis(ax001)
        ax001.scatter(
            *poles(orientations, hkl=[0, 0, 1], **kwargs), s=0.3, alpha=0.33, zorder=11
        )

    fig.savefig(_io.resolve_path(savefile), bbox_inches="tight")


def point_density(
    x_data, y_data, axial=True, gridsteps=100, weights=1, kernel="kamb_count", **kwargs
):
    """Calculate point density of spherical directions projected onto a circle.

    Calculate the density of points on a unit radius disk.
    Counting on a regular grid as well as smoothing is performed
    using the specified `kernel` in preparation for contour plotting.
    The data is assumed to be either axial or vectorial spherical directions
    projected onto the disk using an equal-area azimuthal transformation.
    The `x_data` and `y_data` values should normally come from `poles`.

    Args:
    - `x_data` (array) — data point coordinates on the first ℝ² axis
    - `y_data` (array) — data point coordinates on the second ℝ² axis
    - `axial` (bool, optional) — toggle axial or vectorial interpretation of the data
    - `gridstep` (int, optional) — number of steps along a diameter of the counting grid
    - `weights` (int|float|array, optional) — weights to apply to smoothed density
        values; either a fixed scaling or individual pre-normalised weights in an array
        matching the shape of `x_data` and `y_data`
    - `kernel` (string) — name of smoothing function, see `SPHERICAL_COUNTING_KERNELS`

    Any additional keyword arguments are passed to the kernel function.

    """
    weights = np.asarray(weights, dtype=np.float64)

    # Generate a regular grid of "counters" to measure on.
    x_counters, y_counters = np.mgrid[-1 : 1 : gridsteps * 1j, -1 : 1 : gridsteps * 1j]
    # Mask to remove any counters beyond the unit circle.
    mask = np.zeros(x_counters.shape, bool) | (
        np.sqrt(x_counters**2 + y_counters**2) > 1
    )

    def _apply_mask(a):
        return np.ma.array(a, mask=mask, fill_value=np.nan)

    x_counters = _apply_mask(x_counters)
    y_counters = _apply_mask(y_counters)

    # Basically, we can't model this as a convolution as we're not in Cartesian space,
    # so we have to iterate through and call the kernel function at each "counter".
    data = np.column_stack([x_data, y_data])
    counters = np.column_stack([x_counters.ravel(), y_counters.ravel()])
    totals = np.zeros(counters.shape[0])
    for i, counter in enumerate(counters):
        if axial:
            cos_dist = np.abs(np.dot(counter, data.transpose()))
        else:
            cos_dist = np.dot(counter, data.transpose())
        density, scale = kernel(cos_dist, axial=axial, **kwargs)
        density *= weights
        totals[i] = (density.sum() - 0.5) / scale

    # Traditionally, the negative values
    # (while valid, as they represent areas with less than expected point-density)
    # are not returned.
    # TODO: Make this a kwarg option.
    totals[totals < 0] = 0
    return x_counters, y_counters, _apply_mask(np.reshape(totals, x_counters.shape))


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
    as the upward pointing axis for the Lambert equal area projection.

    See also: `lambert_equal_area`, `set_polefig_axis`.

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
    xvals = _directions[:, axes_map[upward_axes]]
    yvals = _directions[:, axes_map[ref_axes[0]]]
    zvals = _directions[:, axes_map[ref_axes[1]]]
    return lambert_equal_area(xvals, yvals, zvals)


def lambert_equal_area(xvals, yvals, zvals):
    """Project points from a 3D sphere onto a 2D circle.

    Project points from a 3D sphere, given in Cartesian coordinates,
    to points on a 2D circle using the Lambert equal area azimuthal projection.
    Returns arrays of the X and Y coordinates in the unit circle.

    """

    # Sign of sin(x) defines the sign of the square root.
    @nb.njit()
    def _sgn_sin(xs):
        out = np.empty_like(xs)
        for i, x in enumerate(xs):
            if 0 < x < np.pi:
                out[i] = 1
            elif -np.pi < x < 0:
                out[i] = -1
            else:
                out[i] = 0
        return out

    # FIXME: Deal with xvals[i] == -1 (zero div.)
    # Mardia & Jupp 2009 (Directional Statistics), eq. 9.1.1,
    # The equation is given in spherical coordinates, [cosθ, sinθcosφ, sinθsinφ].
    # Also, they project onto a disk of radius 2, which in Cartesian would be:
    #   _sgn_sin(np.arccos(xvals) / 2) * 2 / np.sqrt(2) * 1 / np.sqrt(1 + xvals)
    # But that is silly, and we will use a disk of radius 1, as Euler intended.
    prefactor = _sgn_sin(np.arccos(xvals)) / np.sqrt(2) * 1 / np.sqrt(1 + xvals)
    return prefactor * yvals, prefactor * zvals


def _kamb_radius(n, σ, axial):
    """Radius of kernel for Kamb-style smoothing."""
    r = σ**2 / (float(n) + σ**2)
    if axial is True:
        return 1 - r
    return 1 - 2 * r


def _kamb_units(n, radius):
    """Normalization function for Kamb-style counting."""
    return np.sqrt(n * radius * (1 - radius))


def exponential_kamb(cos_dist, σ=3, axial=True):
    """Kernel function from Vollmer 1995 for exponential smoothing."""
    n = float(cos_dist.size)
    if axial:
        f = 2 * (1.0 + n / σ**2)
        units = np.sqrt(n * (f / 2.0 - 1) / f**2)
    else:
        f = 1 + n / σ**2
        units = np.sqrt(n * (f - 1) / (4 * f**2))

    count = np.exp(f * (cos_dist - 1))
    return count, units


def linear_inverse_kamb(cos_dist, σ=3, axial=True):
    """Kernel function from Vollmer 1995 for linear smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, σ, axial=axial)
    f = 2 / (1 - radius)
    cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius)
    return count, _kamb_units(n, radius)


def square_inverse_kamb(cos_dist, σ=3, axial=True):
    """Kernel function from Vollmer 1995 for inverse square smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, σ, axial=axial)
    f = 3 / (1 - radius) ** 2
    cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius) ** 2
    return count, _kamb_units(n, radius)


def kamb_count(cos_dist, σ=3, axial=True):
    """Original Kamb 1959 kernel function (raw count within radius)."""
    n = float(cos_dist.size)
    dist = _kamb_radius(n, σ, axial=axial)
    count = (cos_dist >= dist).astype(float)
    return count, _kamb_units(n, dist)


def schmidt_count(cos_dist, axial=None):
    """Schmidt (a.k.a. 1%) counting kernel function."""
    radius = 0.01
    count = ((1 - cos_dist) <= radius).astype(float)
    # To offset the count.sum() - 0.5 required for the kamb methods...
    count = 0.5 / count.size + count
    return count, (cos_dist.size * radius)


SPHERICAL_COUNTING_KERNELS = {
    kamb_count,
    schmidt_count,
    exponential_kamb,
    linear_inverse_kamb,
    square_inverse_kamb,
}
"""Kernel functions that return an un-summed distribution and a normalization factor.

Supported kernel functions are based on the discussion in
[Vollmer 1995](https://doi.org/10.1016/0098-3004(94)00058-3).
Kamb methods accept the parameter `σ` (default: 3) to control the degree of smoothing.

"""


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
