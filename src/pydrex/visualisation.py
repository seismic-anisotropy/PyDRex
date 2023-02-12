"""PyDRex: Visualisation helpers for tests/examples."""
import functools as ft
import pathlib as pl

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def _savefig_deep(fig, path):
    """Saves a `plt.Figure` to `path`, creating necessary parent directories."""
    pl.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


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

    _savefig_deep(fig, savefile)


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
    cpo_threshold=0.4,
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

    _savefig_deep(fig, savefile)
