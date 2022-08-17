"""PyDRex: Visualisation helpers for tests/examples."""
import numpy as np
from matplotlib import pyplot as plt


def simple_shear_2d(
    angles,
    indices,
    timestop,
    savefile="pydrex_simlpe_shear_2d.png",
    markers=("."),
    labels=None,
    refval=None,
):
    """Plot diagnostics for 2D simple shear tests."""
    fig = plt.figure(figsize=(5, 8), dpi=300)
    grid = fig.add_gridspec(2, 1, hspace=0.05)
    ax_mean = fig.add_subplot(grid[0])
    ax_mean.set_ylabel("Mean angle ∈ [0, 90]°")
    ax_mean.axhline(0, color="k")
    ax_mean.tick_params(labelbottom=False)
    ax_strength = fig.add_subplot(grid[1], sharex=ax_mean)
    ax_strength.set_ylabel("Texture strength (M-index)")
    ax_strength.set_xlabel(r"Strain-time ($\dot{ε}_0 t$)")

    if len(angles) % len(markers) != 0:
        raise ValueError(
            "Number of data series must be divisible by number of markers."
            + f" You've supplied {len(angles)} data series and {len(markers)} markers."
        )

    for i, (misorient_angles, misorient_indices) in enumerate(zip(angles, indices)):
        timesteps = np.linspace(0, timestop, len(misorient_angles))
        label = None
        if labels is not None:
            label = labels[int(i / (len(angles) / len(labels)))]
        markerseq = markers[int(i / (len(angles) / len(markers)))]

        ax_mean.plot(
            timesteps,
            misorient_angles,
            markerseq,
            markersize=5,
            alpha=0.33,
            label=label,
        )
        ax_strength.plot(
            timesteps,
            misorient_indices,
            markerseq,
            markersize=5,
            alpha=0.33,
            label=label,
        )

    if refval is not None:
        ax_mean.plot(
            timesteps,
            refval * np.exp(timesteps * (np.cos(np.deg2rad(refval * 2)) - 1)),
            "r--",
            label=r"$θ ⋅ \exp[t ⋅ (\cos2θ - 1)]$,"
            + "\n\t"
            + rf"$\theta = {refval:.1f}$",
        )

    if labels is not None:
        ax_mean.legend()

    fig.savefig(savefile, bbox_inches="tight")


def corner_flow_2d(
    angles,
    indices,
    r_vals,
    θ_vals,
    timestep,
    savefile="pydrex_corner_flow_2d.png",
    markers=("."),
    labels=None,
):
    """Plot diagnostics for 2D corner flow tests."""
    fig = plt.figure(figsize=(5, 12), dpi=300)
    grid = fig.add_gridspec(4, 1, hspace=0.05, height_ratios=(0.3, 0.3, 0.1, 0.3))
    ax_mean = fig.add_subplot(grid[0])
    ax_mean.set_ylabel("Mean angle ∈ [0, 90]°")
    ax_mean.axhline(0, color="k")
    ax_mean.tick_params(labelbottom=False)
    ax_strength = fig.add_subplot(grid[1], sharex=ax_mean)
    ax_strength.set_ylabel("Texture strength (M-index)")
    ax_strength.set_xlabel(r"Strain-time ($\dot{ε}_0 t$)")
    ax_lag = fig.add_subplot(grid[3])
    ax_lag.set_ylabel("z")
    ax_lag.set_xlabel("x")
    ax_lag.xaxis.set_ticks_position("top")
    ax_lag.xaxis.set_label_position("top")
    for axes in (ax_mean, ax_strength, ax_lag):
        axes.grid()

    if len(angles) % len(markers) != 0:
        raise ValueError(
            "Number of data series must be divisible by number of markers."
            + f" You've supplied {len(angles)} data series and {len(markers)} markers."
        )

    for i, (misorient_angles, misorient_indices, r_series, θ_series) in enumerate(
        zip(angles, indices, r_vals, θ_vals)
    ):
        timesteps = np.linspace(
            0, len(misorient_angles) * timestep, len(misorient_angles)
        )
        label = None
        if labels is not None:
            label = labels[int(i / (len(angles) / len(labels)))]
        markerseq = markers[int(i / (len(angles) / len(markers)))]

        ax_mean.plot(
            timesteps,
            misorient_angles,
            markerseq,
            markersize=5,
            alpha=0.33,
            label=label,
        )
        ax_strength.plot(
            timesteps,
            misorient_indices,
            markerseq,
            markersize=5,
            alpha=0.33,
            label=label,
        )
        r_series = np.asarray(r_series)
        θ_series = np.asarray(θ_series)
        ax_lag.plot(
            r_series * np.sin(θ_series),
            -r_series * np.cos(θ_series),
            markerseq,
            markersize=5,
            alpha=0.33,
            label=label,
        )
        ax_lag.set_xlim(0, 5)
        ax_lag.set_ylim(-1, 0)

    if labels is not None:
        ax_mean.legend()

    fig.savefig(savefile, bbox_inches="tight")
