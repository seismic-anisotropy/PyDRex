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

    ax_mean.plot(timesteps, 45 * np.exp(timesteps * (np.cos(np.pi) - 1)), "r--")

    if labels is not None:
        ax_mean.legend()

    fig.savefig(savefile, bbox_inches="tight")
