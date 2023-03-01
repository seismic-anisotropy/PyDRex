#!/usr/bin/env python3
import argparse

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from DRexParam import dim, gridCoords, name
from numpy import linspace, load


def makeFilledContourPlot(
    mplAxis, xCoords, yCoords, data, levels, cmap, cbarTicks, cbarLab
):
    ctrf = mplAxis.contourf(
        xCoords, yCoords, data, antialiased=True, levels=levels, cmap=cmap
    )
    for iso in ctrf.collections:
        iso.set_edgecolor("face")
    mplAxis.set_aspect("equal", adjustable="box")
    mplAxis.set_ylabel(
        "Depth (km)",
        fontsize=18,
        fontweight="semibold",
        labelpad=20,
        color="xkcd:white",
        bbox=dict(boxstyle="round", facecolor="xkcd:black"),
    )
    mplAxis.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
    mplAxis.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    mplAxis.tick_params(which="major", length=7, labelsize=16, width=2)
    mplAxis.tick_params(which="minor", length=4, width=2, color="xkcd:bright red")
    cbar = fig.colorbar(ctrf, ax=mplAxis, ticks=cbarTicks, fraction=0.05, aspect=10)
    cbar.set_label(
        cbarLab,
        labelpad=-25,
        fontweight="semibold",
        fontsize=18,
        color="xkcd:white",
        path_effects=[
            path_effects.Stroke(linewidth=3, foreground="black"),
            path_effects.Normal(),
        ],
    )
    cbar.ax.tick_params(which="major", labelsize=16, width=2, length=6)
    cbar.ax.tick_params(
        which="minor", labelsize=16, width=2, length=3, color="xkcd:bright red"
    )
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.set_ticks_position("right")
    return mplAxis


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="""
A simple plotting script using Matplotlib to generate filled contour plots.
Requires the input file DRexParam.py in the same directory.""",
)
parser.add_argument("input", help="input file (expects a Numpy NpzFile)")
args = parser.parse_args()

DRexArr = load(args.input)

fig, (ax, bx) = plt.subplots(
    nrows=2, ncols=1, sharex=True, sharey=True, num=0, figsize=(20, 10)
)
ax = makeFilledContourPlot(
    ax,
    gridCoords[0][:-1] / 1e3,
    gridCoords[-1][:-1] / 1e3,
    DRexArr["percani"] if dim == 2 else DRexArr["percani"][:, :, 0],
    linspace(0, 7, 22),
    "inferno",
    linspace(0, 7, 8),
    "Azimuthal Anisotropy (%)",
)
bx = makeFilledContourPlot(
    bx,
    gridCoords[0][:-1] / 1e3,
    gridCoords[-1][:-1] / 1e3,
    DRexArr["radani"] if dim == 2 else DRexArr["radani"][:, :, 0],
    linspace(0.85, 1.15, 31),
    "RdBu",
    linspace(0.85, 1.15, 7),
    "Radial Anisotropy",
)

ax.invert_yaxis()
bx.set_xlabel(
    "Y Axis (km)",
    fontsize=18,
    fontweight="semibold",
    labelpad=15,
    color="xkcd:white",
    bbox=dict(boxstyle="round", facecolor="xkcd:black"),
)
fig.tight_layout()
fig.savefig(f"PyDRex{dim}D_{name}_Figure.pdf", bbox_inches="tight", dpi=500)
