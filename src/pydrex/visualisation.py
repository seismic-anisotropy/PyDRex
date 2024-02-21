"""> PyDRex: Visualisation functions for test outputs and examples."""

import numpy as np
from cmcrameri import cm as cmc
from matplotlib import projections as mproj
from matplotlib import pyplot as plt

from pydrex import axes as _axes
from pydrex import core as _core
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import utils as _utils

plt.rcParams["axes.grid"] = True
# Always draw grid behind everything else.
plt.rcParams["axes.axisbelow"] = True
# Always use constrained layout by default (modern version of tight layout).
plt.rcParams["figure.constrained_layout.use"] = True
# Use 300 DPI by default, NASA can keep their blurry images.
plt.rcParams["figure.dpi"] = 300
# Make sure we have the required matplotlib "projections" (really just Axes subclasses).
if "pydrex.polefigure" not in mproj.get_projection_names():
    _log.warning(
        "failed to find pydrex.polefigure projection; it should be registered in %s",
        _axes,
    )


def polefigures(
    orientations,
    ref_axes,
    i_range,
    density=False,
    savefile="polefigures.png",
    strains=None,
    **kwargs,
):
    """Plot pole figures of a series of (Nx3x3) orientation matrix stacks.

    Produces [100], [010] and [001] pole figures for (resampled) orientations.
    For the argument specification, check the output of `pydrex-polefigures --help`
    on the command line.

    """
    if len(orientations) != len(i_range):
        raise ValueError("mismatched length of 'orientations' and 'i_range'")
    if strains is not None and len(strains) != len(i_range):
        raise ValueError("mismatched length of 'strains'")
    n_orientations = len(orientations)
    fig = plt.figure(figsize=(n_orientations, 4), dpi=600)

    if len(i_range) == 1:
        grid = fig.add_gridspec(3, n_orientations, hspace=0, wspace=0.2)
        first_row = 0
    else:
        grid = fig.add_gridspec(
            4, n_orientations, height_ratios=((1, 3, 3, 3)), hspace=0, wspace=0.2
        )
        fig_strain = fig.add_subfigure(grid[0, :])
        first_row = 1
        ax_strain = fig_strain.add_subplot(111)

        if strains is None:
            fig_strain.suptitle(
                f"N ⋅ (max strain) / {i_range.stop}", x=0.5, y=0.85, fontsize="small"
            )
            ax_strain.set_xlim(
                (i_range.start - i_range.step / 2, i_range.stop - i_range.step / 2)
            )
            ax_strain.set_xticks(list(i_range))
        else:
            fig_strain.suptitle("strain (%)", x=0.5, y=0.85, fontsize="small")
            ax_strain.set_xticks(strains)
            ax_strain.set_xlim(
                (
                    strains[0] - strains[1] / 2,
                    strains[-1] + strains[1] / 2,
                )
            )

        ax_strain.set_frame_on(False)
        ax_strain.grid(False)
        ax_strain.yaxis.set_visible(False)
        ax_strain.xaxis.set_tick_params(labelsize="x-small", length=0)

    fig100 = fig.add_subfigure(
        grid[first_row, :], edgecolor=plt.rcParams["grid.color"], linewidth=1
    )
    fig100.suptitle("[100]", fontsize="small")
    fig010 = fig.add_subfigure(
        grid[first_row + 1, :], edgecolor=plt.rcParams["grid.color"], linewidth=1
    )
    fig010.suptitle("[010]", fontsize="small")
    fig001 = fig.add_subfigure(
        grid[first_row + 2, :], edgecolor=plt.rcParams["grid.color"], linewidth=1
    )
    fig001.suptitle("[001]", fontsize="small")
    for n, orientations in enumerate(orientations):
        ax100 = fig100.add_subplot(
            1, n_orientations, n + 1, projection="pydrex.polefigure"
        )
        pf100 = ax100.polefigure(
            orientations,
            hkl=[1, 0, 0],
            ref_axes=ref_axes,
            density=density,
            density_kwargs=kwargs,
        )
        ax010 = fig010.add_subplot(
            1, n_orientations, n + 1, projection="pydrex.polefigure"
        )
        pf010 = ax010.polefigure(
            orientations,
            hkl=[0, 1, 0],
            ref_axes=ref_axes,
            density=density,
            density_kwargs=kwargs,
        )
        ax001 = fig001.add_subplot(
            1, n_orientations, n + 1, projection="pydrex.polefigure"
        )
        pf001 = ax001.polefigure(
            orientations,
            hkl=[0, 0, 1],
            ref_axes=ref_axes,
            density=density,
            density_kwargs=kwargs,
        )
        if density:
            for ax, pf in zip(
                (ax100, ax010, ax001), (pf100, pf010, pf001), strict=True
            ):
                cbar = fig.colorbar(
                    pf,
                    ax=ax,
                    fraction=0.05,
                    location="bottom",
                    orientation="horizontal",
                )
                cbar.ax.xaxis.set_tick_params(labelsize="xx-small")

    fig.savefig(_io.resolve_path(savefile))


def pathline_box2d(
    ax,
    get_velocity,
    ref_axes,
    colors,
    positions,
    marker,
    min_coords,
    max_coords,
    resolution,
    aspect="equal",
    cmap=cmc.batlow,
    cpo_vectors=None,
    cpo_strengths=None,
    tick_formatter=lambda x, pos: f"{x/1e3:.1f} km",
    **kwargs,
):
    """Plot pathlines and velocity arrows for a 2D box domain.

    If `ax` is None, a new figure and axes are created with `figure_unless`.

    Args:
    - `get_velocity` (callable) — object with call signature f(x) that returns
      the 3D velocity vector at a given 3D position vector
    - `ref_axes` (two letters from {"x", "y", "z"}) — labels for the horizontal and
      vertical axes (these also define the projection for the 3D velocity/position)
    - `colors` (array) — monotonic values along a representative pathline in the flow
    - `positions` (Nx3 array) — 3D position vectors along the same pathline
    - `min_coords` (array) — 2D coordinates of the lower left corner of the domain
    - `max_coords` (array) — 2D coordinates of the upper right corner of the domain
    - `resolution` (array) — 2D resolution of the velocity arrow grid (i.e. number of
      grid points in the horizontal and vertical directions) which can be set to None to
      prevent drawing velocity vectors
    - `aspect` (str|float, optional) — see `matplotlib.axes.Axes.set_aspect`
    - `cmap` (Matplotlib color map, optional) — color map for `colors`
    - `cpo_vectors` (array, optional) — vectors to plot as bars at pathline locations
    - `cpo_strengths` (array, optional) — strengths used to scale the cpo bars
    - `tick_formatter` (callable, optional) — function used to format tick labels

    Additional keyword arguments are passed to the `matplotlib.axes.Axes.quiver` call
    used to plot the velocity vectors.

    Returns the figure handle, the axes handle, the quiver collection (velocities) and
    the scatter collection (pathline).

    """
    fig, ax = figure_unless(ax)
    ax.set_xlabel(ref_axes[0])
    ax.set_ylabel(ref_axes[1])

    x_min, y_min = min_coords
    x_max, y_max = max_coords
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_aspect(aspect)
    ax.xaxis.set_major_formatter(tick_formatter)
    ax.yaxis.set_major_formatter(tick_formatter)

    _ref_axes = ref_axes.lower()
    axes_map = {"x": 0, "y": 1, "z": 2}
    horizontal = axes_map[_ref_axes[0]]
    vertical = axes_map[_ref_axes[1]]

    velocities = None
    if resolution is not None:
        x_res, y_res = resolution
        X = np.linspace(x_min, x_max, x_res)
        Y = np.linspace(y_min, y_max, y_res)
        X_grid, Y_grid = np.meshgrid(X, Y)

        U = np.zeros_like(X_grid.ravel())
        V = np.zeros_like(Y_grid.ravel())
        for i, (x, y) in enumerate(zip(X_grid.ravel(), Y_grid.ravel(), strict=True)):
            p = np.zeros(3)
            p[horizontal] = x
            p[vertical] = y
            v3d = get_velocity(p)
            U[i] = v3d[horizontal]
            V[i] = v3d[vertical]

        velocities = ax.quiver(
            X_grid,
            Y_grid,
            U.reshape(X_grid.shape),
            V.reshape(Y_grid.shape),
            pivot="mid",
            alpha=0.25,
            **kwargs,
        )

    P = np.asarray([[p[horizontal], p[vertical]] for p in positions])
    if cpo_vectors is not None:
        if cpo_strengths is None:
            cpo_strengths = np.full(len(cpo_vectors), 1.0)
        C = np.asarray(
            [
                f * np.asarray([c[horizontal], c[vertical]])
                for f, c in zip(cpo_strengths, cpo_vectors, strict=True)
            ]
        )
        cpo = ax.quiver(
            P[:, 0],
            P[:, 1],
            C[:, 0],
            C[:, 1],
            colors,
            cmap=cmap,
            pivot="mid",
            width=3e-3,
            headaxislength=0,
            headlength=0,
            zorder=10,
        )
    else:
        cpo = ax.scatter(P[:, 0], P[:, 1], marker=marker, c=colors, cmap=cmap)
    return fig, ax, velocities, cpo


def alignment(
    ax,
    strains,
    angles,
    markers,
    labels,
    err=None,
    θ_max=90,
    θ_fse=None,
    colors=None,
    cmaps=None,
):
    """Plot `angles` (in degrees) versus `strains` on the given axis.

    Alignment angles could be either bingham averages or the a-axis in the hexagonal
    symmetry projection, measured from e.g. the shear direction. In the first case,
    they should be calculated from resampled grain orientations. Expects as many
    `markers` and `labels` as there are data series in `angles`.

    If `ax` is None, a new figure and axes are created with `figure_unless`.

    Args:
    - `strains` (array) — X-values, accumulated strain (tensorial) during CPO evolution,
      may be a 2D array of multiple strain series
    - `angles` (array) — Y-values, may be a 2D array of multiple angle series
    - `markers` (sequence) — MatPlotLib markers to use for the data series
    - `labels` (sequence) — labels to use for the data series
    - `err` (array, optional) — standard errors for the `angles`, shapes must match
    - `θ_max` (int) — maximum angle (°) to show on the plot, should be less than 90
    - `θ_fse` (array, optional) — an array of angles from the long axis of the finite
      strain ellipsoid to the reference direction (e.g. shear direction)
    - `colors` (array, optional) — color coordinates for series of angles
    - `cmaps` (Matplotlib color maps, optional) — color maps for `colors`

    If `colors` and `cmaps` are used, then angle values are colored individually within
    each angle series.

    Returns a tuple of the figure handle, the axes handle and the set of colors used for
    the data series plots.

    """
    _strains = np.atleast_2d(strains)
    _angles = np.atleast_2d(angles)
    if err is not None:
        _angles_err = np.atleast_2d(err)
    if not np.all(_strains.shape == _angles.shape):
        # Assume strains are all the same for each series in `angles`, try np.tile().
        _strains = np.tile(_strains, (len(_angles), 1))

    fig, ax = figure_unless(ax)
    ax.set_ylabel("Mean angle ∈ [0, 90]°")
    ax.set_ylim((0, θ_max))
    ax.set_xlabel("Strain (ε = γ/2)")
    ax.set_xlim((np.min(strains), np.max(strains)))
    _colors = []
    for i, (strains, θ_cpo, marker, label) in enumerate(
        zip(_strains, _angles, markers, labels, strict=True)
    ):
        if colors is not None:
            ax.scatter(
                strains,
                θ_cpo,
                marker=marker,
                label=label,
                c=colors[i],
                cmap=cmaps[i],
                alpha=0.6,
                edgecolor=plt.rcParams["axes.edgecolor"],
            )
            _colors.append(colors[i])
        else:
            lines = ax.plot(
                strains, θ_cpo, marker, markersize=5, alpha=0.33, label=label
            )
            _colors.append(lines[0].get_color())
        if err is not None:
            ax.fill_between(
                strains,
                θ_cpo - _angles_err[i],
                θ_cpo + _angles_err[i],
                alpha=0.22,
                color=_colors[i],
            )

    if θ_fse is not None:
        ax.plot(strains, θ_fse, linestyle=(0, (5, 5)), alpha=0.6, label="FSE")
    if not all(b is None for b in labels):
        _utils.redraw_legend(ax)
    return fig, ax, _colors


def strengths(
    ax,
    strains,
    strengths,
    ylabel,
    markers,
    labels,
    err=None,
    cpo_threshold=None,
    colors=None,
    cmaps=None,
):
    """Plot CPO `strengths` (e.g. M-indices) versus `strains` on the given axis.

    If `ax` is None, a new figure and axes are created with `figure_unless`.

    Args:
    - `strains` (array) — X-values, accumulated strain (tensorial) during CPO evolution,
      may be a 2D array of multiple strain series
    - `strengths` (array) — Y-values, may be a 2D array of multiple strength series
    - `markers` (sequence) — MatPlotLib markers to use for the data series
    - `labels` (sequence) — labels to use for the data series
    - `err` (array, optional) — standard errors for the `strengths`, shapes must match
    - `colors` (array, optional) — color coordinates for series of strengths
    - `cpo_threshold` (float, optional) — plot a dashed line at this threshold
    - `cmaps` (Matplotlib color maps, optional) — color maps for `colors`

    If `colors` and `cmaps` are used, then strength values are colored individually
    within each strength series.

    Returns a tuple of the figure handle, the axes handle and the set of colors used for
    the data series plots.

    """
    _strains = np.atleast_2d(strains)
    _strengths = np.atleast_2d(strengths)
    if err is not None:
        _strengths_err = np.atleast_2d(err)
    if not np.all(_strains.shape == _strengths.shape):
        # Assume strains are all the same for each series in `strengths`, try np.tile().
        _strains = np.tile(_strains, (len(_strengths), 1))

    fig, ax = figure_unless(ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Strain (ε = γ/2)")
    ax.set_xlim((np.min(strains), np.max(strains)))

    if cpo_threshold is not None:
        ax.axhline(cpo_threshold, color=plt.rcParams["axes.edgecolor"], linestyle="--")

    _colors = []
    for i, (strains, strength, marker, label) in enumerate(
        zip(_strains, _strengths, markers, labels, strict=True)
    ):
        if colors is not None:
            ax.scatter(
                strains,
                strength,
                marker=marker,
                label=label,
                c=colors[i],
                cmap=cmaps[i],
                alpha=0.6,
                edgecolor=plt.rcParams["axes.edgecolor"],
            )
            _colors.append(colors[i])
        else:
            lines = ax.plot(
                strains, strength, marker, markersize=5, alpha=0.33, label=label
            )
            _colors.append(lines[0].get_color())
        if err is not None:
            ax.fill_between(
                strains,
                strength - _strengths_err[i],
                strength + _strengths_err[i],
                alpha=0.22,
                color=_colors[i],
            )

    if not all(b is None for b in labels):
        _utils.redraw_legend(ax)
    return fig, ax, _colors


def grainsizes(ax, strains, fractions):
    """Plot grain volume `fractions` versus `strains` on the given axis.

    If `ax` is None, a new figure and axes are created with `figure_unless`.

    """
    n_grains = len(fractions[0])
    fig, ax = figure_unless(ax)
    ax.set_ylabel(r"Normalized grain sizes ($log_{10}$)")
    ax.set_xlabel("Strain (ε = γ/2)")
    parts = ax.violinplot(
        [np.log10(f * n_grains) for f in fractions], positions=strains, widths=0.8
    )
    for part in parts["bodies"]:
        part.set_color("black")
        part.set_alpha(1)
    parts["cbars"].set_alpha(0)
    parts["cmins"].set_visible(False)
    parts["cmaxes"].set_color("red")
    parts["cmaxes"].set_alpha(0.5)
    return fig, ax, parts


def show_Skemer2016_ShearStrainAngles(
    ax, studies, markers, colors, fillstyles, labels, fabric
):
    """Show data from `src/pydrex/data/thirdparty/Skemer2016_ShearStrainAngles.scsv`.

    Plot data from the Skemer 2016 datafile on the axis given by `ax`. Select the
    studies from which to plot the data, which must be a list of strings with exact
    matches in the `study` column in the datafile.
    Also filter the data to select only the given `fabric`
    (see `pydrex.core.MineralFabric`).

    If `ax` is None, a new figure and axes are created with `figure_unless`.

    Returns a tuple of the figure handle, the axes handle and the set of colors used for
    the data series plots.

    """
    fabric_map = {
        _core.MineralFabric.olivine_A: "A",
        _core.MineralFabric.olivine_B: "B",
        _core.MineralFabric.olivine_C: "C",
        _core.MineralFabric.olivine_D: "D",
        _core.MineralFabric.olivine_E: "E",
    }
    fig, ax = figure_unless(ax)
    data_Skemer2016 = _io.read_scsv(
        _io.data("thirdparty") / "Skemer2016_ShearStrainAngles.scsv"
    )
    for study, marker, color, fillstyle, label in zip(
        studies, markers, colors, fillstyles, labels, strict=True
    ):
        # Note: np.nonzero returns a tuple.
        indices = np.nonzero(
            np.logical_and(
                np.asarray(data_Skemer2016.study) == study,
                np.asarray(data_Skemer2016.fabric) == fabric_map[fabric],
            )
        )[0]
        ax.plot(
            np.take(data_Skemer2016.shear_strain, indices) / 200,
            np.take(data_Skemer2016.angle, indices),
            marker=marker,
            fillstyle=fillstyle,
            linestyle="none",
            markersize=5,
            color=color,
            label=label,
        )
    if not all(b is None for b in labels):
        _utils.redraw_legend(ax)
    return fig, ax, colors


def spin(ax, initial_angles, rotation_rates, target_rotation_rates=None):
    """Plot rotation rates of grains with known, unique initial [100] angles from X.

    If `ax` is None, a new figure and axes are created with `figure_unless`.

    Returns a tuple of the figure handle, the axes handle and the set of colors used for
    the data series plots.

    """
    if len(initial_angles) != len(rotation_rates) or (
        target_rotation_rates is not None
        and len(target_rotation_rates) != len(rotation_rates)
    ):
        raise ValueError("mismatch in lengths of inputs")
    fig, ax = figure_unless(ax)
    ax.set_ylabel("rotation rate")
    ax.set_xlabel("initial [100] angle (°)")
    ax.set_xlim((0, 360))
    ax.set_xticks(np.linspace(0, 360, 9))
    colors = []
    if target_rotation_rates is not None:
        lines = ax.plot(
            initial_angles,
            target_rotation_rates,
            c="tab:orange",
            lw=1,
            label="target spins",
        )
        colors.append(lines[0].get_color())
    series = ax.scatter(
        initial_angles,
        rotation_rates,
        facecolors="none",
        edgecolors=plt.rcParams["axes.edgecolor"],
        s=8,
        lw=1,
        label="computed spins",
    )
    colors.append(series.get_edgecolors()[0])
    _utils.redraw_legend(ax)
    return fig, ax, colors


def growth(ax, initial_angles, fractions_diff, target_fractions_diff=None):
    if len(initial_angles) != len(fractions_diff) or (
        target_fractions_diff is not None
        and len(target_fractions_diff) != len(fractions_diff)
    ):
        raise ValueError("mismatch in lengths of inputs")
    fig, ax = figure_unless(ax)
    ax.set_ylabel("grain growth rate")
    ax.set_xlabel("initial [100] angle (°)")
    ax.set_xlim((0, 360))
    ax.set_xticks(np.linspace(0, 360, 9))
    colors = []
    if target_fractions_diff is not None:
        lines = ax.plot(
            initial_angles,
            target_fractions_diff,
            c="tab:orange",
            lw=1,
            label="target growth",
        )
        colors.append(lines[0].get_color())
    series = ax.scatter(
        initial_angles,
        fractions_diff,
        facecolors="none",
        edgecolors=plt.rcParams["axes.edgecolor"],
        s=8,
        lw=1,
        label="computed growth",
    )
    colors.append(series.get_edgecolors()[0])
    _utils.redraw_legend(ax)
    return fig, ax, colors


def figure_unless(ax):
    """Create figure and axes if `ax` is None, or return existing figure for `ax`.

    If `ax` is None, a new figure is created for the axes with a few opinionated default
    settings (grid, constrained layout, high DPI).

    Returns a tuple containing the figure handle and the axes object.

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig = ax.get_figure()
    return fig, ax


def figure(**kwargs):
    """Create new figure with a few opinionated default settings.

    (e.g. grid, constrained layout, high DPI).

    """
    return plt.figure(**kwargs)
