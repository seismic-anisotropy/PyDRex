"""> PyDRex: Visualisation functions for test outputs and examples."""

import numpy as np
from cmcrameri import cm as cmc
from matplotlib import projections as mproj
from matplotlib import pyplot as plt

from pydrex import axes as _axes
from pydrex import core as _core
from pydrex import geometry as _geo
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import utils as _utils

# Get default figure size for easy referencing and scaling.
DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT = plt.rcParams["figure.figsize"]
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


def default_tick_formatter(x, pos):
    return f"{x/1e3:.1f}"


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


def steady_box2d(
    ax: plt.Axes | None,
    velocity: tuple,
    geometry: tuple,
    ref_axes: str,
    cpo: tuple | None,
    colors,
    aspect="equal",
    cmap=cmc.batlow,
    marker=".",
    tick_formatter=default_tick_formatter,
    label_suffix="(km)",
    **kwargs,
) -> tuple:
    """Plot pathlines and steady-state velocity arrows for a 2D box domain.

    If `ax` is None, a new figure and axes are created with `figure_unless`.

    Args:
    - `velocity` — tuple containing a velocity callable¹ and the 2D resolution of the
      velocity arrow grid, e.g. [20, 20] for 20x20 arrows over the rectangular domain
    - `geometry` — tuple containing the array of 3D pathline positions and two 2D
      coordinates (of the lower-left and upper-right domain corners)
    - `ref_axes` — two letters from {"x", "y", "z"} used to label the horizontal and
      vertical axes (these also define the projection for the 3D velocity/position)
    - `cpo` — tuple containing one array of CPO strengths and one of 3D CPO vectors;
      alternatively set this to `None` and use `marker` to only plot pathline positions
    - `colors` — monotonic, increasing values along the pathline (e.g. time or strain)
    - `aspect` — optional, see `matplotlib.axes.Axes.set_aspect`
    - `cmap` — optional custom color map for `colors`
    - `marker` — optional pathline position marker used when `cpo` is `None`
    - `tick_formatter` — optional custom tick formatter callable
    - `label_suffix` — optional suffix added to the axes labels

    ¹with signature `f(t, x)` where `t` is not used and `x` is a 3D position vector

    Additional keyword arguments are passed to the `matplotlib.axes.Axes.quiver` call
    used to plot the velocity vectors.

    Returns the figure handle, the axes handle, the quiver collection (velocities) and
    the scatter collection (pathline).

    """
    fig, ax = figure_unless(ax)
    ax.set_xlabel(f"{ref_axes[0]} {label_suffix}")
    ax.set_ylabel(f"{ref_axes[1]} {label_suffix}")

    get_velocity, resolution = velocity
    positions, min_coords, max_coords = geometry
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
            v3d = get_velocity(np.nan, p)
            U[i] = v3d[horizontal]
            V[i] = v3d[vertical]

        velocities = ax.quiver(
            X_grid,
            Y_grid,
            U.reshape(X_grid.shape),
            V.reshape(Y_grid.shape),
            pivot=kwargs.pop("pivot", "mid"),
            alpha=kwargs.pop("alpha", 0.25),
            **kwargs,
        )

    dummy_dim = ({0, 1, 2} - set(_geo.to_indices2d(*ref_axes))).pop()
    xi_2D = np.asarray([_utils.remove_dim(p, dummy_dim) for p in positions])
    qcoll: plt.Quiver | plt.PathCollection
    if cpo is None:
        qcoll = ax.scatter(xi_2D[:, 0], xi_2D[:, 1], marker=marker, c=colors, cmap=cmap)
    else:
        cpo_strengths, cpo_vectors = cpo
        cpo_2D = np.asarray([
            s * _utils.remove_dim(v, dummy_dim)
            for s, v in zip(cpo_strengths, cpo_vectors, strict=True)
        ])
        qcoll = ax.quiver(
            xi_2D[:, 0],
            xi_2D[:, 1],
            cpo_2D[:, 0],
            cpo_2D[:, 1],
            colors,
            cmap=cmap,
            pivot="mid",
            width=kwargs.pop("width", 3e-3),
            headaxislength=0,
            headlength=0,
            zorder=kwargs.pop("zorder", 10) + 1,  # Always above velocity vectors.
        )
    return fig, ax, velocities, qcoll


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
    **kwargs,
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

    Additional keyword arguments are passed to `matplotlib.axes.Axes.scatter` if
    `colors` is not `None`, or to `matplotlib.axes.Axes.plot` otherwise.

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
    ax.set_ylabel(r"$\overline{θ}$ ∈ [0, 90]°")
    ax.set_ylim((0, θ_max))
    ax.set_xlabel("Strain (ε)")
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
                alpha=kwargs.pop("alpha", 0.6),
                edgecolor=kwargs.pop("edgecolor", plt.rcParams["axes.edgecolor"]),
                **kwargs,
            )
            _colors.append(colors[i])
        else:
            lines = ax.plot(strains, θ_cpo, marker, alpha=0.6, label=label, **kwargs)
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
    **kwargs,
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

    Additional keyword arguments are passed to `matplotlib.axes.Axes.scatter` if
    `colors` is not `None`, or to `matplotlib.axes.Axes.plot` otherwise.

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
    ax.set_xlabel("Strain (ε)")
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
                alpha=kwargs.pop("alpha", 0.6),
                edgecolor=kwargs.pop("edgecolor", plt.rcParams["axes.edgecolor"]),
                **kwargs,
            )
            _colors.append(colors[i])
        else:
            lines = ax.plot(
                strains, strength, marker, alpha=0.33, label=label, **kwargs
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
    ax.set_ylabel(r"$\log_{10}(f × N)$")
    ax.set_xlabel("Strain (ε)")
    parts = ax.violinplot(
        [np.log10(f * n_grains) for f in fractions], positions=strains, widths=0.8
    )
    for part in parts["bodies"]:
        part.set_color("black")
        part.set_alpha(1)
    parts["cbars"].set_alpha(0)
    parts["cmins"].set_visible(False)
    parts["cmaxes"].set_visible(False)
    # parts["cmaxes"].set_color("red")
    # parts["cmaxes"].set_alpha(0.5)
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

    Returns a tuple containing:
    - the figure handle
    - the axes handle
    - the set of colors used for the data series plots
    - the Skemer 2016 dataset
    - the indices used to select data according to the "studies" and "fabric" filters

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
            color=color,
            label=label,
        )
    if not all(b is None for b in labels):
        _utils.redraw_legend(ax)
    return fig, ax, colors, data_Skemer2016, indices


def spin(
    ax,
    initial_angles,
    rotation_rates,
    target_initial_angles=None,
    target_rotation_rates=None,
    labels=("target", "computed"),
    shear_axis=None,
):
    """Plot rotation rates of grains with known, unique initial [100] angles from X.

    If `ax` is None, a new figure and axes are created with `figure_unless`.
    The default labels ("target", "computed") can also be overriden.
    If `shear_axis` is not None, a dashed line will be drawn at the given x-value
    (and its reflection around 180°).

    Returns a tuple of the figure handle, the axes handle and the set of colors used for
    the data series plots.

    """
    fig, ax = figure_unless(ax)
    ax.set_ylabel("Rotation rate (°/s)")
    ax.set_xlabel("Initial [100] angle (°)")
    ax.set_xlim((0, 360))
    ax.set_xticks(np.linspace(0, 360, 9))
    if shear_axis is not None:
        ax.axvline(shear_axis, color="k", linestyle="--", alpha=0.5)
        ax.axvline(
            (shear_axis + 180) % 360,
            color="k",
            linestyle="--",
            alpha=0.5,
            label="shear axis",
        )
    colors = []
    if target_rotation_rates is not None:
        lines = ax.plot(
            target_initial_angles,
            target_rotation_rates,
            c="tab:orange",
            label=labels[0],
        )
        colors.append(lines[0].get_color())
    series = ax.scatter(
        initial_angles,
        rotation_rates,
        facecolors="none",
        edgecolors=plt.rcParams["axes.edgecolor"],
        label=labels[1],
    )
    colors.append(series.get_edgecolors()[0])
    _utils.redraw_legend(ax)
    return fig, ax, colors


def growth(
    ax,
    initial_angles,
    fractions_diff,
    target_initial_angles=None,
    target_fractions_diff=None,
    labels=("target", "computed"),
    shear_axis=None,
):
    """Plot grain growth of grains with known, unique initial [100] angles from X.

    If `ax` is None, a new figure and axes are created with `figure_unless`.
    The default labels ("target", "computed") can also be overriden.
    If `shear_axis` is not None, a dashed line will be drawn at the given x-value
    (and its reflection around 180°).

    Returns a tuple of the figure handle, the axes handle and the set of colors used for
    the data series plots.

    """
    fig, ax = figure_unless(ax)
    ax.set_ylabel("Grain growth rate (s⁻¹)")
    ax.set_xlabel("Initial [100] angle (°)")
    ax.set_xlim((0, 360))
    ax.set_xticks(np.linspace(0, 360, 9))
    if shear_axis is not None:
        ax.axvline(shear_axis, color="k", linestyle="--", alpha=0.5)
        ax.axvline(
            (shear_axis + 180) % 360,
            color="k",
            linestyle="--",
            alpha=0.5,
            label="shear axis",
        )
    colors = []
    if target_fractions_diff is not None:
        lines = ax.plot(
            target_initial_angles,
            target_fractions_diff,
            c="tab:orange",
            label=labels[0],
        )
        colors.append(lines[0].get_color())
    series = ax.scatter(
        initial_angles,
        fractions_diff,
        facecolors="none",
        edgecolors=plt.rcParams["axes.edgecolor"],
        label=labels[1],
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


def figure(figscale=None, **kwargs):
    """Create new figure with a few opinionated default settings.

    (e.g. grid, constrained layout, high DPI).

    The keyword argument `figscale` can be used to scale the figure width and height
    relative to the default values by passing a tuple. Any additional keyword arguments
    are passed to `matplotlib.pyplot.figure()`.

    """
    # NOTE: Opinionated defaults are set using rcParams at the top of this file.
    _figsize = kwargs.pop("figsize", (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT))
    if figscale is not None:
        _figsize = (DEFAULT_FIG_WIDTH * figscale[0], DEFAULT_FIG_HEIGHT * figscale[1])
    return plt.figure(figsize=_figsize, **kwargs)
