"""> PyDRex: Visualisation functions for test outputs and examples."""
import numpy as np
from matplotlib import projections as mproj
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pydrex import axes as _axes
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import utils as _utils

# Always show XY grid by default.
plt.rcParams["axes.grid"] = True
# Always use constrained layout by default (modern version of tight layout).
plt.rcParams["figure.constrained_layout.use"] = True
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
            ax_strain.set_xticks(strains[i_range.start : i_range.stop : i_range.step])
            ax_strain.set_xlim(
                (
                    strains[i_range.start] - strains[i_range.step] / 2,
                    strains[i_range.stop - i_range.step] + strains[i_range.step] / 2,
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
            for ax, pf in zip((ax100, ax010, ax001), (pf100, pf010, pf001)):
                cbar = fig.colorbar(
                    pf,
                    ax=ax,
                    fraction=0.05,
                    location="bottom",
                    orientation="horizontal",
                )
                cbar.ax.xaxis.set_tick_params(labelsize="xx-small")

    fig.savefig(_io.resolve_path(savefile))


def _get_marker_and_label(data, seq_index, markers, labels=None):
    marker = markers[int(seq_index / (len(data) / len(markers)))]
    label = None
    if labels is not None:
        label = labels[int(seq_index / (len(data) / len(labels)))]
    return marker, label


def simple_shear_stationary_2d(
    strains,
    angles,
    point100_symmetry,
    target_angles=None,
    angles_err=None,
    savefile="pydrex_simple_shear_stationary_2d.png",
    markers=("."),
    θ_fse=None,
    labels=None,
    a_type=True,
):
    """Plot diagnostics for stationary A-type olivine 2D simple shear box tests."""
    fig = plt.figure(figsize=(5, 8), dpi=300)
    grid = fig.add_gridspec(2, 1, hspace=0.05)
    ax_mean = fig.add_subplot(grid[0])
    ax_mean.set_ylabel("Mean angle ∈ [0, 90]°")
    ax_mean.tick_params(labelbottom=False)
    ax_mean.set_xlim((strains[0], strains[-1]))
    ax_mean.set_ylim((0, 60))
    ax_symmetry = fig.add_subplot(grid[1], sharex=ax_mean)
    ax_symmetry.set_xlim((strains[0], strains[-1]))
    ax_symmetry.set_ylim((0, 1))
    ax_symmetry.set_ylabel(r"Texture symmetry ($P_{[100]}$)")
    ax_symmetry.set_xlabel(r"Strain ($D_0 t = γ/2$)")

    angles = np.atleast_2d(angles)
    point100_symmetry = np.atleast_2d(point100_symmetry)
    if target_angles is None:
        target_angles = [None] * len(angles)
    for i, (θ_target, θ, point100) in enumerate(
        zip(target_angles, angles, point100_symmetry)
    ):
        marker, label = _get_marker_and_label(angles, i, markers, labels)

        lines = ax_mean.plot(strains, θ, marker, markersize=5, alpha=0.33)
        color = lines[0].get_color()
        if θ_target is not None:
            lines = ax_mean.plot(
                strains, θ_target, alpha=0.66, label=label, color=color
            )
        if angles_err is not None:
            ax_mean.fill_between(
                strains, θ - angles_err[i], θ + angles_err[i], alpha=0.22, color=color
            )
        ax_symmetry.plot(
            strains,
            point100,
            marker,
            markersize=5,
            alpha=0.33,
            color=color,
            label=label,
        )

    if a_type:
        data_Skemer2016 = _io.read_scsv(
            _io.data("thirdparty") / "Skemer2016_ShearStrainAngles.scsv"
        )
        indices_ZK1200 = np.nonzero(np.asarray(data_Skemer2016.study) == "Z&K 1200 C")[
            0  # Note: np.nonzero returns a tuple.
        ]
        ax_mean.plot(
            np.take(data_Skemer2016.shear_strain, indices_ZK1200) / 200,
            np.take(data_Skemer2016.angle, indices_ZK1200),
            marker="v",
            fillstyle="none",
            linestyle="none",
            markersize=5,
            color="k",
            label="Zhang & Karato, 1995\n(1200°C)",
        )
        indices_ZK1300 = np.nonzero(np.asarray(data_Skemer2016.study) == "Z&K 1300 C")[
            0  # Note: np.nonzero returns a tuple.
        ]
        ax_mean.plot(
            np.take(data_Skemer2016.shear_strain, indices_ZK1300) / 200,
            np.take(data_Skemer2016.angle, indices_ZK1300),
            marker="^",
            linestyle="none",
            markersize=5,
            color="k",
            label="Zhang & Karato, 1995\n(1300°C)",
        )
    if θ_fse is not None:
        ax_mean.plot(strains, θ_fse, linestyle=(0, (5, 5)), alpha=0.66, label="FSE")
    if labels is not None:
        ax_mean.legend()
        ax_symmetry.legend()

    fig.savefig(_io.resolve_path(savefile))


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
    """Plot diagnostics for prescribed path 2D corner flow tests.

    .. warning:: This method is in need of repair.

    """
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
        Π_series = _utils.lag_2d_corner_flow(np.arctan2(x_series, -z_series))
        Π_max_step = np.argmax(Π_series)
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
            color=plt.rcParams["axes.edgecolor"],
            zorder=11,
            alpha=0.33,
        )
        ax_strength.plot(
            t_series[corner_step],
            misorient_indices[corner_step],
            marker,
            markersize=5,
            color=plt.rcParams["axes.edgecolor"],
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
            color=plt.rcParams["axes.edgecolor"],
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
        _utils.lag_2d_corner_flow(θ_grid),
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
        cpo_threshold, color=plt.rcParams["axes.edgecolor"], linestyle="--"
    )
    if labels is not None:
        ax_mean.legend()

    fig.savefig(_io.resolve_path(savefile))


def single_olivineA_simple_shear(
    initial_angles,
    rotation_rates,
    target_rotation_rates,
    savefile="single_olivineA_simple_shear.png",
):
    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.subplots(nrows=1, ncols=1)
    ax.set_ylabel("rotation rate")
    ax.set_xlabel("initial angle (°)")
    ax.set_xlim((0, 360))
    ax.set_xticks(np.linspace(0, 360, 5))
    ax.plot(initial_angles, target_rotation_rates, c="tab:orange", lw=1)
    ax.scatter(
        initial_angles, rotation_rates, facecolors="none", edgecolors="k", s=8, lw=1
    )
    fig.savefig(_io.resolve_path(savefile))
