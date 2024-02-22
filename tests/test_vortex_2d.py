"""> PyDRex: tests for CPO stability in 2D vortex and Stokes cell flows."""

import functools as ft
from multiprocessing import Pool

import numpy as np
import pytest

from pydrex import core as _core
from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import pathlines as _path
from pydrex import utils as _utils
from pydrex import velocity as _velocity
from pydrex import visualisation as _vis

# Subdirectory of `outdir` used to store outputs from these tests.
SUBDIR = "2d_vortex"


class TestCellOlivineA:
    """Tests for A-type olivine polycrystals in a 2D Stokes cell."""

    class_id = "cell_olivineA"
    _ensemble_n_grains = [100, 500, 1000, 5000, 10000]

    @classmethod
    def _make_ensemble_figure(cls, outdir):
        # Create the combined figure from outputs of the parametrized ensemble test.
        data = []
        out_basepath = f"{outdir}/{SUBDIR}/{cls.class_id}"
        try:
            for n_grains in cls._ensemble_n_grains:
                data.append(np.load(f"{out_basepath}_xz_ensemble_N{n_grains}_data.npz"))
        except FileNotFoundError:
            _log.debug(
                "skipping visualisation of 2D cell ensemble results (missing datafiles)"
            )
            return

        fig = _vis.figure()
        mosaic = fig.subplot_mosaic([["a)"], ["b)"]], sharex=True)
        fig, mosaic["a)"], colors = _vis.alignment(
            mosaic["a)"],
            np.asarray([d["strains"] for d in data]),
            np.asarray([d["angles_mean"] for d in data]),
            ("s", "o", "v", "*", "."),
            list(map(str, cls._ensemble_n_grains)),
            err=np.asarray([d["angles_err"] for d in data]),
        )
        fig, mosaic["b)"], colors = _vis.alignment(
            mosaic["b)"],
            np.asarray([d["strains"] for d in data]),
            np.asarray([d["max_sizes_mean"] for d in data]),
            ("s", "o", "v", "*", "."),
            list(map(str, cls._ensemble_n_grains)),
            err=np.asarray([d["max_sizes_err"] for d in data]),
            θ_max=4,
        )
        mosaic["b)"].set_ylabel(r"$\log_{10}(\overline{S}_{\mathrm{max}})$")

        mosaic["a)"].label_outer()
        mosaic["b)"].get_legend().remove()
        fig.savefig(
            _io.resolve_path(
                f"{outdir}/{SUBDIR}/{cls.class_id}_xz_ensemble_combined.pdf"
            )
        )

    @classmethod
    def run(
        cls,
        params,
        final_location,
        get_velocity,
        get_velocity_gradient,
        min_coords,
        max_coords,
        max_strain,
        seed=None,
    ):
        """Run 2D Stokes cell A-type olivine simulation."""
        mineral = _minerals.Mineral(
            phase=_core.MineralPhase.olivine,
            fabric=_core.MineralFabric.olivine_A,
            regime=_core.DeformationRegime.dislocation,
            n_grains=params["number_of_grains"],
            seed=seed,
        )
        deformation_gradient = np.eye(3)

        timestamps_back, get_position = _path.get_pathline(
            final_location,
            get_velocity,
            get_velocity_gradient,
            min_coords,
            max_coords,
            max_strain,
        )
        timestamps = np.linspace(
            timestamps_back[-1], timestamps_back[0], int(max_strain * 10)
        )
        positions = [get_position(t) for t in timestamps]
        velocity_gradients = [get_velocity_gradient(np.asarray(x)) for x in positions]
        strains = np.empty_like(timestamps)
        strains[0] = 0
        for t, time in enumerate(timestamps[:-1], start=1):
            strains[t] = strains[t - 1] + (
                _utils.strain_increment(timestamps[t] - time, velocity_gradients[t])
            )
            _log.info("step %d/%d (ε = %.2f)", t, len(timestamps) - 1, strains[t])

            deformation_gradient = mineral.update_orientations(
                params,
                deformation_gradient,
                get_velocity_gradient,
                pathline=(time, timestamps[t], get_position),
            )
        return timestamps, positions, strains, mineral, deformation_gradient

    @pytest.mark.big
    def test_xz_10k(self, outdir, seed):
        """Run 2D cell test with 10000 grains (~14GiB RAM requirement)."""
        self.test_xz(outdir, seed, 10000)

    @pytest.mark.parametrize("n_grains", [100, 500, 1000, 5000])
    def test_xz(self, outdir, seed, n_grains):
        """Test to check that 5000 grains is "enough" to resolve transient features."""
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_xz_N{n_grains}"

        params = _io.DEFAULT_PARAMS
        params["number_of_grains"] = n_grains
        get_velocity, get_velocity_gradient = _velocity.cell_2d("X", "Z", 1)

        timestamps, positions, strains, mineral, deformation_gradient = self.run(
            params,
            np.asarray([0.5, 0.0, -0.75]),
            get_velocity,
            get_velocity_gradient,
            np.asarray([-1, 0, -1]),
            np.asarray([1, 0, 1]),
            7,
            seed=seed,
        )
        angles = [
            _diagnostics.smallest_angle(
                _diagnostics.bingham_average(a, axis="a"), get_velocity(x)
            )
            for a, x in zip(mineral.orientations, positions, strict=True)
        ]
        if outdir is not None:
            # First figure with the domain and pathline.
            fig_path, ax_path, q, s = _vis.pathline_box2d(
                None,
                get_velocity,
                "XZ",
                strains,
                positions,
                ".",
                [-1, -1],
                [1, 1],
                [20, 20],
                cmap="cmc.batlow_r",
                tick_formatter=lambda x, pos: str(x),
            )
            fig_path.colorbar(s, ax=ax_path, aspect=25, label="Strain (ε)")
            fig_path.savefig(_io.resolve_path(f"{out_basepath}_path.pdf"))
            # Second figure with the angles and grain sizes at every 10 strain values.
            fig = _vis.figure()
            axθ = fig.add_subplot(2, 1, 1)
            fig, axθ, colors = _vis.alignment(
                axθ,
                strains,
                angles,
                (".",),
                (None,),
                colors=[strains],
                cmaps=["cmc.batlow_r"],
            )
            ax_sizes = fig.add_subplot(2, 1, 2, sharex=axθ)
            fig, ax_sizes, parts = _vis.grainsizes(
                ax_sizes, strains[::10], mineral.fractions[::10]
            )
            axθ.label_outer()
            fig.savefig(_io.resolve_path(f"{out_basepath}.pdf"))

        # Some checks for when we should have "enough" grains.
        # Based on empirical model outputs, it seems like the dip at ε ≈ 3.75 is the
        # least sensitive feature to the random state (seed) so we will use that.
        if n_grains >= 5000:
            # Can we resolve the temporary alignment to below 20° at ε ≈ 3.75?
            mean_θ_in_dip = np.mean(angles[34:43])
            assert mean_θ_in_dip < 12, mean_θ_in_dip
            # Can we resolve corresponding dip in max grain size (normalized, log_10)?
            mean_size_in_dip = np.log10(
                np.mean([np.max(f) for f in mineral.fractions[34:43]]) * n_grains
            )
            assert 2 < mean_size_in_dip < 3, mean_size_in_dip
            # Can we resolve subsequent peak in max grain size (normalized, log_10)?
            max_size_post_dip = np.log10(
                np.max([np.max(f) for f in mineral.fractions[43:]]) * n_grains
            )
            assert max_size_post_dip > 3, max_size_post_dip

    @pytest.mark.slow
    @pytest.mark.parametrize("n_grains", [100, 500, 1000, 5000, 10000])
    def test_xz_ensemble(self, outdir, seeds_nearX45, ncpus, n_grains):
        """Test to demonstrate stability of the dip at ε ≈ 3.75 for 5000+ grains."""
        _seeds = seeds_nearX45
        n_seeds = len(_seeds)
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_xz_ensemble_N{n_grains}"

        params = _io.DEFAULT_PARAMS
        params["number_of_grains"] = n_grains
        get_velocity, get_velocity_gradient = _velocity.cell_2d("X", "Z", 1)

        _run = ft.partial(
            self.run,
            params,
            np.asarray([0.5, 0.0, -0.75]),
            get_velocity,
            get_velocity_gradient,
            np.asarray([-1, 0, -1]),
            np.asarray([1, 0, 1]),
            7,
        )
        angles = np.empty((n_seeds, 70))
        max_sizes = np.empty_like(angles)
        with Pool(processes=ncpus) as pool:
            for s, out in enumerate(pool.imap_unordered(_run, _seeds)):
                timestamps, positions, strains, mineral, deformation_gradient = out
                angles[s] = [
                    _diagnostics.smallest_angle(
                        _diagnostics.bingham_average(a, axis="a"), get_velocity(x)
                    )
                    for a, x in zip(mineral.orientations, positions, strict=True)
                ]
                max_sizes[s] = np.log10(np.max(mineral.fractions, axis=1) * n_grains)

        if outdir is not None:
            # Figure with the angles and max grain sizes (ensemble averages).
            fig = _vis.figure()
            axθ = fig.add_subplot(2, 1, 1)
            angles_mean = np.mean(angles, axis=0)
            angles_err = np.std(angles, axis=0)
            fig, axθ, colors = _vis.alignment(
                axθ,
                strains,
                angles_mean,
                (".",),
                (None,),
                err=angles_err,
            )
            ax_maxsize = fig.add_subplot(2, 1, 2, sharex=axθ)
            ax_maxsize.set_ylabel(r"$\log_{10}(\overline{S}_{\mathrm{max}})$")
            max_sizes_mean = np.mean(max_sizes, axis=0)
            ax_maxsize.plot(strains, max_sizes_mean, color=colors[0])
            max_sizes_err = np.std(max_sizes, axis=0)
            ax_maxsize.fill_between(
                strains,
                max_sizes_mean - max_sizes_err,
                max_sizes_mean + max_sizes_err,
                alpha=0.22,
                color=colors[0],
            )
            axθ.label_outer()
            fig.savefig(_io.resolve_path(f"{out_basepath}.pdf"))
            np.savez(
                _io.resolve_path(f"{out_basepath}_data.npz"),
                strains=strains,
                max_sizes_mean=max_sizes_mean,
                max_sizes_err=max_sizes_err,
                angles_mean=angles_mean,
                angles_err=angles_err,
            )
