"""> PyDRex: tests for CPO stability in 2D vortex and Stokes cell flows."""

import contextlib as cl
import functools as ft
import sys

import numpy as np
import pytest
from numpy import testing as nt

from pydrex import core as _core
from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import pathlines as _path
from pydrex import stats as _stats
from pydrex import utils as _utils
from pydrex import velocity as _velocity
from pydrex import visualisation as _vis
from pydrex import geometry as _geo

Pool, HAS_RAY = _utils.import_proc_pool()

SUBDIR = "2d_vortex"
"""Subdirectory of `outdir` used to store outputs from these tests."""


def run_singlephase(params: dict, seed: int, assert_each=None, **kwargs) -> tuple:
    """Run 2D convection cell simulation for a single mineral phase.

    Uses A-type olivine by default.

    Args:
    - `params` — see `pydrex.core.DefaultParams`
    - `seed` — seed for random number generation
    - `assert_each` — optional callable with signature `f(mineral,
      deformation_gradient)` that performs assertions at each step

    Optional keyword args are consumed by:
    1. `pydrex.velocity.cell_2d` and
    2. the `pydrex.minerals.Mineral` constructor

    Returns a tuple containing:
    1. the `mineral` (instance of `pydrex.minerals.Mineral`)
    2. the resampled texture (a tuple of the orientations and fractions)
    3. a tuple of `fig, ax, q, s` as returned from `pydrex.visualisation.pathline_box2d`

    """
    horizontal: str = kwargs.pop("horizontal", "X")
    vertical: str = kwargs.pop("vertical", "Z")
    velocity_edge: float = kwargs.pop("velocity_edge", 6.342e-10)
    edge_length: float = kwargs.pop("edge_length", 2e5)

    max_strain = kwargs.pop(
        # This should be enough to go around the cell one time.
        "max_strain",
        int(np.ceil(velocity_edge * (edge_length / 2) ** 2)),
    )
    get_velocity, get_velocity_gradient = _velocity.cell_2d(
        horizontal,
        vertical,
        velocity_edge,
        edge_length,
    )
    mineral = _minerals.Mineral(
        phase=kwargs.pop("phase", _core.MineralPhase.olivine),
        fabric=kwargs.pop("fabric", _core.MineralFabric.olivine_A),
        regime=kwargs.pop("regime", _core.DeformationRegime.matrix_dislocation),
        n_grains=params["number_of_grains"],
        seed=seed,
        **kwargs,
    )

    size = edge_length / 2
    dummy_dim = ({0, 1, 2} - set(_geo.to_indices2d(horizontal, vertical))).pop()

    timestamps, get_position = _path.get_pathline(
        _utils.add_dim([0.5, -0.75], dummy_dim) * size,
        get_velocity,
        get_velocity_gradient,
        _utils.add_dim([-size, -size], dummy_dim),
        _utils.add_dim([size, size], dummy_dim),
        max_strain,
        regular_steps=max_strain * 10,
    )
    positions = [get_position(t) for t in timestamps]
    velocity_gradients = [  # Steady flow, time variable is np.nan.
        get_velocity_gradient(np.nan, np.asarray(x)) for x in positions
    ]

    strains = np.empty_like(timestamps)
    strains[0] = 0
    deformation_gradient = np.eye(3)

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
        if assert_each is not None:
            assert_each(mineral, deformation_gradient)

    orientations, fractions = _stats.resample_orientations(
        mineral.orientations, mineral.fractions, seed=seed
    )
    cpo_strengths =np.full(len(orientations), 1.0)
    cpo_vectors = [_diagnostics.bingham_average(o) for o in orientations]
    fig_path, ax_path, q, s = _vis.steady_box2d(
        None,
        (get_velocity, [20, 20]),
        (positions, [-size, -size], [size, size]),
        horizontal + vertical,
        (cpo_strengths, cpo_vectors),
        strains,
        cmap="cmc.batlow_r",
        aspect="equal",
        alpha=1,
    )
    fig_path.colorbar(s, ax=ax_path, aspect=25, label="Strain (ε)")

    return mineral, (orientations, fractions), (fig_path, ax_path, q, s)


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
            regime=_core.DeformationRegime.matrix_dislocation,
            n_grains=params["number_of_grains"],
            seed=seed,
        )
        deformation_gradient = np.eye(3)

        timestamps, get_position = _path.get_pathline(
            final_location,
            get_velocity,
            get_velocity_gradient,
            min_coords,
            max_coords,
            max_strain,
            regular_steps=int(max_strain * 10),
        )
        positions = [get_position(t) for t in timestamps]
        velocity_gradients = [
            get_velocity_gradient(np.nan, np.asarray(x)) for x in positions
        ]
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

    @pytest.mark.skipif(sys.platform == "win32", reason="Unable to allocate memory")
    @pytest.mark.parametrize("n_grains", [100, 500, 1000, 5000])
    def test_xz(self, outdir, seed, n_grains):
        """Test to check that 5000 grains is "enough" to resolve transient features."""
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_xz_N{n_grains}"

        params = _core.DefaultParams().as_dict()
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
                _diagnostics.bingham_average(a, axis="a"), get_velocity(np.nan, x)
            )
            for a, x in zip(mineral.orientations, positions, strict=True)
        ]
        if outdir is not None:
            # First figure with the domain and pathline.
            fig_path, ax_path, q, s = _vis.steady_box2d(
                None,
                (get_velocity, [20, 20]),
                (positions, [-1, -1], [1, 1]),
                "XZ",
                None,
                strains,
                cmap="cmc.batlow_r",
                tick_formatter=lambda x, pos: str(x),
                aspect="equal",
                alpha=1,
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

        params = _core.DefaultParams().as_dict()
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
                        _diagnostics.bingham_average(a, axis="a"),
                        get_velocity(np.nan, x),
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


class TestDiffusionCreep:
    """Tests for diffusion creep regime."""

    class_id = "diff_creep"

    def test_cell_olA(self, outdir, seed, ncpus, orientations_init_y):
        params = _core.DefaultParams().as_dict()
        params["gbm_mobility"] = 10

        def get_assert_each(i):  # The order of orientations_init_y is significant.
            @_utils.serializable
            def assert_each(mineral, deformation_gradient):
                # Check that surrogate grain sizes are not changing.
                nt.assert_allclose(
                    mineral.fractions[-1], mineral.fractions[-2], atol=1e-16, rtol=0
                )
                p, g, r = _diagnostics.symmetry_pgr(mineral.orientations[-1])
                nt.assert_allclose(
                    np.array([p, g, r]),
                    _diagnostics.symmetry_pgr(mineral.orientations[-2]),
                    atol=0.25,
                    rtol=0,
                )
                match i:
                    case 0:
                        # Check that symmetry remains mostly random.
                        assert r > 0.9, f"{r}"
                    case 1:
                        # Check that symmetry remains mostly girdled.
                        assert g > 0.9, f"{g}"
                    case 2:
                        # Check that symmetry remains mostly clustered.
                        assert p > 0.9, f"{g}"

            return assert_each

        @_utils.serializable
        def _run(assert_each, orientations_init):
            return run_singlephase(
                params,
                seed,
                regime=_core.DeformationRegime.matrix_diffusion,
                orientations_init=orientations_init,
            )

        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_olA"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")

        assert_each_list = [
            get_assert_each(i) for i, _ in enumerate(orientations_init_y)
        ]
        orientations_init_list = [
            f(params["number_of_grains"]) for f in orientations_init_y
        ]
        with optional_logging:
            with Pool(processes=ncpus) as pool:
                for i, out in enumerate(
                    pool.starmap(_run, zip(assert_each_list, orientations_init_list))
                ):
                    mineral, resampled_texture, fig_objects = out
                    fig_objects[0].savefig(
                        _io.resolve_path(f"{out_basepath}_path_{i}.pdf")
                    )
