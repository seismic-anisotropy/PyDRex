"""> PyDRex: Simple shear 3D tests."""

import contextlib as cl
import functools as ft
from time import process_time

import numpy as np
import pytest

from pydrex import core as _core
from pydrex import diagnostics as _diagnostics
from pydrex import geometry as _geo
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import stats as _stats
from pydrex import utils as _utils
from pydrex import velocity as _velocity

Pool, HAS_RAY = _utils.import_proc_pool()
if HAS_RAY:
    import ray  # noqa: F401

    from pydrex import distributed as _dstr  # noqa: F401

# Subdirectory of `outdir` used to store outputs from these tests.
SUBDIR = "3d_simple_shear"


class TestFraters2021:
    """Tests inspired by the benchmarks presented in [Fraters & Billen, 2021].

    [Fraters & Billen, 2021]: https://doi.org/10.1029/2021GC009846

    """

    class_id = "Fraters2021"

    @classmethod
    def run(
        cls,
        params,
        timestamps,
        get_velocity_gradient_initial,
        get_velocity_gradient_final,
        switch_time,
        msg,
        seed=None,
    ):
        """Run simulation with stationary particles in the given velocity gradient.

        The optional RNG `seed` is used for the initial pseudorandom orientations.
        A prefix `msg` will be printed before each timestep log message if given.
        Other keyword args are passed to `pydrex.Mineral.update_orientations`.

        Returns a tuple containing one olivine (A-type) and one enstatite mineral.
        If `params.enstatite_fraction` is zero, then the second tuple element will be
        `None` instead.

        """

        def get_position(t):
            return np.full(3, np.nan)

        olivine = _minerals.Mineral(
            phase=_core.MineralPhase.olivine,
            fabric=_core.MineralFabric.olivine_A,
            regime=_core.DeformationRegime.matrix_dislocation,
            n_grains=params.number_of_grains,
            seed=seed,
        )
        if params.enstatite_fraction > 0:
            enstatite = _minerals.Mineral(
                phase=_core.MineralPhase.enstatite,
                fabric=_core.MineralFabric.enstatite_AB,
                regime=_core.DeformationRegime.matrix_dislocation,
                n_grains=params.number_of_grains,
                seed=seed,
            )
        else:
            enstatite = None
        deformation_gradient = np.eye(3)  # Undeformed initial state.

        for t, time in enumerate(timestamps[:-1], start=1):
            _log.info(
                "%s; # %d; step %d/%d (t = %s)", msg, seed, t, len(timestamps) - 1, time
            )
            if time > switch_time:
                get_velocity_gradient = get_velocity_gradient_final
            else:
                get_velocity_gradient = get_velocity_gradient_initial

            if params.enstatite_fraction > 0:
                enstatite.update_orientations(
                    params,
                    deformation_gradient,
                    get_velocity_gradient,
                    pathline=(time, timestamps[t], get_position),
                )
            deformation_gradient = olivine.update_orientations(
                params,
                deformation_gradient,
                get_velocity_gradient,
                pathline=(time, timestamps[t], get_position),
            )
            _log.debug(
                "› velocity gradient = %s",
                get_velocity_gradient(np.nan, np.full(3, np.nan)).flatten(),
            )
        return olivine, enstatite

    @pytest.mark.slow
    @pytest.mark.parametrize("switch_time_Ma", [0, 1, 2.5, np.inf])
    def test_direction_change(
        self, outdir, seeds, mock, switch_time_Ma, ncpus, ray_session
    ):
        """Test a-axis alignment in simple shear with instantaneous geometry change.

        The simulation runs for 5 Ma with a strain rate of 1.58e-14/s, resulting in an
        accumulated strain invariant of 2.5.

        The initial shear has nonzero du/dz and the final shear has nonzero dv/dx where
        u is the velocity along X and v the velocity along Y.

        """
        # Strain rate in units of strain per year, avoids tiny numbers ∼1e-14.
        strain_rate = 5e-7
        # With 500 steps, each step is ~3e11 seconds which is about as small as
        # geodynamic modelling could feasibly go in most cases.
        n_timestamps = 500
        # Solve until D₀t=2.5 ('shear' γ=5).
        timestamps = np.linspace(0, 5e6, n_timestamps)
        _seeds = seeds[:500]  # 500 runs as per Fraters & Billen, 2021, Figure 5.
        n_seeds = len(_seeds)

        _, get_velocity_gradient_initial = _velocity.simple_shear_2d(
            "X", "Z", strain_rate
        )
        _, get_velocity_gradient_final = _velocity.simple_shear_2d(
            "Y", "X", strain_rate
        )

        # Output arrays for mean [100] angles.
        olA_from_proj_XZ = np.empty((n_seeds, n_timestamps))
        olA_from_proj_YX = np.empty_like(olA_from_proj_XZ)
        ens_from_proj_XZ = np.empty_like(olA_from_proj_XZ)
        ens_from_proj_YX = np.empty_like(olA_from_proj_XZ)
        # Output arrays for M-index (CPO strength).
        olA_strength = np.empty_like(olA_from_proj_XZ)
        ens_strength = np.empty_like(olA_from_proj_XZ)

        _id = _io.stringify(switch_time_Ma * 1e6)
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_direction_change_{_id}"
            optional_logging = _io.logfile_enable(f"{out_basepath}.log")

        with optional_logging:
            clock_start = process_time()
            _run = ft.partial(
                self.run,
                mock.ParamsFraters2021(),
                timestamps,
                get_velocity_gradient_initial,
                get_velocity_gradient_final,
                switch_time_Ma * 1e6,
                _id,
            )
            with Pool(processes=ncpus) as pool:
                for s, out in enumerate(pool.imap_unordered(_run, _seeds)):
                    olivine, enstatite = out
                    _log.info("%s; # %d; postprocessing olivine...", _id, _seeds[s])
                    olA_resampled, _ = _stats.resample_orientations(
                        olivine.orientations, olivine.fractions, seed=_seeds[s]
                    )
                    olA_mean_vectors = np.array(
                        [
                            _diagnostics.bingham_average(dcm, axis="a")
                            for dcm in olA_resampled
                        ]
                    )
                    olA_from_proj_XZ[s, :] = np.array(
                        [
                            _diagnostics.smallest_angle(v, v - v * [0, 1, 0])
                            for v in olA_mean_vectors
                        ]
                    )
                    olA_from_proj_YX[s, :] = np.array(
                        [
                            _diagnostics.smallest_angle(v, v - v * [0, 0, 1])
                            for v in olA_mean_vectors
                        ]
                    )
                    olA_downsampled, _ = _stats.resample_orientations(
                        olivine.orientations,
                        olivine.fractions,
                        seed=_seeds[s],
                        n_samples=1000,
                    )
                    olA_strength[s, :] = _diagnostics.misorientation_indices(
                        olA_downsampled, _geo.LatticeSystem.orthorhombic, pool=pool
                    )

                    del olivine, olA_resampled, olA_mean_vectors

                    _log.info("%s; # %d; postprocessing enstatite...", _id, _seeds[s])
                    ens_resampled, _ = _stats.resample_orientations(
                        enstatite.orientations, enstatite.fractions, seed=_seeds[s]
                    )
                    ens_mean_vectors = np.array(
                        [
                            _diagnostics.bingham_average(dcm, axis="a")
                            for dcm in ens_resampled
                        ]
                    )
                    ens_from_proj_XZ[s, :] = np.array(
                        [
                            _diagnostics.smallest_angle(v, v - v * [0, 1, 0])
                            for v in ens_mean_vectors
                        ]
                    )
                    ens_from_proj_YX[s, :] = np.array(
                        [
                            _diagnostics.smallest_angle(v, v - v * [0, 0, 1])
                            for v in ens_mean_vectors
                        ]
                    )
                    ens_downsampled, _ = _stats.resample_orientations(
                        enstatite.orientations,
                        enstatite.fractions,
                        seed=_seeds[s],
                        n_samples=1000,
                    )
                    ens_strength[s, :] = _diagnostics.misorientation_indices(
                        ens_downsampled, _geo.LatticeSystem.orthorhombic, pool=pool
                    )
                    del enstatite, ens_resampled, ens_mean_vectors

            _log.info("elapsed CPU time: %s", np.abs(process_time() - clock_start))
            _log.info("calculating ensemble averages...")
            olA_from_proj_XZ_mean = olA_from_proj_XZ.mean(axis=0)
            olA_from_proj_XZ_err = olA_from_proj_XZ.std(axis=0)

            olA_from_proj_YX_mean = olA_from_proj_YX.mean(axis=0)
            olA_from_proj_YX_err = olA_from_proj_YX.std(axis=0)

            ens_from_proj_XZ_mean = ens_from_proj_XZ.mean(axis=0)
            ens_from_proj_XZ_err = ens_from_proj_XZ.std(axis=0)

            ens_from_proj_YX_mean = ens_from_proj_YX.mean(axis=0)
            ens_from_proj_YX_err = ens_from_proj_YX.std(axis=0)

            if outdir is not None:
                np.savez(
                    f"{out_basepath}.npz",
                    olA_from_proj_XZ_mean=olA_from_proj_XZ_mean,
                    olA_from_proj_XZ_err=olA_from_proj_XZ_err,
                    olA_from_proj_YX_mean=olA_from_proj_YX_mean,
                    olA_from_proj_YX_err=olA_from_proj_YX_err,
                    ens_from_proj_XZ_mean=ens_from_proj_XZ_mean,
                    ens_from_proj_XZ_err=ens_from_proj_XZ_err,
                    ens_from_proj_YX_mean=ens_from_proj_YX_mean,
                    ens_from_proj_YX_err=ens_from_proj_YX_err,
                )
                np.savez(
                    f"{out_basepath}_strength.npz",
                    olA_mean=olA_strength.mean(axis=0),
                    ens_mean=ens_strength.mean(axis=0),
                    olA_err=olA_strength.std(axis=0),
                    ens_err=ens_strength.std(axis=0),
                )
