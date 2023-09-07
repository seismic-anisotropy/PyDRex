"""> PyDRex: 2D simple shear tests."""
import contextlib as cl
import functools as ft
from multiprocessing import Pool
from time import process_time

import numpy as np
import pytest
from numpy import testing as nt
from scipy.interpolate import PchipInterpolator

from pydrex import core as _core
from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import stats as _stats
from pydrex import utils as _utils
from pydrex import velocity_gradients as _dv
from pydrex import visualisation as _vis

# Subdirectory of `outdir` used to store outputs from these tests.
SUBDIR = "2d_simple_shear"


class TestOlivineA:
    """Tests for stationary A-type olivine polycrystals in 2D simple shear."""

    class_id = "olivineA"

    @classmethod
    def get_position(cls, t):
        return np.zeros(3)  # These crystals are stationary.

    @classmethod
    def run(
        cls,
        params,
        timestamps,
        strain_rate,
        get_velocity_gradient,
        shear_direction,
        seed=None,
        return_fse=None,
    ):
        """Reusable logic for 2D olivine (A-type) simple shear tests.

        Returns a tuple with the mineral and the FSE angle (or `None` if `return_fse` is
        `None`).

        """
        mineral = _minerals.Mineral(
            phase=_core.MineralPhase.olivine,
            fabric=_core.MineralFabric.olivine_A,
            n_grains=params["number_of_grains"],
            seed=seed,
        )
        deformation_gradient = np.eye(3)  # Undeformed initial state.
        θ_fse = np.empty_like(timestamps)
        θ_fse[0] = 45

        for t, time in enumerate(timestamps[:-1], start=1):
            # Set up logging message depending on dynamic parameter and seeds.
            msg_start = (
                f"N = {params['number_of_grains']}; "
                + f"λ∗ = {params['nucleation_efficiency']}; "
                + f"X = {params['gbs_threshold']}; "
                + f"M∗ = {params['gbm_mobility']}; "
            )
            if seed is not None:
                msg_start += f"# {seed}; "

            _log.info(msg_start + "step %s/%s (t = %s)", t, len(timestamps) - 1, time)

            deformation_gradient = mineral.update_orientations(
                params,
                deformation_gradient,
                get_velocity_gradient,
                pathline=(time, timestamps[t], cls.get_position),
            )
            _log.debug(
                "› velocity gradient = %s",
                get_velocity_gradient(None).flatten(),
            )
            _log.debug("› strain D₀t = %.2f", strain_rate * timestamps[t])
            _log.debug(
                "› grain fractions: median = %s, max = %s, min = %s",
                np.median(mineral.fractions[-1]),
                np.max(mineral.fractions[-1]),
                np.min(mineral.fractions[-1]),
            )
            if return_fse:
                _, fse_v = _diagnostics.finite_strain(deformation_gradient)
                θ_fse[t] = _diagnostics.smallest_angle(fse_v, shear_direction)
            else:
                θ_fse = None

        return mineral, θ_fse

    @classmethod
    def interp_GBM_Kaminski2001(cls, strains):
        """Interpolate Kaminski & Ribe, 2001 data to get target angles at `strains`."""
        _log.info("interpolating target CPO angles...")
        data = _io.read_scsv(_io.data("thirdparty") / "Kaminski2001_GBMshear.scsv")
        cs_M0 = PchipInterpolator(
            _utils.remove_nans(data.equivalent_strain_M0) / 200,
            _utils.remove_nans(data.angle_M0),
        )
        cs_M50 = PchipInterpolator(
            _utils.remove_nans(data.equivalent_strain_M50) / 200,
            _utils.remove_nans(data.angle_M50),
        )
        cs_M200 = PchipInterpolator(
            _utils.remove_nans(data.equivalent_strain_M200) / 200,
            _utils.remove_nans(data.angle_M200),
        )
        return [cs_M0(strains), cs_M50(strains), cs_M200(strains)]

    @classmethod
    def interp_GBM_FortranDRex(cls, strains):
        """Interpolate angles produced using 'tools/drex_forward_simpleshear.f90'."""
        _log.info("interpolating target CPO  angles...")
        data = _io.read_scsv(_io.data("drexF90") / "olA_D1E4_dt50_X0_L5.scsv")
        data_strains = np.linspace(0, 1, 200)
        cs_M0 = PchipInterpolator(data_strains, _utils.remove_nans(data.M0_angle))
        cs_M50 = PchipInterpolator(data_strains, _utils.remove_nans(data.M50_angle))
        cs_M200 = PchipInterpolator(data_strains, _utils.remove_nans(data.M200_angle))
        return [cs_M0(strains), cs_M50(strains), cs_M200(strains)]

    @classmethod
    def interp_GBS_FortranDRex(cls, strains):
        """Interpolate angles produced using 'tools/drex_forward_simpleshear.f90'."""
        _log.info("interpolating target CPO  angles...")
        data = _io.read_scsv(_io.data("thirdparty") / "a_axis_GBS_fortran.scsv")
        data_strains = np.linspace(0, 1, 200)
        cs_X0 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X0))
        cs_X20 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X20))
        cs_X40 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X40))
        return [cs_X0(strains), cs_X20(strains), cs_X40(strains)]

    @classmethod
    def interp_GBS_long_FortranDRex(cls, strains):
        """Interpolate angles produced using 'tools/drex_forward_simpleshear.f90'."""
        _log.info("interpolating target CPO  angles...")
        data = _io.read_scsv(_io.data("thirdparty") / "a_axis_GBS_long_fortran.scsv")
        data_strains = np.linspace(0, 2.5, 500)
        cs_X0 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X0))
        cs_X20 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X20))
        cs_X40 = PchipInterpolator(data_strains, _utils.remove_nans(data.a_mean_X40))
        return [cs_X0(strains), cs_X20(strains), cs_X40(strains)]

    @classmethod
    def interp_GBS_Kaminski2004(cls, strains):
        """Interpolate Kaminski & Ribe, 2001 data to get target angles at `strains`."""
        _log.info("interpolating target CPO angles...")
        data = _io.read_scsv(_io.data("thirdparty") / "Kaminski2004_GBSshear.scsv")
        cs_X0 = PchipInterpolator(
            _utils.remove_nans(data.dimensionless_time_X0),
            45 + _utils.remove_nans(data.angle_X0),
        )
        cs_X0d2 = PchipInterpolator(
            _utils.remove_nans(data.dimensionless_time_X0d2),
            45 + _utils.remove_nans(data.angle_X0d2),
        )
        cs_X0d4 = PchipInterpolator(
            _utils.remove_nans(data.dimensionless_time_X0d4),
            45 + _utils.remove_nans(data.angle_X0d4),
        )
        return [cs_X0(strains), cs_X0d2(strains), cs_X0d4(strains)]

    @pytest.mark.slow
    @pytest.mark.parametrize("gbs_threshold", [0, 0.2, 0.4])
    @pytest.mark.parametrize("nucleation_efficiency", [3, 5, 10])
    def test_dvdx_ensemble(
        self, outdir, seeds_nearX45, ncpus, gbs_threshold, nucleation_efficiency
    ):
        r"""Test a-axis alignment to shear in Y direction (init. SCCS near 45° from X).

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        strain_rate = 1
        timestamps = np.linspace(0, 1, 201)  # Solve until D₀t=1 ('shear' γ=2).
        n_timestamps = len(timestamps)
        # Use `seeds` instead of `seeds_nearX45` if you have even more RAM and CPU time.
        _seeds = seeds_nearX45
        n_seeds = len(_seeds)

        shear_direction = [0, 1, 0]  # Used to calculate the angular diagnostics.
        get_velocity_gradient = _dv.simple_shear_2d("Y", "X", strain_rate)

        gbm_mobilities = [0, 50, 125, 200]
        markers = ("x", "*", "d", "s")

        _id = f"X{_io.stringify(gbs_threshold)}_L{_io.stringify(nucleation_efficiency)}"
        # Output setup with optional logging and data series labels.
        θ_fse = np.empty_like(timestamps)
        angles = np.empty((len(gbm_mobilities), n_seeds, n_timestamps))
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_dvdx_ensemble_{_id}"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            clock_start = process_time()
            for m, gbm_mobility in enumerate(gbm_mobilities):
                if m == 0:
                    return_fse = True
                else:
                    return_fse = False

                params = {
                    "olivine_fraction": 1.0,
                    "enstatite_fraction": 0.0,
                    "stress_exponent": 1.5,
                    "deformation_exponent": 3.5,
                    "gbm_mobility": gbm_mobility,
                    "gbs_threshold": gbs_threshold,
                    "nucleation_efficiency": nucleation_efficiency,
                    "number_of_grains": 5000,
                    "initial_olivine_fabric": "A",
                }

                _run = ft.partial(
                    self.run,
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    return_fse=return_fse,
                )
                with Pool(processes=ncpus) as pool:
                    for s, out in enumerate(pool.imap_unordered(_run, _seeds)):
                        mineral, fse_angles = out
                        angles[m, s, :] = [
                            _diagnostics.smallest_angle(v, shear_direction)
                            for v in _diagnostics.elasticity_components(
                                _minerals.voigt_averages([mineral], params)
                            )["hexagonal_axis"]
                        ]
                        # Save the whole mineral for the first seed only.
                        if outdir is not None and s == 0:
                            postfix = (
                                f"M{_io.stringify(gbm_mobility)}"
                                + f"_X{_io.stringify(gbs_threshold)}"
                                + f"_L{_io.stringify(nucleation_efficiency)}"
                            )
                            mineral.save(f"{out_basepath}.npz", postfix=postfix)
                        if return_fse:
                            θ_fse += fse_angles

                if return_fse:
                    θ_fse /= n_seeds

                if outdir is not None:
                    labels.append(f"$M^∗$ = {gbm_mobility}")

            _log.info(
                "elapsed CPU time: %s",
                _utils.readable_timestamp(np.abs(process_time() - clock_start)),
            )

        # Take ensemble means and optionally plot figure.
        strains = timestamps * strain_rate
        _log.info("postprocessing results for %s", _id)
        result_angles = angles.mean(axis=1)
        result_angles_err = angles.std(axis=1)

        if outdir is not None:
            schema = {
                "delimiter": ",",
                "missing": "-",
                "fields": [
                    {
                        "name": "strain",
                        "type": "integer",
                        "unit": "percent",
                        "fill": 999999,
                    }
                ],
            }
            _io.save_scsv(
                f"{out_basepath}_strains.scsv",
                schema,
                [[int(D * 200) for D in strains]],  # Shear strain % is 200 * D₀.
            )
            fig, ax, colors = _vis.alignment(
                None,
                strains,
                result_angles,
                markers,
                labels,
                err=result_angles_err,
                θ_max=60,
                θ_fse=θ_fse,
            )
            fig.savefig(_io.resolve_path(f"{out_basepath}.pdf"))

    @pytest.mark.slow
    def test_dvdx_GBM(self, outdir, seeds_nearX45, ncpus):
        r"""Test a-axis alignment to shear in Y direction (init. SCCS near 45° from X).

        Velocity gradient:
        $$
        \bm{L} = 10^{-4} ×
            \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}
        $$

        Results are compared to the Fortran 90 output.

        """
        shear_direction = [0, 1, 0]  # Used to calculate the angular diagnostics.
        strain_rate = 1e-4
        get_velocity_gradient = _dv.simple_shear_2d("Y", "X", strain_rate)
        timestamps = np.linspace(0, 1e4, 51)  # Solve until D₀t=1 ('shear' γ=2).
        i_strain_40p = 10  # Index of 40% strain, lower strains are not relevant here.
        i_strain_100p = 25  # Index of 100% strain, when M*=0 matches FSE.
        params = _io.DEFAULT_PARAMS
        params["gbs_threshold"] = 0  # No GBS, to match the Fortran parameters.
        gbm_mobilities = (0, 10, 50, 125, 200)  # Must be in ascending order.
        markers = ("x", ".", "*", "d", "s")
        # Use `seeds` instead of `seeds_nearX45` if you have even more RAM and CPU time.
        _seeds = seeds_nearX45
        n_seeds = len(_seeds)
        angles = np.empty((len(gbm_mobilities), n_seeds, len(timestamps)))
        θ_fse = np.empty_like(timestamps)
        strains = timestamps * strain_rate
        M0_drexF90, M50_drexF90, M200_drexF90 = self.interp_GBM_FortranDRex(strains)

        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_mobility"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            clock_start = process_time()
            for m, gbm_mobility in enumerate(gbm_mobilities):
                if m == 0:
                    return_fse = True
                else:
                    return_fse = False
                params["gbm_mobility"] = gbm_mobility

                _run = ft.partial(
                    self.run,
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    return_fse=True,
                )
                with Pool(processes=ncpus) as pool:
                    for s, out in enumerate(pool.imap_unordered(_run, _seeds)):
                        mineral, fse_angles = out
                        angles[m, s, :] = [
                            _diagnostics.smallest_angle(v, shear_direction)
                            for v in _diagnostics.elasticity_components(
                                _minerals.voigt_averages([mineral], params)
                            )["hexagonal_axis"]
                        ]
                        # Save the whole mineral for the first seed only.
                        if outdir is not None and s == 0:
                            mineral.save(
                                f"{out_basepath}.npz",
                                postfix=f"M{_io.stringify(gbm_mobility)}",
                            )
                        if return_fse:
                            θ_fse += fse_angles

                if return_fse:
                    θ_fse /= n_seeds

                if outdir is not None:
                    labels.append(f"$M^∗$ = {params['gbm_mobility']}")

            _log.info(
                "elapsed CPU time: %s",
                _utils.readable_timestamp(np.abs(process_time() - clock_start)),
            )

        # Take ensemble means and optionally plot figure.
        _log.info("postprocessing results...")
        result_angles = angles.mean(axis=1)
        result_angles_err = angles.std(axis=1)

        if outdir is not None:
            schema = {
                "delimiter": ",",
                "missing": "-",
                "fields": [
                    {
                        "name": "strain",
                        "type": "integer",
                        "unit": "percent",
                        "fill": 999999,
                    }
                ],
            }
            _io.save_scsv(
                f"{out_basepath}_strains.scsv",
                schema,
                [[int(D * 200) for D in strains]],  # Shear strain % is 200 * D₀.
            )
            fig, ax, colors = _vis.alignment(
                None,
                strains,
                result_angles,
                markers,
                labels,
                err=result_angles_err,
                θ_max=60,
                θ_fse=θ_fse,
            )
            ax.plot(strains, M0_drexF90, c=colors[0])
            ax.plot(strains, M50_drexF90, c=colors[2])
            ax.plot(strains, M200_drexF90, c=colors[4])
            fig.savefig(_io.resolve_path(f"{out_basepath}.pdf"))

        # Check that GBM speeds up the alignment between 40% and 100% strain.
        _log.info("checking grain orientations...")
        for i, θ in enumerate(result_angles[:-1], start=1):
            nt.assert_array_less(
                result_angles[i][i_strain_40p:i_strain_100p],
                θ[i_strain_40p:i_strain_100p],
            )

        # Check that M*=0 matches FSE (±1°) past 100% strain.
        nt.assert_allclose(
            result_angles[0][i_strain_100p:],
            θ_fse[i_strain_100p:],
            atol=1,
            rtol=0,
        )

        # Check that results match Fortran output past 40% strain.
        nt.assert_allclose(
            result_angles[0][i_strain_40p:],
            M0_drexF90[i_strain_40p:],
            atol=0,
            rtol=0.1,  # At 40% strain the match is worse than at higher strain.
        )
        nt.assert_allclose(
            result_angles[2][i_strain_40p:],
            M50_drexF90[i_strain_40p:],
            atol=1,
            rtol=0,
        )
        nt.assert_allclose(
            result_angles[4][i_strain_40p:],
            M200_drexF90[i_strain_40p:],
            atol=1.5,
            rtol=0,
        )

        # TODO: Make this into a separate non-ensemble test.
        # # Check that M*=0 doesn't affect grain sizes.
        # _log.info("checking grain sizes...")
        # for i, time in enumerate(timestamps):
        #     nt.assert_allclose(
        #         minerals[0].fractions[i],
        #         np.full(params["number_of_grains"], 1 / params["number_of_grains"]),
        #     )

        # TODO: Make this into a separate non-ensemble test.
        # # Check that GBM causes decreasing grain size median.
        # assert np.all(
        #     np.array(
        #         [
        #             np.median(m.fractions[halfway])
        #             - np.median(minerals[i].fractions[halfway])
        #             for i, m in enumerate(minerals[:-1], start=1)
        #         ]
        #     )
        #     > 0
        # )

    def test_boundary_sliding(self, seed, ncpus, outdir):
        """Test that the grain boundary sliding parameter has an effect."""
        strain_rate = 1.0
        shear_direction = [1, 0, 0]  # Used to calculate the angular diagnostics.
        get_velocity_gradient = _dv.simple_shear_2d("X", "Z", strain_rate)
        timestamps = np.linspace(0, 2.5, 251)  # Solve until D₀t=2.5 ('shear' γ=5).
        i_strain_200p = 100  # Index of 50% strain.
        params = _io.DEFAULT_PARAMS
        gbs_thresholds = (0, 0.2, 0.4, 0.6)  # Must be in ascending order.
        markers = (".", "*", "d", "s")
        angles = np.empty((len(gbs_thresholds), len(timestamps)))
        symmetry = np.empty_like(angles)
        θ_fse = np.empty_like(angles)
        minerals = []

        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_sliding"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            for f, gbs_threshold in enumerate(gbs_thresholds):
                params["gbs_threshold"] = gbs_threshold
                mineral, fse_angles = self.run(
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    seed=seed,
                    return_fse=True,
                )
                minerals.append(mineral)
                angles[f] = [
                    _diagnostics.smallest_angle(v, shear_direction)
                    for v in _diagnostics.elasticity_components(
                        _minerals.voigt_averages([mineral], params)
                    )["hexagonal_axis"]
                ]
                symmetry[f] = [
                    _diagnostics.symmetry(
                        o, axis=_minerals.OLIVINE_PRIMARY_AXIS[mineral.fabric]
                    )[
                        0
                    ]  # P_[100] diagnostic.
                    for o in _stats.resample_orientations(
                        mineral.orientations,
                        mineral.fractions,
                        n_samples=1000,
                        seed=seed,
                    )[0]
                ]
                θ_fse[f] = fse_angles
                if outdir is not None:
                    labels.append(f"$f_{{gbs}}$ = {params['gbs_threshold']}")

        if outdir is not None:
            strains = timestamps * strain_rate
            fig, ax, colors = _vis.alignment(
                None,
                strains,
                angles,
                markers,
                labels,
                θ_max=60,
                θ_fse=np.mean(θ_fse, axis=0),
            )
            fig.savefig(_io.resolve_path(f"{out_basepath}.png"))
            # Save mineral for the X=0.2 run, for polefigs.
            minerals[2].save(f"{out_basepath}.npz")

        # Check that GBS sets an upper bound on P_[100].
        _log.info("checking degree of [100] point symmetry...")
        nt.assert_allclose(
            np.full(len(symmetry[0][i_strain_200p:]), 0.0),
            symmetry[0][i_strain_200p:] - 0.95,
            atol=0.9,
            rtol=0,
        )
        nt.assert_allclose(
            np.full(len(symmetry[1][i_strain_200p:]), 0.0),
            symmetry[1][i_strain_200p:] - 0.775,
            atol=0.9,
            rtol=0,
        )
        nt.assert_allclose(
            np.full(len(symmetry[2][i_strain_200p:]), 0.0),
            symmetry[2][i_strain_200p:] - 0.61,
            atol=0.9,
            rtol=0,
        )
        nt.assert_allclose(
            np.full(len(symmetry[3][i_strain_200p:]), 0.0),
            symmetry[3][i_strain_200p:] - 0.44,
            atol=0.9,
            rtol=0,
        )
        # Check that angles always reach within 7.5° of the shear direction.
        _log.info("checking grain orientations...")
        for θ in angles:
            nt.assert_allclose(
                np.full(len(θ[i_strain_200p:]), 0.0),
                θ[i_strain_200p:],
                atol=7.5,
                rtol=0,
            )

    @pytest.mark.slow
    def test_ngrains(self, seed, outdir):
        """Test that solvers work up to 10000 grains."""
        shear_direction = [0, 1, 0]
        strain_rate = 1.0
        get_velocity_gradient = _dv.simple_shear_2d("Y", "X", strain_rate)
        timestamps = np.linspace(0, 1, 201)  # Solve until D₀t=1 ('shear' γ=2).
        params = _io.DEFAULT_PARAMS
        grain_counts = (100, 1000, 2000, 5000, 10000)

        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_ngrains"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")

        with optional_logging:
            for i, N in enumerate(grain_counts):
                params["number_of_grains"] = N
                self.run(
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    seed=seed,
                    return_fse=True,
                )
