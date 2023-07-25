"""> PyDRex: 2D simple shear tests."""
import contextlib as cl
from multiprocessing import Pool
import functools as ft
from time import perf_counter, process_time

import numpy as np
import pytest

# from numpy import testing as nt
from scipy.interpolate import PchipInterpolator

from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import stats as _stats
from pydrex import utils as _utils
from pydrex import visualisation as _vis
from pydrex import velocity_gradients as _dv

# Subdirectory of `outdir` used to store outputs from these tests.
SUBDIR = "2d_simple_shear"


class TestOlivineA:
    """Tests for A-type olivine polycrystals in 2D simple shear."""

    class_id = "olivineA"

    @classmethod
    def get_position(cls, t):
        return np.zeros(3)

    @classmethod
    def run(
        cls,
        params,
        timestamps,
        strain_rate,
        get_velocity_gradient,
        shear_direction,
        seed=None,
        log_param=None,
        use_bingham_average=False,
        return_fse=True,
    ):
        """Reusable logic for 2D olivine simple shear tests.

        Always returns a tuple with 4 elements
        (mineral, mean_angles, texture_symmetry, θ_fse),
        but if `return_fse` is None then the last tuple element is also None.

        """
        mineral = _minerals.Mineral(n_grains=params["number_of_grains"], seed=seed)
        deformation_gradient = np.eye(3)  # Undeformed initial state.

        n_timestamps = len(timestamps)
        if return_fse:
            θ_fse = np.empty(n_timestamps)
            θ_fse[0] = 45

        for t, time in enumerate(timestamps[:-1], start=1):
            # Set up logging message depending on dynamic parameter and seeds.
            match log_param:
                case "gbs_threshold":
                    msg_start = f"X = {params['gbs_threshold']}; "
                case "gbm_mobility":
                    msg_start = f"M∗ = {params['gbm_mobility']}; "
                case _:
                    msg_start = ""

            if seed is not None:
                msg_start += f"# {seed}; "

            _log.info(msg_start + "step %s/%s (t = %s)", t, n_timestamps - 1, time)

            deformation_gradient = mineral.update_orientations(
                params,
                deformation_gradient,
                get_velocity_gradient,
                pathline=(time, timestamps[t], cls.get_position),
            )
            _log.info(
                "› velocity gradient = %s",
                get_velocity_gradient(None).flatten(),
            )
            _log.info("› strain D₀t = %.2f", strain_rate * time)
            if return_fse:
                _, fse_v = _diagnostics.finite_strain(deformation_gradient)
                θ_fse[t] = _diagnostics.smallest_angle(fse_v, shear_direction)

        # Compute texture diagnostics.
        texture_symmetry = np.zeros(n_timestamps)
        if use_bingham_average:
            mean_angles = np.zeros(n_timestamps)
        for idx, matrices in enumerate(mineral.orientations):
            orientations_resampled, _ = _stats.resample_orientations(
                matrices, mineral.fractions[idx], seed=seed
            )
            if use_bingham_average:
                direction_mean = _diagnostics.bingham_average(
                    orientations_resampled,
                    axis=_minerals.OLIVINE_PRIMARY_AXIS[mineral.fabric],
                )
                mean_angles[idx] = _diagnostics.smallest_angle(
                    direction_mean, shear_direction
                )
            texture_symmetry[idx] = _diagnostics.symmetry(
                orientations_resampled,
                axis=_minerals.OLIVINE_PRIMARY_AXIS[mineral.fabric],
            )[0]

        if not use_bingham_average:
            # Use SCCS axis (hexagonal symmetry) for the angle instead (opt).
            mean_angles = np.array(
                [
                    _diagnostics.smallest_angle(
                        _diagnostics.anisotropy(v)[1][2, :], shear_direction
                    )
                    for v in _minerals.voigt_averages([mineral], params)
                ]
            )

        if return_fse:
            return mineral, mean_angles, texture_symmetry, θ_fse
        return mineral, mean_angles, texture_symmetry, None

    @classmethod
    def postprocess_ensemble(
        cls,
        timestamps,
        strain_rate,
        angles,
        point100_symmetry,
        θ_fse,
        labels,
        outdir,
        out_basepath,
    ):
        """Reusable postprocessing routine for ensemble simulations."""
        strains = timestamps * strain_rate
        result_angles = angles.mean(axis=1)
        result_angles_err = angles.std(axis=1)
        result_point100_symmetry = point100_symmetry.mean(axis=1)
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
            _vis.simple_shear_stationary_2d(
                strains,
                cls.interp_GBM_Kaminski2001(strains),
                result_angles,
                result_point100_symmetry,
                angles_err=result_angles_err,
                savefile=f"{out_basepath}.png",
                markers=("o", "v", "s"),
                θ_fse=θ_fse,
                labels=labels,
            )

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
    def test_dvdx_GBM_ensemble(
        self,
        params_Kaminski2001_fig5_solid,  # GBM = 0
        params_Kaminski2001_fig5_shortdash,  # GBM = 50
        params_Kaminski2001_fig5_longdash,  # GBM = 200
        seeds_nearX45,  # Use `seeds` if you have lots of RAM and patience (or cores)
        outdir,
        ncpus,
    ):
        r"""Test a-axis alignment to shear in Y direction (init. SCCS near 45° from X).

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        Tests the effect of grain boundary migration, similar to Fig. 5 in
        [Kaminski 2001](https://doi.org/10.1016%2Fs0012-821x%2801%2900356-9).

        """
        strain_rate = 5e-6  # Strain rate from Fraters & Billen, 2021, fig. 3.
        timestamps = np.linspace(0, 2e5, 201)  # Solve until D₀t=2.5 ('shear' γ=5).
        n_timestamps = len(timestamps)
        # i_strain_100p = [0, 50, 100, 150, 200]  # Indices for += 50% strain.

        shear_direction = [0, 1, 0]  # Used to calculate the angular diagnostics.
        get_velocity_gradient = _dv.simple_shear_2d("Y", "X", strain_rate)

        # Output setup with optional logging and data series labels.
        θ_fse = np.empty(n_timestamps)
        angles = np.empty((3, len(seeds_nearX45), n_timestamps))
        point100_symmetry = np.empty_like(angles)
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_dvdx_GBM_ensemble"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            wall_start = perf_counter()
            clock_start = process_time()
            for p, params in enumerate(
                (
                    params_Kaminski2001_fig5_solid,  # GBM = 0
                    params_Kaminski2001_fig5_shortdash,  # GBM = 50
                    params_Kaminski2001_fig5_longdash,  # GBM = 200
                ),
            ):
                if p == 0:
                    return_fse = True
                else:
                    return_fse = False

                _run = ft.partial(
                    self.run,
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    log_param="gbm_mobility",
                    use_bingham_average=False,
                    return_fse=return_fse,
                )
                with Pool(processes=ncpus) as pool:
                    for s, out in enumerate(pool.imap_unordered(_run, seeds_nearX45)):
                        mineral, mean_angles, texture_symmetry, fse_angles = out
                        angles[p, s, :] = mean_angles
                        point100_symmetry[p, s, :] = texture_symmetry
                        if return_fse:
                            θ_fse += fse_angles

                if return_fse:
                    θ_fse /= len(seeds_nearX45)

                # Update labels and store the last mineral of the ensemble for polefigs.
                if outdir is not None:
                    labels.append(f"$M^∗$ = {params['gbm_mobility']}")
                    mineral.save(
                        f"{out_basepath}.npz",
                        postfix=f"M{params['gbm_mobility']}",
                    )

            _log.info(
                "elapsed walltime: %s",
                _utils.readable_timestamp(np.abs(perf_counter() - wall_start)),
            )
            _log.info(
                "elapsed CPU time: %s",
                _utils.readable_timestamp(np.abs(process_time() - clock_start)),
            )

        # Take ensemble means and optionally plot figure.
        self.postprocess_ensemble(
            timestamps,
            strain_rate,
            angles,
            point100_symmetry,
            θ_fse,
            labels,
            outdir,
            out_basepath,
        )

        # # Check that FSE is correct.
        # # First, get theoretical FSE axis for simple shear.
        # # We want the angle from the Y axis (shear direction), so subtract from 90.
        # θ_fse_eq = [90 - _utils.angle_fse_simpleshear(strain) for strain in strains]
        # nt.assert_allclose(result_θ_fse, θ_fse_eq, rtol=1e-7, atol=0)

        # # Check Bingham angles, ignoring the first portion.
        # # Average orientations of near-isotropic distributions are unstable.
        # nt.assert_allclose(
        #     result_θ_fse[i_first_cpo:],
        #     result_angles[0][i_first_cpo:],
        #     rtol=0.11,
        #     atol=0,
        # )
        # nt.assert_allclose(
        #     target_angles[0][i_first_cpo:],
        #     result_angles[0][i_first_cpo:],
        #     rtol=0.11,
        #     atol=0,
        # )
        # nt.assert_allclose(
        #     target_angles[1][i_first_cpo:],
        #     result_angles[1][i_first_cpo:],
        #     rtol=0,
        #     atol=5.7,
        # )
        # nt.assert_allclose(
        #     target_angles[2][i_first_cpo:],
        #     result_angles[2][i_first_cpo:],
        #     rtol=0,
        #     atol=5.5,
        # )

        # # Check point symmetry of [100] at strains of 0%, 50%, 100%, 150% & 200%.
        # nt.assert_allclose(
        #     [0.015, 0.11, 0.19, 0.27, 0.34],
        #     result_point100_symmetry[0].take(i_strain_50p),
        #     rtol=0,
        #     atol=0.015,
        # )
        # nt.assert_allclose(
        #     [0.015, 0.15, 0.33, 0.57, 0.72],
        #     result_point100_symmetry[1].take(i_strain_50p),
        #     rtol=0,
        #     atol=0.015,
        # )
        # nt.assert_allclose(
        #     [0.015, 0.22, 0.64, 0.86, 0.91],
        #     result_point100_symmetry[2].take(i_strain_50p),
        #     rtol=0,
        #     atol=0.015,
        # )

    @pytest.mark.slow
    def test_dudz_GBS_nearX45(
        self,
        params_Kaminski2004_fig4_circles,  # GBS = 0
        params_Kaminski2004_fig4_squares,  # GBS = 0.2
        params_Kaminski2004_fig4_triangles,  # GBS = 0.4
        seeds_nearX45,  # Use `seeds` if you have lots of RAM and patience (or cores)
        outdir,
        ncpus,
    ):
        r"""Test a-axis alignment to shear in X direction (init. SCCS near 45° from X).

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 2 \cr 0 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        strain_rate = 5e-6  # Strain rate from Fraters & Billen, 2021, fig. 3.
        timestamps = np.linspace(0, 5e5, 251)  # Solve until D₀t=2.5 ('shear' γ=5).
        n_timestamps = len(timestamps)
        # i_strain_100p = [0, 50, 100, 150, 200]  # Indices for += 100% strain.

        shear_direction = [1, 0, 0]  # Used to calculate the angular diagnostics.
        get_velocity_gradient = _dv.simple_shear_2d("X", "Z", strain_rate)

        # Output setup with optional logging and data series labels.
        θ_fse = np.empty(n_timestamps)
        angles = np.empty((3, len(seeds_nearX45), n_timestamps))
        point100_symmetry = np.empty_like(angles)
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_dudz_GBS_ensemble"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            wall_start = perf_counter()
            clock_start = process_time()
            for p, params in enumerate(
                (
                    params_Kaminski2004_fig4_circles,  # GBS = 0
                    params_Kaminski2004_fig4_squares,  # GBS = 0.2
                    params_Kaminski2004_fig4_triangles,  # GBS = 0.4
                ),
            ):
                if p == 0:
                    return_fse = True
                else:
                    return_fse = False

                _run = ft.partial(
                    self.run,
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    log_param="gbs_threshold",
                    use_bingham_average=False,
                    return_fse=return_fse,
                )
                with Pool(processes=ncpus) as pool:
                    for s, out in enumerate(pool.imap_unordered(_run, seeds_nearX45)):
                        mineral, mean_angles, texture_symmetry, fse_angles = out
                        angles[p, s, :] = mean_angles
                        point100_symmetry[p, s, :] = texture_symmetry
                        if return_fse:
                            θ_fse += fse_angles

                if return_fse:
                    θ_fse /= len(seeds_nearX45)

                # Update labels and store the last mineral of the ensemble for polefigs.
                if outdir is not None:
                    labels.append(f"$f_{{gbs}}$ = {params['gbs_threshold']}")
                    mineral.save(
                        f"{out_basepath}.npz",
                        postfix=f"X{params['gbs_threshold']}",
                    )

            _log.info(
                "elapsed walltime: %s",
                _utils.readable_timestamp(np.abs(perf_counter() - wall_start)),
            )
            _log.info(
                "elapsed CPU time: %s",
                _utils.readable_timestamp(np.abs(process_time() - clock_start)),
            )

        # Take ensemble means and optionally plot figure.
        self.postprocess_ensemble(
            timestamps,
            strain_rate,
            angles,
            point100_symmetry,
            θ_fse,
            labels,
            outdir,
            out_basepath,
        )

        # # Check that FSE is correct.
        # # First, get theoretical FSE axis for simple shear.
        # # We want the angle from the Y axis (shear direction), so subtract from 90.
        # θ_fse_eq = [90 - _utils.angle_fse_simpleshear(strain) for strain in strains]
        # nt.assert_allclose(θ_fse, θ_fse_eq, rtol=1e-7, atol=0)

        # # Check point symmetry of [100] at strains of 0%, 100%, 200%, 300% & 400%.
        # nt.assert_allclose(
        #     [0.015, 0.52, 0.86, 0.93, 0.94],
        #     point100_symmetry[0].take(i_strain_100p),
        #     rtol=0,
        #     atol=0.015,
        # )
        # nt.assert_allclose(
        #     [0.015, 0.42, 0.71, 0.77, 0.79],
        #     point100_symmetry[1].take(i_strain_100p),
        #     rtol=0,
        #     atol=0.015,
        # )
        # nt.assert_allclose(
        #     [0.015, 0.36, 0.57, 0.6, 0.62],
        #     point100_symmetry[2].take(i_strain_100p),
        #     rtol=0,
        #     atol=0.015,
        # )
