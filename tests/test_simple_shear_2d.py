"""> PyDRex: 2D simple shear tests."""
import contextlib as cl
import functools as ft
from multiprocessing import Pool
from time import process_time

import numpy as np
import pytest
from numpy import testing as nt
from scipy.interpolate import PchipInterpolator

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

        n_timestamps = len(timestamps) - 1
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
                case "number_of_grains":
                    msg_start = f"N = {params['number_of_grains']}; "
                case _:
                    msg_start = ""

            if seed is not None:
                msg_start += f"# {seed}; "

            _log.info(msg_start + "step %s/%s (t = %s)", t, n_timestamps, time)

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
    def postprocess(
        cls,
        strains,
        angles,
        point100_symmetry,
        θ_fse,
        labels,
        markers,
        outdir,
        out_basepath,
        target_interpolator=None,
    ):
        """Reusable postprocessing routine for olivine 2D simple shear simulations."""
        _log.info("postprocessing results...")
        if target_interpolator is not None:
            result_angles = angles.mean(axis=1)
            result_angles_err = angles.std(axis=1)
            result_point100_symmetry = point100_symmetry.mean(axis=1)
            target_angles = target_interpolator(strains)
        else:
            result_angles = angles
            result_angles_err = None
            result_point100_symmetry = point100_symmetry
            target_angles = None

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
                result_angles,
                result_point100_symmetry,
                target_angles=target_angles,
                angles_err=result_angles_err,
                savefile=f"{out_basepath}.png",
                markers=markers,
                θ_fse=θ_fse,
                labels=labels,
            )
        return result_angles, result_angles_err, result_point100_symmetry, target_angles

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
        timestamps = np.linspace(0, 2e5, 201)  # Solve until D₀t=1 ('shear' γ=2).
        n_timestamps = len(timestamps) - 1
        i_strain_50p = [0, 50, 100, 150, 200]  # Indices for += 50% strain.

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
                "elapsed CPU time: %s",
                _utils.readable_timestamp(np.abs(process_time() - clock_start)),
            )

        # Take ensemble means and optionally plot figure.
        strains = timestamps * strain_rate
        res = self.postprocess(
            strains,
            angles,
            point100_symmetry,
            θ_fse,
            labels,
            ("o", "v", "s"),
            outdir,
            out_basepath,
            target_interpolator=self.interp_GBM_Kaminski2001,
        )
        result_angles, result_angles_err, result_point100_symmetry, target_angles = res

        # Check that FSE is correct.
        # First, get theoretical FSE axis for simple shear.
        # We want the angle from the Y axis (shear direction), so subtract from 90.
        θ_fse_eq = [90 - _utils.angle_fse_simpleshear(strain) for strain in strains]
        nt.assert_allclose(θ_fse, θ_fse_eq, rtol=1e-7, atol=0)

        # Check that M*=0 angles match FSE, ignoring the first portion (<100% strain).
        # Average orientations of near-isotropic distributions are unstable.
        nt.assert_allclose(
            θ_fse[i_strain_50p[2] :],
            result_angles[0][i_strain_50p[2] :],
            rtol=0.11,
            atol=0,
        )
        # Check that M*=0 matches target angles for strain > 100%.
        nt.assert_allclose(
            target_angles[0][i_strain_50p[2]],
            result_angles[0][i_strain_50p[2]],
            atol=1,
            rtol=0,
        )
        # Check that standard deviation decreases or stagnates (0.01 tolerance).
        # Check for smooth decrease or stagnation in ensemble average (0.01 tolerance).
        for angles, angles_err in zip(result_angles, result_angles_err):
            assert np.all(np.diff(angles_err) < 0.01)
            assert np.all(np.diff(angles[i_strain_50p[1] :]) < 0.01)

        # Check point symmetry of [100] at strains of 0%, 50%, 100%, 150% & 200%.
        nt.assert_allclose(
            [0.015, 0.11, 0.19, 0.27, 0.34],
            result_point100_symmetry[0].take(i_strain_50p),
            rtol=0,
            atol=0.015,
        )
        nt.assert_allclose(
            [0.015, 0.15, 0.33, 0.57, 0.72],
            result_point100_symmetry[1].take(i_strain_50p),
            rtol=0,
            atol=0.015,
        )
        nt.assert_allclose(
            [0.015, 0.22, 0.64, 0.86, 0.91],
            result_point100_symmetry[2].take(i_strain_50p),
            rtol=0,
            atol=0.015,
        )

    @pytest.mark.slow
    def test_dudz_GBS_ensemble(
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
        n_timestamps = len(timestamps) - 1
        i_strain_100p = [0, 50, 100, 150, 200]  # Indices for += 100% strain.

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
                "elapsed CPU time: %s",
                _utils.readable_timestamp(np.abs(process_time() - clock_start)),
            )

        # Take ensemble means and optionally plot figure.
        strains = timestamps * strain_rate
        res = self.postprocess(
            strains,
            angles,
            point100_symmetry,
            θ_fse,
            labels,
            ("o", "v", "s"),
            outdir,
            out_basepath,
            target_interpolator=self.interp_GBS_Kaminski2004,
        )
        result_angles, result_angles_err, result_point100_symmetry, target_angles = res

        # Check that FSE is correct.
        # First, get theoretical FSE axis for simple shear.
        # We want the angle from the Y axis (shear direction), so subtract from 90.
        θ_fse_eq = [90 - _utils.angle_fse_simpleshear(strain) for strain in strains]
        nt.assert_allclose(θ_fse, θ_fse_eq, rtol=1e-7, atol=0)

        # Check point symmetry of [100] at strains of 0%, 100%, 200%, 300% & 400%.
        nt.assert_allclose(
            [0.015, 0.52, 0.86, 0.93, 0.94],
            point100_symmetry[0].take(i_strain_100p),
            rtol=0,
            atol=0.015,
        )
        nt.assert_allclose(
            [0.015, 0.42, 0.71, 0.77, 0.79],
            point100_symmetry[1].take(i_strain_100p),
            rtol=0,
            atol=0.015,
        )
        nt.assert_allclose(
            [0.015, 0.36, 0.57, 0.6, 0.62],
            point100_symmetry[2].take(i_strain_100p),
            rtol=0,
            atol=0.015,
        )

        # Check that standard deviation decreases or stagnates (0.01 tolerance).
        # Check for smooth decrease or stagnation in ensemble average (0.01 tolerance).
        for angles, angles_err in zip(result_angles, result_angles_err):
            assert np.all(np.diff(angles_err) < 0.01)
            assert np.all(np.diff(angles[i_strain_100p[1] :]) < 0.01)

    def test_boundary_mobility(self, seed, outdir):
        """Test that the grain boundary mobility parameter has an effect."""
        shear_direction = [0, 1, 0]  # Used to calculate the angular diagnostics.
        strain_rate = 1.0
        get_velocity_gradient = _dv.simple_shear_2d("Y", "X", strain_rate)
        timestamps = np.linspace(0, 1, 201)  # Solve until D₀t=1 ('shear' γ=2).
        i_strain_50p = 50  # Index of 50% strain.
        params = _io.DEFAULT_PARAMS
        gbm_mobilities = (0, 10, 50, 125, 200)  # Must be in ascending order.
        markers = ("x", ".", "*", "d", "s")
        angles = np.empty((len(gbm_mobilities), len(timestamps)))
        point100_symmetry = np.empty_like(angles)
        θ_fse = np.empty_like(angles)
        minerals = []

        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_mobility"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            for i, M in enumerate(gbm_mobilities):
                params["gbm_mobility"] = M
                out = self.run(
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    seed=seed,
                    log_param="gbm_mobility",
                    return_fse=True,
                )
                minerals.append(out[0])
                angles[i] = out[1]
                point100_symmetry[i] = out[2]
                θ_fse[i] = out[3]
                if outdir is not None:
                    labels.append(f"$M^∗$ = {params['gbm_mobility']}")

        if outdir is not None:
            strains = timestamps * strain_rate
            self.postprocess(
                strains,
                angles,
                point100_symmetry,
                np.mean(θ_fse, axis=0),
                labels,
                markers,
                outdir,
                out_basepath,
            )

        # Check that GBM speeds up the alignment.
        _log.info("checking grain orientations...")
        halfway = int(len(timestamps) / 2)
        assert np.all(
            np.array(
                [
                    θ[halfway] - angles[i][halfway]
                    for i, θ in enumerate(angles[:-1], start=1)
                ]
            )
            > 0
        )
        assert np.all(
            np.array(
                [θ[-1] - angles[i][-1] for i, θ in enumerate(angles[:-1], start=1)]
            )
            > 0
        )
        # Check that M*=0 doesn't affect grain sizes.
        _log.info("checking grain sizes...")
        for i, time in enumerate(timestamps):
            nt.assert_allclose(
                minerals[0].fractions[i],
                np.full(params["number_of_grains"], 1 / params["number_of_grains"]),
            )
        # Check that M*=0 matches FSE past 100% strain.
        nt.assert_allclose(
            angles[0][i_strain_50p:],
            np.mean(θ_fse, axis=0)[i_strain_50p:],
            atol=1,
            rtol=0,
        )
        # Check that GBM causes decreasing grain size median.
        assert np.all(
            np.array(
                [
                    np.median(m.fractions[halfway])
                    - np.median(minerals[i].fractions[halfway])
                    for i, m in enumerate(minerals[:-1], start=1)
                ]
            )
            > 0
        )

    def test_boudary_sliding(self, seed, outdir):
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
        point100_symmetry = np.empty_like(angles)
        θ_fse = np.empty_like(angles)
        minerals = []

        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_sliding"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            for i, f in enumerate(gbs_thresholds):
                params["gbs_threshold"] = f
                out = self.run(
                    params,
                    timestamps,
                    strain_rate,
                    get_velocity_gradient,
                    shear_direction,
                    seed=seed,
                    log_param="gbs_threshold",
                    return_fse=True,
                )
                minerals.append(out[0])
                angles[i] = out[1]
                point100_symmetry[i] = out[2]
                θ_fse[i] = out[3]
                if outdir is not None:
                    labels.append(f"$M^∗$ = {params['gbs_threshold']}")

        if outdir is not None:
            strains = timestamps * strain_rate
            self.postprocess(
                strains,
                angles,
                point100_symmetry,
                np.mean(θ_fse, axis=0),
                labels,
                markers,
                outdir,
                out_basepath,
            )

        # Check that GBS sets an upper bound on P_[100].
        _log.info("checking degree of [100] point symmetry...")
        nt.assert_allclose(
            np.full(len(point100_symmetry[0][i_strain_200p:]), 0.0),
            point100_symmetry[0][i_strain_200p:] - 0.95,
            atol=0.05,
            rtol=0,
        )
        nt.assert_allclose(
            np.full(len(point100_symmetry[1][i_strain_200p:]), 0.0),
            point100_symmetry[1][i_strain_200p:] - 0.78,
            atol=0.05,
            rtol=0,
        )
        nt.assert_allclose(
            np.full(len(point100_symmetry[2][i_strain_200p:]), 0.0),
            point100_symmetry[2][i_strain_200p:] - 0.61,
            atol=0.05,
            rtol=0,
        )
        nt.assert_allclose(
            np.full(len(point100_symmetry[3][i_strain_200p:]), 0.0),
            point100_symmetry[3][i_strain_200p:] - 0.44,
            atol=0.055,
            rtol=0,
        )
        # Check that angles always reach within 5° of the shear direction.
        _log.info("checking grain orientations...")
        for θ in angles:
            nt.assert_allclose(
                np.full(len(θ[i_strain_200p:]), 0.0),
                2.5 - θ[i_strain_200p:],
                atol=2.5,
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
                    log_param="number_of_grains",
                    return_fse=True,
                )
