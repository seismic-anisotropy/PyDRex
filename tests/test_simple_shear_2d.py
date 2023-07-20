"""> PyDRex: 2D simple shear tests."""
import contextlib as cl

import numpy as np
from numpy import testing as nt
from scipy.interpolate import PchipInterpolator

from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import stats as _stats
from pydrex import utils as _utils
from pydrex import visualisation as _vis

# Subdirectory of `outdir` used to store outputs from these tests.
SUBDIR = "2d_simple_shear"


class TestOlivineA:
    """Tests for A-type olivine polycrystals in 2D simple shear."""

    class_id = "olivineA"

    def get_position(self, t):
        return np.zeros(3)

    def test_dvdx_GBM(
        self,
        params_Kaminski2001_fig5_solid,  # GBM = 0
        params_Kaminski2001_fig5_shortdash,  # GBM = 50
        params_Kaminski2001_fig5_longdash,  # GBM = 200
        rng,
        outdir,
    ):
        r"""Test a-axis alignment to shear in the Y direction.

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        Tests the effect of grain boundary migration, similar to Fig. 5 in
        [Kaminski 2001](https://doi.org/10.1016%2Fs0012-821x%2801%2900356-9).

        """
        strain_rate = 5e-6  # Strain rate from Fraters & Billen, 2021, fig. 3.
        timestamps = np.linspace(0, 2e5, 201)  # Solve until D₀t=1 ('shear strain' γ=2).
        n_timesteps = len(timestamps)
        i_first_cpo = 50  # First index where Bingham averages are sufficiently stable.
        i_strain_50p = [0, 50, 100, 150, 200]  # Indices for += 50% strain.

        def get_velocity_gradient(x):
            # It is independent of time or position in this test.
            grad_v = np.zeros((3, 3))
            grad_v[1, 0] = 2 * strain_rate
            return grad_v

        shear_direction = [0, 1, 0]  # Used to calculate the angular diagnostics.

        # Output setup with optional logging and data series labels.
        θ_fse = [45]
        angles = []
        point100_symmetry = []
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_dvdx_GBM"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            for p, params in enumerate(
                (
                    params_Kaminski2001_fig5_solid,  # GBM = 0
                    params_Kaminski2001_fig5_shortdash,  # GBM = 50
                    params_Kaminski2001_fig5_longdash,  # GBM = 200
                ),
            ):
                mineral = _minerals.Mineral(
                    n_grains=params["number_of_grains"], rng=rng
                )
                deformation_gradient = np.eye(3)  # Undeformed initial state.
                for t, time in enumerate(timestamps[:-1], start=1):
                    _log.info(
                        "step %s/%s (t=%s) with velocity gradient: %s",
                        t,
                        n_timesteps - 1,
                        time,
                        get_velocity_gradient(None).flatten(),
                    )
                    deformation_gradient = mineral.update_orientations(
                        params,
                        deformation_gradient,
                        get_velocity_gradient,
                        pathline=(time, timestamps[t], self.get_position),
                    )
                    fse_λ, fse_v = _diagnostics.finite_strain(deformation_gradient)
                    _log.info("› strain √λ-1=%s (D₀t=%s)", fse_λ, strain_rate * time)
                    if p == 0:
                        θ_fse.append(
                            _diagnostics.smallest_angle(fse_v, shear_direction)
                        )

                texture_symmetry = np.zeros(n_timesteps)
                mean_angles = np.zeros(n_timesteps)
                # Loop over first dimension (time steps) of orientations.
                for idx, matrices in enumerate(mineral.orientations):
                    orientations_resampled, _ = _stats.resample_orientations(
                        matrices, mineral.fractions[idx]
                    )
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

                # Uncomment to use SCCS axis (hexagonal symmetry) for the angle instead.
                # mean_angles = np.array(
                #     [
                #         _diagnostics.smallest_angle(
                #             _diagnostics.anisotropy(v)[1][2, :], shear_direction
                #         )
                #         for v in _minerals.voigt_averages([mineral], params)
                #     ]
                # )

                # Optionally store plotting metadata.
                if outdir is not None:
                    labels.append(f"$M^∗$ = {params['gbm_mobility']}")
                    angles.append(mean_angles)
                    point100_symmetry.append(texture_symmetry)
                    mineral.save(
                        f"{out_basepath}.npz",
                        postfix=f"M{params['gbm_mobility']}",
                    )

            # Interpolate Kaminski & Ribe, 2001 data to get target angles at `strains`.
            _log.info("interpolating target CPO angles...")
            strains = timestamps * strain_rate
            data = _io.read_scsv(_io.data("thirdparty") / "Kaminski2001_GBMshear.scsv")
            cs_M0 = PchipInterpolator(
                _utils.skip_nans(data.equivalent_strain_M0) / 200,
                _utils.skip_nans(data.angle_M0),
            )
            cs_M50 = PchipInterpolator(
                _utils.skip_nans(data.equivalent_strain_M50) / 200,
                _utils.skip_nans(data.angle_M50),
            )
            cs_M200 = PchipInterpolator(
                _utils.skip_nans(data.equivalent_strain_M200) / 200,
                _utils.skip_nans(data.angle_M200),
            )
            target_angles = [cs_M0(strains), cs_M50(strains), cs_M200(strains)]

        # Optionally plot figure.
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
                [[int(γ * 200) for γ in strains]],
            )
            _vis.simple_shear_stationary_2d(
                strains,
                target_angles,
                angles,
                point100_symmetry,
                savefile=f"{out_basepath}.png",
                markers=("o", "v", "s"),
                θ_fse=θ_fse,
                labels=labels,
            )

        # Check that FSE is correct.
        # First, get theoretical FSE axis for simple shear.
        # We want the angle from the Y axis (shear direction), so subtract from 90.
        θ_fse_eq = [90 - _utils.angle_fse_simpleshear(strain) for strain in strains]
        nt.assert_allclose(θ_fse, θ_fse_eq, rtol=1e-7, atol=0)

        # Check Bingham angles, ignoring the first portion.
        # Average orientations of near-isotropic distributions are unstable.
        nt.assert_allclose(
            θ_fse[i_first_cpo:], angles[0][i_first_cpo:], rtol=0.1, atol=0
        )
        nt.assert_allclose(
            target_angles[0][i_first_cpo:], angles[0][i_first_cpo:], rtol=0.1, atol=0
        )
        nt.assert_allclose(
            target_angles[1][i_first_cpo:], angles[1][i_first_cpo:], rtol=0, atol=5.7
        )
        nt.assert_allclose(
            target_angles[2][i_first_cpo:], angles[2][i_first_cpo:], rtol=0, atol=5.5
        )

        # Check point symmetry of [100] at strains of 0%, 50%, 100%, 150% & 200%.
        nt.assert_allclose(
            [0.015, 0.11, 0.19, 0.27, 0.34],
            point100_symmetry[0].take(i_strain_50p),
            rtol=0,
            atol=0.015,
        )
        nt.assert_allclose(
            [0.015, 0.15, 0.33, 0.57, 0.72],
            point100_symmetry[1].take(i_strain_50p),
            rtol=0,
            atol=0.015,
        )
        nt.assert_allclose(
            [0.015, 0.22, 0.64, 0.86, 0.91],
            point100_symmetry[2].take(i_strain_50p),
            rtol=0,
            atol=0.015,
        )

    def test_dudz_GBS(
        self,
        params_Kaminski2004_fig4_circles,  # GBS = 0
        params_Kaminski2004_fig4_squares,  # GBS = 0.2
        params_Kaminski2004_fig4_triangles,  # GBS = 0.4
        rng,
        outdir,
    ):
        r"""Test a-axis alignment to shear in X direction.

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 2 \cr 0 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        strain_rate = 5e-6  # Strain rate from Fraters & Billen, 2021, fig. 3.
        timestamps = np.linspace(0, 5e5, 251)  # Solve until D₀t=2.5 ('shear' γ=5).
        n_timesteps = len(timestamps)
        i_strain_100p = [0, 50, 100, 150, 200]  # Indices for += 100% strain.

        def get_velocity_gradient(x):
            # It is independent of time or position in this test.
            grad_v = np.zeros((3, 3))
            grad_v[0, 2] = 2 * strain_rate
            return grad_v

        shear_direction = [1, 0, 0]  # Used to calculate the angular diagnostics.

        # Output setup with optional logging and data series labels.
        θ_fse = [45]
        angles = []
        point100_symmetry = []
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_dudz_GBS"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            labels = []

        with optional_logging:
            for p, params in enumerate(
                (
                    params_Kaminski2004_fig4_circles,  # GBS = 0
                    params_Kaminski2004_fig4_squares,  # GBS = 0.2
                    params_Kaminski2004_fig4_triangles,  # GBS = 0.4
                ),
            ):
                mineral = _minerals.Mineral(
                    n_grains=params["number_of_grains"], rng=rng
                )
                deformation_gradient = np.eye(3)  # Undeformed initial state.
                for t, time in enumerate(timestamps[:-1], start=1):
                    _log.info(
                        "step %s/%s (t=%s) with velocity gradient: %s",
                        t,
                        n_timesteps - 1,
                        time,
                        get_velocity_gradient(None).flatten(),
                    )
                    deformation_gradient = mineral.update_orientations(
                        params,
                        deformation_gradient,
                        get_velocity_gradient,
                        pathline=(time, timestamps[t], self.get_position),
                    )
                    fse_λ, fse_v = _diagnostics.finite_strain(deformation_gradient)
                    _log.info("› strain √λ-1=%s (D₀t=%s)", fse_λ, strain_rate * time)
                    if p == 0:
                        θ_fse.append(
                            _diagnostics.smallest_angle(fse_v, shear_direction)
                        )

                texture_symmetry = np.zeros(n_timesteps)
                mean_angles = np.zeros(n_timesteps)
                # Loop over first dimension (time steps) of orientations.
                for idx, matrices in enumerate(mineral.orientations):
                    orientations_resampled, _ = _stats.resample_orientations(
                        matrices, mineral.fractions[idx]
                    )
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

                # Uncomment to use SCCS axis (hexagonal symmetry) for the angle instead.
                # mean_angles = np.array(
                #     [
                #         _diagnostics.smallest_angle(
                #             _diagnostics.anisotropy(v)[1][2, :], shear_direction
                #         )
                #         for v in _minerals.voigt_averages([mineral], params)
                #     ]
                # )

                # Optionally store plotting metadata.
                if outdir is not None:
                    labels.append(f"$f_{{gbs}}$ = {params['gbs_threshold']}")
                    angles.append(mean_angles)
                    point100_symmetry.append(texture_symmetry)
                    mineral.save(
                        f"{out_basepath}.npz",
                        postfix=f"X{params['gbs_threshold']}",
                    )

            # Interpolate Kaminski & Ribe, 2001 data to get target angles at `strains`.
            _log.info("interpolating target CPO angles...")
            strains = timestamps * strain_rate
            data = _io.read_scsv(_io.data("thirdparty") / "Kaminski2004_GBSshear.scsv")
            cs_X0 = PchipInterpolator(
                _utils.skip_nans(data.dimensionless_time_X0),
                45 + _utils.skip_nans(data.angle_X0),
            )
            cs_X0d2 = PchipInterpolator(
                _utils.skip_nans(data.dimensionless_time_X0d2),
                45 + _utils.skip_nans(data.angle_X0d2),
            )
            cs_X0d4 = PchipInterpolator(
                _utils.skip_nans(data.dimensionless_time_X0d4),
                45 + _utils.skip_nans(data.angle_X0d4),
            )
            target_angles = [cs_X0(strains), cs_X0d2(strains), cs_X0d4(strains)]

        # Optionally plot figure.
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
                [[int(γ * 200) for γ in strains]],
            )
            _vis.simple_shear_stationary_2d(
                strains,
                target_angles,
                angles,
                point100_symmetry,
                savefile=f"{out_basepath}.png",
                markers=("o", "v", "s"),
                labels=labels,
            )

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
