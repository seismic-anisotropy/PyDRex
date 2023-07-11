"""> PyDRex: 2D simple shear tests."""
import contextlib as cl

import numpy as np

from pydrex import diagnostics as _diagnostics
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import stats as _stats
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
        strain_rate = 1

        def get_velocity_gradient(x):
            grad_v = np.zeros((3, 3))
            grad_v[1, 0] = 2 * strain_rate
            return grad_v

        shear_direction = [0, 1, 0]
        timestamps = np.linspace(0, 2e3, 500)

        # Optional logging and plotting setup.
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_dvdx_GBM"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            θ_fse = [45]
            labels = []
            angles = []
            indices = []

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
                        "calculating CPO at %s (t=%s) with velocity gradient: %s",
                        self.get_position(None),
                        time,
                        get_velocity_gradient(None).flatten(),
                    )
                    deformation_gradient = mineral.update_orientations(
                        params,
                        deformation_gradient,
                        get_velocity_gradient,
                        pathline=(time, time + timestamps[t], self.get_position),
                    )
                    fse_λ, fse_v = _diagnostics.finite_strain(deformation_gradient)
                    _log.info("strain = %s", fse_λ)
                    if p == 0:
                        θ_fse.append(
                            _diagnostics.smallest_angle(fse_v, shear_direction)
                        )

                n_timesteps = len(mineral.orientations)
                misorient_angles = np.zeros(n_timesteps)
                misorient_indices = np.zeros(n_timesteps)
                # Loop over first dimension (time steps) of orientations.
                for idx, matrices in enumerate(mineral.orientations):
                    orientations_resampled, _ = _stats.resample_orientations(
                        matrices, mineral.fractions[idx]
                    )
                    direction_mean = _diagnostics.bingham_average(
                        orientations_resampled,
                        axis=_minerals.OLIVINE_PRIMARY_AXIS[mineral.fabric],
                    )
                    misorient_angles[idx] = _diagnostics.smallest_angle(
                        direction_mean, shear_direction
                    )
                    # misorient_indices[idx] = _diagnostics.misorientation_index(
                    #     orientations_resampled
                    # )

                # Optionally store plotting metadata.
                if outdir is not None:
                    labels.append(f"$M^∗$ = {params['gbm_mobility']}")
                    # angles.append(misorient_angles)
                    angles.append(
                        [
                            # _diagnostics.smallest_angle(
                            #     _diagnostics.anisotropy(x)[1][2, :], shear_direction
                            # )
                            np.abs(
                                np.rad2deg(
                                    np.arcsin(_diagnostics.anisotropy(x)[1][2, 2])
                                )
                            )
                            for x in _minerals.voigt_averages([mineral], params)
                        ]
                    )
                    indices.append(misorient_indices)
                    mineral.save(
                        f"{out_basepath}.npz",
                        postfix=f"M{params['gbm_mobility']}",
                    )

        # Optionally plot figure.
        if outdir is not None:
            _vis.simple_shear_stationary_2d(
                angles,
                indices,
                timestop=timestamps[-1],
                savefile=f"{out_basepath}.png",
                markers=("o", "v", "s"),
                labels=labels,
                θ_fse=θ_fse,
            )

    def test_dudz_GBS(
        self,
        params_Kaminski2004_fig4_triangles,  # GBS = 0.4
        params_Kaminski2004_fig4_squares,  # GBS = 0.2
        params_Kaminski2004_fig4_circles,  # GBS = 0
        rng,
        outdir,
    ):
        r"""Test a-axis alignment to shear in X direction.

        Velocity gradient:
        $$\bm{L} = \begin{bmatrix} 0 & 0 & 2 \cr 0 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}$$

        """
        strain_rate = 1

        def get_velocity_gradient(x):
            grad_v = np.zeros((3, 3))
            grad_v[0, 2] = 2 * strain_rate
            return grad_v

        shear_direction = [1, 0, 0]
        timestamps = np.linspace(0, 2e3, 500)

        # Optional plotting and logging setup.
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_dudz_GBS"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            θ_fse = [45]
            labels = []
            angles = []
            indices = []

        with optional_logging:
            for p, params in enumerate(
                (
                    params_Kaminski2004_fig4_triangles,  # GBS = 0.4
                    params_Kaminski2004_fig4_squares,  # GBS = 0.2
                    params_Kaminski2004_fig4_circles,  # GBS = 0
                ),
            ):
                mineral = _minerals.Mineral(
                    n_grains=params["number_of_grains"], rng=rng
                )
                deformation_gradient = np.eye(3)  # Undeformed initial state.
                for t, time in enumerate(timestamps[:-1], start=1):
                    _log.info(
                        "calculating CPO at %s (t=%s) with velocity gradient: %s",
                        self.get_position(None),
                        time,
                        get_velocity_gradient(None).flatten(),
                    )
                    deformation_gradient = mineral.update_orientations(
                        params,
                        deformation_gradient,
                        get_velocity_gradient,
                        pathline=(time, time + timestamps[t], self.get_position),
                    )
                    fse_λ, fse_v = _diagnostics.finite_strain(deformation_gradient)
                    _log.info("strain = %s", fse_λ)
                    if p == 0:
                        θ_fse.append(
                            _diagnostics.smallest_angle(fse_v, shear_direction)
                        )

                n_timesteps = len(mineral.orientations)
                misorient_angles = np.zeros(n_timesteps)
                misorient_indices = np.zeros(n_timesteps)
                # Loop over first dimension (time steps) of orientations.
                for idx, matrices in enumerate(mineral.orientations):
                    orientations_resampled, _ = _stats.resample_orientations(
                        matrices, mineral.fractions[idx]
                    )
                    direction_mean = _diagnostics.bingham_average(
                        orientations_resampled,
                        axis=_minerals.OLIVINE_PRIMARY_AXIS[mineral.fabric],
                    )
                    misorient_angles[idx] = _diagnostics.smallest_angle(
                        direction_mean, [1, 0, 0]
                    )
                    # misorient_indices[idx] = _diagnostics.misorientation_index(
                    #     orientations_resampled
                    # )

                # Optionally store plotting metadata.
                if outdir is not None:
                    labels.append(f"$f_{{gbs}}$ = {params['gbs_threshold']}")
                    # angles.append(misorient_angles)
                    angles.append(
                        [
                            # _diagnostics.smallest_angle(
                            #     _diagnostics.anisotropy(x)[1][2, :], shear_direction
                            # )
                            np.abs(
                                np.rad2deg(
                                    np.arcsin(_diagnostics.anisotropy(x)[1][2, 2])
                                )
                            )
                            for x in _minerals.voigt_averages([mineral], params)
                        ]
                    )
                    indices.append(misorient_indices)
                    mineral.save(
                        f"{out_basepath}.npz",
                        postfix=f"X{params['gbs_threshold']}",
                    )

        # Optionally plot figure.
        if outdir is not None:
            _vis.simple_shear_stationary_2d(
                angles,
                indices,
                timestop=timestamps[-1],
                savefile=f"{out_basepath}.png",
                markers=("o", "v", "s"),
                labels=labels,
                θ_fse=θ_fse,
            )
