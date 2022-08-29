"""PyDRex: 2D simple shear tests.

NOTE: In scipy, rotations are represented as a matrix
that transforms [1, 0, 0] to the new a-axis vector.
We need the inverse of that rotation,
which represents the change of coordinates
from the grain-local to the global (Eulerian) frame.

"""
import numpy as np
from numpy import random as rn
from scipy.spatial.transform import Rotation

from pydrex import deformation_mechanism as _defmech
from pydrex import diagnostics as _diagnostics
from pydrex import minerals as _minerals
from pydrex import visualisation as _vis


class TestSinglePolycrystalOlivineA:
    """Tests for a single A-type olivine polycrystal in 2D simple shear."""

    def test_shearYZ_initQ1(
        self,
        params_Kaminski2001_fig5_solid,  # GBM = 0
        params_Kaminski2001_fig5_shortdash,  # GBM = 50
        params_Kaminski2001_fig5_longdash,  # GBM = 200
        outdir,
    ):
        """Test clockwise a-axis rotation around X.

        Initial condition: randomised a-axis orientation within the first
        quadrant (between +Y and +Z axes) and steady flow with velocity gradient

                0 0 0
            L = 0 0 2
                0 0 0

        Orientations set up for slip on (010)[100].
        Tests the effect of grain boundary migration,
        similar to Fig. 5 in [Kaminski 2001].

        [Kaminski 2001]: https://doi.org/10.1016%2Fs0012-821x%2801%2900356-9

        """
        strain_rate_scale = 2e-5
        velocity_gradient = np.zeros((3, 3))
        velocity_gradient[1, 2] = strain_rate_scale
        timescale = 1 / (strain_rate_scale / 2)
        n_grains = 1000

        # Initial orientations with a-axis in first quadrant of the YZ plane,
        # and c-axis along the +X direction (important!)
        # This means that the starting average is the same,
        # which is convenient for comparing the texture evolution.
        orientations_init = (
            Rotation.from_euler(
                "zxz",
                [
                    [x * np.pi / 2, np.pi / 2, np.pi / 2]
                    for x in rn.default_rng().random(n_grains)
                ],
            )
            .inv()
            .as_matrix()
        )
        # Uncomment to check a-axis vectors, should be near [0, a, a].
        # assert False, f"{orientations_init[0:10, 0, :]}"
        # Uncomment to check c-axis vectors, should be near [1, 0, 0].
        # assert False, f"{orientations_init[0:10, 2, :]}"

        # One mineral to test each value of grain boundary mobility.
        minerals = [
            _minerals.Mineral(
                _minerals.MineralPhase.olivine,
                _minerals.OlivineFabric.A,
                _defmech.Regime.dislocation,
                n_grains=n_grains,
                fractions_init=np.full(n_grains, 1 / n_grains),
                orientations_init=orientations_init,
            )
            for i in range(3)
        ]

        # Optional plotting setup.
        if outdir is not None:
            labels = []
            angles = []
            indices = []

        for mineral, params in zip(
            minerals,
            (
                params_Kaminski2001_fig5_solid,  # GBM = 0
                params_Kaminski2001_fig5_shortdash,  # GBM = 50
                params_Kaminski2001_fig5_longdash,  # GBM = 200
            ),
        ):
            time = 0
            timestep = 0.025
            timestop = 1
            deformation_gradient = np.eye(3)  # Undeformed initial state.
            while time < timestop * timescale:
                deformation_gradient = mineral.update_orientations(
                    params,
                    deformation_gradient,
                    velocity_gradient,
                    integration_time=timestep * timescale,
                )
                time += timestep * timescale

            n_timesteps = len(mineral.orientations)
            misorient_angles = np.zeros(n_timesteps)
            misorient_indices = np.zeros(n_timesteps)
            # Loop over first dimension (time steps) of orientations.
            for idx, matrices in enumerate(mineral.orientations):
                orientations_resampled, _ = _diagnostics.resample_orientations(
                    matrices, mineral.fractions[idx]
                )
                direction_mean = _diagnostics.bingham_average(
                    orientations_resampled,
                    axis=_minerals.get_primary_axis(mineral.fabric),
                )
                misorient_angles[idx] = _diagnostics.smallest_angle(
                    direction_mean, [0, 1, 0]
                )
                misorient_indices[idx] = _diagnostics.misorientation_index(
                    orientations_resampled
                )

            # Check for uncorrupted record of initial condition.
            assert np.isclose(misorient_angles[0], 45.0, atol=5.0)
            assert misorient_indices[0] < 0.71
            # Check for mostly smoothly decreasing misalignment.
            angles_diff = np.diff(misorient_angles)
            assert np.max(angles_diff) < 3.2
            assert np.min(angles_diff) > -7.5
            assert np.sum(angles_diff) < -25.0

            # Check alignment and texture strength (half way and final value).
            halfway = round(n_timesteps / 2)
            match params["gbm_mobility"]:
                case 0:
                    np.testing.assert_allclose(
                        misorient_angles,
                        misorient_angles[0]
                        * np.exp(
                            np.linspace(0, timestop, n_timesteps)
                            * (np.cos(2 * np.deg2rad(misorient_angles[0])) - 1)
                        ),
                        atol=5.0,
                    )
                    assert np.isclose(misorient_angles[halfway], 25, atol=2.0)
                    assert np.isclose(misorient_angles[-1], 17.0, atol=1.0)
                    assert np.isclose(misorient_indices[halfway], 0.925, atol=0.075)
                    assert np.isclose(misorient_indices[-1], 0.975, atol=0.05)
                case 50:
                    assert np.isclose(misorient_angles[halfway], 15, atol=1.5)
                    assert np.isclose(misorient_angles[-1], 10, atol=1.0)
                    assert np.isclose(misorient_indices[halfway], 0.925, atol=0.075)
                    assert np.isclose(misorient_indices[-1], 0.97, atol=0.03)
                case 200:
                    assert np.isclose(misorient_angles[halfway], 9, atol=1.5)
                    assert np.isclose(misorient_angles[-1], 7, atol=1.0)
                    assert np.isclose(misorient_indices[halfway], 0.975, atol=0.05)
                    assert np.isclose(misorient_indices[-1], 0.99, atol=0.03)

            # Optionally store plotting metadata.
            if outdir is not None:
                labels.append(f"$M^∗$ = {params['gbm_mobility']}")
                angles.append(misorient_angles)
                indices.append(misorient_indices)

        # Optionally plot figure.
        if outdir is not None:
            _vis.simple_shear_2d(
                angles,
                indices,
                timestop=timestop,
                savefile=f"{outdir}/simple_shearYZ_single_olivineA_initQ1.png",
                markers=("o", "v", "s"),
                labels=labels,
                refval=45,
            )

    def test_shearXZ_initQ1(
        self,
        params_Kaminski2004_fig4_triangles,  # GBS = 0.4
        params_Kaminski2004_fig4_squares,  # GBS = 0.2
        params_Kaminski2004_fig4_circles,  # GBS = 0
        outdir,
    ):
        """Test clockwise a-axis rotation around Y.

        Initial condition: randomised a-axis orientation within the first
        quadrant (between +X and +Z axes) and steady flow with velocity gradient

                0 0 2
            L = 0 0 0
                0 0 0

        Orientations set up for slip on (010)[100].

        """
        strain_rate_scale = 2e-5
        velocity_gradient = np.zeros((3, 3))
        velocity_gradient[0, 2] = strain_rate_scale
        timescale = 1 / (strain_rate_scale / 2)
        n_grains = 1000

        # Initial orientations with a-axis in first quadrant of the XZ plane,
        # and c-axis along the -Y direction (important!)
        # This means that the starting average is the same,
        # which is convenient for comparing the texture evolution.
        orientations_init = (
            Rotation.from_euler(
                "zxz",
                [
                    [x * np.pi / 2, np.pi / 2, 0]
                    for x in rn.default_rng().random(n_grains)
                ],
            )
            .inv()
            .as_matrix()
        )
        # Uncomment to check a-axis vectors, should be near [a, 0, a].
        # assert False, f"{orientations_init[0:10, 0, :]}"
        # Uncomment to check c-axis vectors, should be near [0, -1, 0].
        # assert False, f"{orientations_init[0:10, 2, :]}"

        # One mineral to test each grain boundary sliding threshold.
        minerals = [
            _minerals.Mineral(
                _minerals.MineralPhase.olivine,
                _minerals.OlivineFabric.A,
                _defmech.Regime.dislocation,
                n_grains=n_grains,
                fractions_init=np.full(n_grains, 1 / n_grains),
                orientations_init=orientations_init,
            )
            for i in range(3)
        ]

        # Optional plotting setup.
        if outdir is not None:
            labels = []
            angles = []
            indices = []

        for mineral, params in zip(
            minerals,
            (
                params_Kaminski2004_fig4_triangles,  # GBS = 0.4
                params_Kaminski2004_fig4_squares,  # GBS = 0.2
                params_Kaminski2004_fig4_circles,  # GBS = 0
            ),
        ):
            time = 0
            timestep = 0.025
            timestop = 1
            deformation_gradient = np.eye(3)  # Undeformed initial state.
            while time < timestop * timescale:
                deformation_gradient = mineral.update_orientations(
                    params,
                    deformation_gradient,
                    velocity_gradient,
                    integration_time=timestep * timescale,
                )
                time += timestep * timescale

            n_timesteps = len(mineral.orientations)
            misorient_angles = np.zeros(n_timesteps)
            misorient_indices = np.zeros(n_timesteps)
            # Loop over first dimension (time steps) of orientations.
            for idx, matrices in enumerate(mineral.orientations):
                orientations_resampled, _ = _diagnostics.resample_orientations(
                    matrices, mineral.fractions[idx]
                )
                direction_mean = _diagnostics.bingham_average(
                    orientations_resampled,
                    axis=_minerals.get_primary_axis(mineral.fabric),
                )
                misorient_angles[idx] = _diagnostics.smallest_angle(
                    direction_mean, [1, 0, 0]
                )
                misorient_indices[idx] = _diagnostics.misorientation_index(
                    orientations_resampled
                )

            # Check for uncorrupted record of initial condition.
            assert np.isclose(misorient_angles[0], 45.0, atol=5.0)
            assert misorient_indices[0] < 0.71
            # Check for mostly smoothly decreasing misalignment.
            angles_diff = np.diff(misorient_angles)
            assert np.max(angles_diff) < 3.2
            assert np.min(angles_diff) > -7.5
            assert np.sum(angles_diff) < -25.0
            # Check that recrystallization is causing faster alignment.
            np.testing.assert_array_less(
                misorient_angles - 3.8,  # Tolerance for GBM onset latency.
                misorient_angles[0]
                * np.exp(
                    np.linspace(0, timestop, n_timesteps)
                    * (np.cos(2 * np.deg2rad(misorient_angles[0])) - 1)
                ),
            )

            # Check alignment and texture strength (half way and final value).
            halfway = round(n_timesteps / 2)
            match params["gbs_threshold"]:
                case 0:
                    assert np.isclose(misorient_angles[halfway], 11, atol=1.5)
                    assert np.isclose(misorient_angles[-1], 8, atol=1.25)
                    assert np.isclose(misorient_indices[halfway], 0.975, atol=0.075)
                    assert np.isclose(misorient_indices[-1], 0.99, atol=0.03)
                case 0.2:
                    assert np.isclose(misorient_angles[halfway], 13, atol=2.05)
                    assert np.isclose(misorient_angles[-1], 11, atol=1.5)
                    assert np.isclose(misorient_indices[halfway], 0.755, atol=0.08)
                    assert np.isclose(misorient_indices[-1], 0.755, atol=0.075)
                case 0.4:
                    assert np.isclose(misorient_angles[halfway], 19, atol=2.0)
                    assert np.isclose(misorient_angles[-1], 16, atol=2.25)
                    assert misorient_indices[halfway] < 0.75
                    assert misorient_indices[-1] < 0.7

            # Optionally store plotting metadata.
            if outdir is not None:
                labels.append(f"$f_{{gbs}}$ = {params['gbs_threshold']}")
                angles.append(misorient_angles)
                indices.append(misorient_indices)

        # Optionally plot figure.
        if outdir is not None:
            _vis.simple_shear_2d(
                angles,
                indices,
                timestop=timestop,
                savefile=f"{outdir}/simple_shearXZ_single_olivineA_initQ1.png",
                markers=("o", "v", "s"),
                labels=labels,
                refval=45,
            )
