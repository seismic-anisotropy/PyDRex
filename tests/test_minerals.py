import random

import pytest
import numpy as np
from scipy.spatial.transform import Rotation

import pydrex.minerals as _minerals
import pydrex.deformation_mechanism as _defmech


def test_minerals_random_init(olivine_disl_random_500, enstatite_disl_random_500):
    for mineral in olivine_disl_random_500:
        assert mineral.fractions is not None
        assert mineral.orientations is not None
        assert np.allclose(mineral._fractions_init, mineral.fractions)
        assert np.allclose(mineral._orientations_init, mineral.orientations)
    for mineral in enstatite_disl_random_500:
        assert mineral.fractions is not None
        assert mineral.orientations is not None
        assert np.allclose(mineral._fractions_init, mineral.fractions)
        assert np.allclose(mineral._orientations_init, mineral.orientations)


def test_update_orientations_noop(
    mock_params_Fraters2021, olivine_disl_random_500, enstatite_disl_random_500
):
    # Translation in X, no CPO.
    velocity_gradient = np.array([[1e-5, 0, 0], [0, 0, 0], [0, 0, 0]])
    strain_rate = np.array([[5e-6, 0, 0], [0, 0, 0], [0, 0, 0]])
    strain_rate_max = 5e-6
    nondim_strain_rate = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    nondim_velocity_gradient = np.array([[2.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # One particle for each fabric type of each mineral phase.
    for olivine in olivine_disl_random_500:
        olivine.update_orientations(
            nondim_strain_rate,
            strain_rate_max,
            nondim_velocity_gradient,
            mock_params_Fraters2021,
            1000,
        )
        assert np.isclose(olivine.fractions.sum(), 1, atol=1e-10)
        assert np.allclose(olivine.fractions, olivine._fractions_init)
        assert np.allclose(olivine.orientations, olivine._orientations_init)
    for enstatite in enstatite_disl_random_500:
        enstatite.update_orientations(
            nondim_strain_rate,
            strain_rate_max,
            nondim_velocity_gradient,
            mock_params_Fraters2021,
            1000,
        )
        assert np.isclose(enstatite.fractions.sum(), 1, atol=1e-10)
        assert np.allclose(enstatite.fractions, enstatite._fractions_init)
        assert np.allclose(enstatite.orientations, enstatite._orientations_init)


@pytest.mark.wip
def test_update_orientations_olivine_A(
    mock_params_Kaminski2004_fig4_circles,
    mock_params_Kaminski2004_fig4_squares,
    mock_params_Kaminski2004_fig4_triangles,
    olivine_A_disl_random_500,
):
    # Simple shear in XY plane.
    velocity_gradient = np.array([[0, 0, 1e-5], [0, 0, 0], [0, 0, 0]])
    strain_rate = np.array([[0, 0, 5e-6], [0, 0, 0], [5e-6, 0, 0]])
    strain_rate_max = 5e-6
    nondim_strain_rate = np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])
    nondim_velocity_gradient = np.array([[0, 0, 2.0], [0, 0, 0], [0, 0, 0]])
    # Iterate 1000 times until end_time; end_time * strain_rate_max == 1
    for dt in np.diff(np.linspace(0, 2e5 + 1000, 1001)):
        # Pure olivine aggregate with 500 grains.
        mineral = olivine_A_disl_random_500
        mineral.update_orientations(
            nondim_strain_rate,
            strain_rate_max,
            nondim_velocity_gradient,
            mock_params_Kaminski2004_fig4_circles,
            dt,
        )
        assert np.isclose(mineral.fractions.sum(), 1, atol=1e-10)
        # Project grain orientations onto shear plane.
        # Extract mean angle (use mineral.fractions for weighting)
        # Should be -45±2° (fig. 4 Kaminski 2004, circles at dimensionless time = 1)
        orientations = Rotation.from_matrix(mineral.orientations)
        mean_orientation = orientations.mean(mineral.fractions)
        print(mean_orientation.as_rotvec(degrees=True))
        print(mean_orientation.as_euler("zxz", degrees=True))
        assert np.isclose(mean_orientation.as_rotvec(degrees=True)[-1], -45, atol=2)
