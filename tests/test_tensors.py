"""> PyDRex: Tests for tensor operations."""
import numpy as np

from pydrex.minerals import OLIVINE_STIFFNESS, ENSTATITE_STIFFNESS
from pydrex import tensors as _tensors


def test_voigt_tensor():
    """Test elasticity tensor <-> 6x6 Voigt matrix conversions."""
    olivine_tensor = np.array(
        [
            [
                [[320.71, 0.0, 0.0], [0.0, 69.84, 0.0], [0.0, 0.0, 71.22]],
                [[0.0, 78.36, 0.0], [78.36, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 77.67], [0.0, 0.0, 0.0], [77.67, 0.0, 0.0]],
            ],
            [
                [[0.0, 78.36, 0.0], [78.36, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[69.84, 0.0, 0.0], [0.0, 197.25, 0.0], [0.0, 0.0, 74.8]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 63.77], [0.0, 63.77, 0.0]],
            ],
            [
                [[0.0, 0.0, 77.67], [0.0, 0.0, 0.0], [77.67, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 63.77], [0.0, 63.77, 0.0]],
                [[71.22, 0.0, 0.0], [0.0, 74.8, 0.0], [0.0, 0.0, 234.32]],
            ],
        ]
    )
    np.testing.assert_array_equal(
        _tensors.voigt_to_elastic_tensor(OLIVINE_STIFFNESS),
        olivine_tensor,
    )
    np.testing.assert_array_equal(
        _tensors.elastic_tensor_to_voigt(
            _tensors.voigt_to_elastic_tensor(OLIVINE_STIFFNESS)
        ),
        OLIVINE_STIFFNESS,
    )
    np.testing.assert_array_equal(
        _tensors.voigt_to_elastic_tensor(
            _tensors.elastic_tensor_to_voigt(olivine_tensor)
        ),
        olivine_tensor,
    )
    np.testing.assert_array_equal(
        _tensors.elastic_tensor_to_voigt(
            _tensors.voigt_to_elastic_tensor(ENSTATITE_STIFFNESS)
        ),
        ENSTATITE_STIFFNESS,
    )


def test_voigt_to_vector():
    """Test Voigt vector construction."""
    np.testing.assert_allclose(
        _tensors.voigt_matrix_to_vector(ENSTATITE_STIFFNESS),
        np.array(
            [
                236.9,
                180.5,
                230.4,
                80.32733034,
                89.37829714,
                112.57139956,
                168.6,
                158.8,
                160.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        OLIVINE_STIFFNESS,
        _tensors.voigt_vector_to_matrix(
            _tensors.voigt_matrix_to_vector(OLIVINE_STIFFNESS),
        ),
    )
    r = np.random.rand(6, 6)
    np.testing.assert_array_equal(
        _tensors.voigt_matrix_to_vector(r),
        np.array(
            [
                r[0, 0],
                r[1, 1],
                r[2, 2],
                np.sqrt(2) * r[1, 2],
                np.sqrt(2) * r[2, 0],
                np.sqrt(2) * r[0, 1],
                2 * r[3, 3],
                2 * r[4, 4],
                2 * r[5, 5],
                2 * r[0, 3],
                2 * r[1, 4],
                2 * r[2, 5],
                2 * r[2, 3],
                2 * r[0, 4],
                2 * r[1, 5],
                2 * r[1, 3],
                2 * r[2, 4],
                2 * r[0, 5],
                2 * np.sqrt(2) * r[4, 5],
                2 * np.sqrt(2) * r[5, 3],
                2 * np.sqrt(2) * r[3, 4],
            ]
        ),
    )
