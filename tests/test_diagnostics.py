import pytest
import numpy as np
from numpy import random as rn
from scipy.spatial.transform import Rotation

from pydrex import diagnostics as _diagnostics


class TestBinghamStats:
    """Tests for antipodally symmetric (Bingham) statistics."""

    def test_average_0(self):
        """Test bingham average of vectors aligned to the reference frame."""
        orientations = Rotation.from_rotvec([[0, 0, 0]] * 10).as_matrix()
        a_mean, b_mean, c_mean = _diagnostics.bingham_average(orientations)
        assert np.array_equal(a_mean, [1, 0, 0])
        assert np.array_equal(b_mean, [0, 1, 0])
        assert np.array_equal(c_mean, [0, 0, 1])

    def test_average_twopoles90Z(self):
        """Test bingham average of vectors rotated by ±90° around Z."""
        orientations = Rotation.from_rotvec(
            [
                [0, 0, -np.pi / 2],
                [0, 0, np.pi / 2],
            ]
        ).as_matrix()
        a_mean, b_mean, c_mean = _diagnostics.bingham_average(orientations)
        assert np.array_equal(a_mean, [0, 1, 0])
        assert np.array_equal(b_mean, [1, 0, 0])
        assert np.array_equal(c_mean, [0, 0, 1])

    def test_average_spread10X(self):
        """Test bingham average of vectors spread within 10° of the ±X-axis."""
        orientations = Rotation.from_rotvec(
            np.stack(
                [
                    [0, x * np.pi / 18 - np.pi / 36, x * np.pi / 18 - np.pi / 36]
                    for x in rn.random_sample(100)
                ]
            )
        ).as_matrix()
        a_mean, b_mean, c_mean = _diagnostics.bingham_average(orientations)
        print(a_mean, b_mean, c_mean)
        assert np.allclose(np.abs(a_mean), [1.0, 0, 0], atol=np.sin(np.pi / 18))
        assert np.allclose(np.abs(b_mean), [0, 1.0, 0], atol=np.sin(np.pi / 18))
        assert np.allclose(np.abs(c_mean), [0, 0, 1.0], atol=np.sin(np.pi / 18))


class TestMIndex:
    """Tests for the M-index texture strength diagnostic."""

    def test_texture_uniform(self):
        """Test M-index for random (uniform distribution) grain orientations."""
        orientations = Rotation.random(1000).as_matrix()
        assert np.isclose(_diagnostics.M_index(orientations), 0.44, atol=1e-2)

    def test_texture_spread10X(self):
        """Test M-index for grains spread within 10° of the ±X axis."""
        orientations = Rotation.from_rotvec(
            np.stack(
                [
                    [0, x * np.pi / 18 - np.pi / 36, x * np.pi / 18 - np.pi / 36]
                    for x in rn.random_sample(100)
                ]
            )
        ).as_matrix()
        assert np.isclose(_diagnostics.M_index(orientations), 0.99, atol=1e-2)

    def test_texture_spread30X(self):
        """Test M-index for grains spread within 45° of the ±X axis."""
        orientations = Rotation.from_rotvec(
            np.stack(
                [
                    [0, x * np.pi / 4 - np.pi / 8, x * np.pi / 4 - np.pi / 8]
                    for x in rn.random_sample(100)
                ]
            )
        ).as_matrix()
        assert np.isclose(_diagnostics.M_index(orientations), 0.84, atol=0.4e-1)

    def test_textures_increasing(self):
        """Test M-index for textures of increasing strength."""
        M_vals = np.empty(4)
        factors = (2, 4, 8, 16)
        for i, factor in enumerate(factors):
            orientations = Rotation.from_rotvec(
                np.stack(
                    [
                        [
                            0,
                            x * np.pi / factor - np.pi / factor / 2,
                            x * np.pi / factor - np.pi / factor / 2,
                        ]
                        for x in rn.random_sample(500)
                    ]
                )
            ).as_matrix()
            M_vals[i] = _diagnostics.M_index(orientations)
        assert np.all(
            np.diff(M_vals) >= 0
        ), f"M-index values {M_vals} are not increasing"
