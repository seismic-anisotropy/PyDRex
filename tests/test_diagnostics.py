"""PyDRex: tests for texture diagnostics."""
import numpy as np
from numpy import random as rn
from scipy.spatial.transform import Rotation

from pydrex import diagnostics as _diagnostics


class TestSymmetryPGR:
    """Test Point-Girdle-Random (eigenvalue) symmetry diagnostics."""

    def test_pointX(self):
        """Test diagnostics of point symmetry aligned to the X axis."""
        # Initial orientations within 10°.
        orientations = (
            Rotation.from_rotvec(
                np.stack(
                    [
                        [0, x * np.pi / 18 - np.pi / 36, x * np.pi / 18 - np.pi / 36]
                        for x in rn.default_rng().random(100)
                    ]
                )
            )
            .inv()
            .as_matrix()
        )
        np.testing.assert_allclose(
            _diagnostics.symmetry(orientations, axis="a"), (1, 0, 0), atol=0.05
        )

    def test_random(self):
        """Test diagnostics of random grain orientations."""
        orientations = Rotation.random(1000).as_matrix()
        np.testing.assert_allclose(
            _diagnostics.symmetry(orientations, axis="a"), (0, 0, 1), atol=0.15
        )

    # TODO: More symmetry tests.


class TestVolumeWeighting:
    """Tests for volumetric resampling of orientation data."""

    def test_upsample(self):
        """Test upsampling of the raw orientation data."""
        orientations = (
            Rotation.from_rotvec(
                [
                    [0, 0, 0],
                    [0, 0, np.pi / 6],
                    [np.pi / 6, 0, 0],
                ]
            )
            .inv()
            .as_matrix()
        )
        fractions = np.array([0.25, 0.6, 0.15])
        new_orientations = _diagnostics.resample_orientations(
            orientations,
            fractions,
            25,
        )
        assert np.all(a in orientations for a in new_orientations)

    def test_downsample(self):
        """Test downsampling of orientation data."""
        orientations = (
            Rotation.from_rotvec(
                [
                    [0, 0, 0],
                    [0, 0, np.pi / 6],
                    [np.pi / 6, 0, 0],
                ]
            )
            .inv()
            .as_matrix()
        )
        fractions = np.array([0.25, 0.6, 0.15])
        new_orientations = _diagnostics.resample_orientations(
            orientations,
            fractions,
            2,
        )
        assert np.all(a in orientations for a in new_orientations)


class TestBinghamStats:
    """Tests for antipodally symmetric (bingham) statistics."""

    def test_average_0(self):
        """Test bingham average of vectors aligned to the reference frame."""
        orientations = Rotation.from_rotvec([[0, 0, 0]] * 10).inv().as_matrix()
        a_mean = _diagnostics.bingham_average(orientations, axis="a")
        b_mean = _diagnostics.bingham_average(orientations, axis="b")
        c_mean = _diagnostics.bingham_average(orientations, axis="c")
        np.testing.assert_array_equal(a_mean, [1, 0, 0])
        np.testing.assert_array_equal(b_mean, [0, 1, 0])
        np.testing.assert_array_equal(c_mean, [0, 0, 1])

    def test_average_twopoles90Z(self):
        """Test bingham average of vectors rotated by ±90° around Z."""
        orientations = (
            Rotation.from_rotvec(
                [
                    [0, 0, -np.pi / 2],
                    [0, 0, np.pi / 2],
                ]
            )
            .inv()
            .as_matrix()
        )
        a_mean = _diagnostics.bingham_average(orientations, axis="a")
        b_mean = _diagnostics.bingham_average(orientations, axis="b")
        c_mean = _diagnostics.bingham_average(orientations, axis="c")
        np.testing.assert_array_equal(a_mean, [0, 1, 0])
        np.testing.assert_array_equal(b_mean, [1, 0, 0])
        np.testing.assert_array_equal(c_mean, [0, 0, 1])

    def test_average_spread10X(self):
        """Test bingham average of vectors spread within 10° of the ±X-axis."""
        orientations = (
            Rotation.from_rotvec(
                np.stack(
                    [
                        [0, x * np.pi / 18 - np.pi / 36, x * np.pi / 18 - np.pi / 36]
                        for x in rn.default_rng().random(100)
                    ]
                )
            )
            .inv()
            .as_matrix()
        )
        a_mean = _diagnostics.bingham_average(orientations, axis="a")
        b_mean = _diagnostics.bingham_average(orientations, axis="b")
        c_mean = _diagnostics.bingham_average(orientations, axis="c")
        np.testing.assert_allclose(np.abs(a_mean), [1.0, 0, 0], atol=np.sin(np.pi / 18))
        np.testing.assert_allclose(np.abs(b_mean), [0, 1.0, 0], atol=np.sin(np.pi / 18))
        np.testing.assert_allclose(np.abs(c_mean), [0, 0, 1.0], atol=np.sin(np.pi / 18))


class TestMIndex:
    """Tests for the M-index texture strength diagnostic."""

    def test_texture_uniform(self):
        """Test M-index for random (uniform distribution) grain orientations."""
        orientations = Rotation.random(1000).as_matrix()
        assert np.isclose(
            _diagnostics.misorientation_index(orientations), 0.38, atol=1e-2
        )

    def test_texture_spread10X(self):
        """Test M-index for grains spread within 10° of the ±X axis."""
        orientations = (
            Rotation.from_rotvec(
                np.stack(
                    [
                        [0, x * np.pi / 18 - np.pi / 36, x * np.pi / 18 - np.pi / 36]
                        for x in rn.default_rng().random(100)
                    ]
                )
            )
            .inv()
            .as_matrix()
        )
        assert np.isclose(
            _diagnostics.misorientation_index(orientations), 0.99, atol=1e-2
        )

    def test_texture_spread30X(self):
        """Test M-index for grains spread within 45° of the ±X axis."""
        orientations = (
            Rotation.from_rotvec(
                np.stack(
                    [
                        [0, x * np.pi / 4 - np.pi / 8, x * np.pi / 4 - np.pi / 8]
                        for x in rn.default_rng().random(100)
                    ]
                )
            )
            .inv()
            .as_matrix()
        )
        assert np.isclose(
            _diagnostics.misorientation_index(orientations), 0.8, atol=0.1
        )

    def test_textures_increasing(self):
        """Test M-index for textures of increasing strength."""
        M_vals = np.empty(4)
        factors = (2, 4, 8, 16)
        for i, factor in enumerate(factors):
            orientations = (
                Rotation.from_rotvec(
                    np.stack(
                        [
                            [
                                0,
                                x * np.pi / factor - np.pi / factor / 2,
                                x * np.pi / factor - np.pi / factor / 2,
                            ]
                            for x in rn.default_rng().random(500)
                        ]
                    )
                )
                .inv()
                .as_matrix()
            )
            M_vals[i] = _diagnostics.misorientation_index(orientations)
        assert np.all(
            np.diff(M_vals) >= 0
        ), f"M-index values {M_vals} are not increasing"
