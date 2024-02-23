"""> PyDRex: tests for texture diagnostics."""

import numpy as np
import pytest
from numpy import random as rn
from scipy.spatial.transform import Rotation

from pydrex import diagnostics as _diagnostics
from pydrex import geometry as _geo
from pydrex import stats as _stats


class TestElasticityComponents:
    """Test symmetry decomposition of elastic tensors."""

    def test_olivine_Browaeys2004(self):
        C = np.array(
            [
                [192, 66, 60, 0, 0, 0],
                [66, 160, 56, 0, 0, 0],
                [60, 56, 272, 0, 0, 0],
                [0, 0, 0, 60, 0, 0],
                [0, 0, 0, 0, 62, 0],
                [0, 0, 0, 0, 0, 49],
            ]
        )
        out = _diagnostics.elasticity_components([C])
        # FIXME: How do they get 15.2% for hexagonal? It isn't the ratio of the norms
        # nor the ratio of the squared norms...
        expected = {
            "bulk_modulus": 109.8,
            "shear_modulus": 63.7,
            "percent_anisotropy": 20.7,
            # "percent_hexagonal": 15.2,
            # "percent_tetragonal": 0.625,
            # "percent_orthorhombic": 4.875,
            "percent_monoclinic": 0,
            "percent_triclinic": 0,
        }
        for k, v in expected.items():
            assert np.isclose(out[k][0], v, atol=0.1, rtol=0), f"{k}: {out[k]} != {v}"
        np.testing.assert_allclose(out["hexagonal_axis"][0], [0, 0, 1])

    def test_enstatite_Browaeys2004(self):
        C = np.array(
            [
                [225, 54, 72, 0, 0, 0],
                [54, 214, 53, 0, 0, 0],
                [72, 53, 178, 0, 0, 0],
                [0, 0, 0, 78, 0, 0],
                [0, 0, 0, 0, 82, 0],
                [0, 0, 0, 0, 0, 76],
            ]
        )
        out = _diagnostics.elasticity_components([C])
        # FIXME: Test remaining percentages when I figure out how they get the values.
        expected = {
            "bulk_modulus": 108.3,
            "shear_modulus": 76.4,
            "percent_anisotropy": 9.2,
            # "percent_hexagonal": 4.3,
            # "percent_tetragonal": ?,  # + ortho = 4.9
            # "percent_orthorhombic": ?,  # + tetra = 4.9
            "percent_monoclinic": 0,
            "percent_triclinic": 0,
        }
        for k, v in expected.items():
            assert np.isclose(out[k][0], v, atol=0.1, rtol=0), f"{k}: {out[k]} != {v}"
        np.testing.assert_allclose(out["hexagonal_axis"][0], [0, 0, 1])


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
            _diagnostics.symmetry_pgr(orientations, axis="a"), (1, 0, 0), atol=0.05
        )

    def test_random(self):
        """Test diagnostics of random grain orientations."""
        orientations = Rotation.random(1000).as_matrix()
        np.testing.assert_allclose(
            _diagnostics.symmetry_pgr(orientations, axis="a"), (0, 0, 1), atol=0.15
        )

    def test_girdle(self):
        """Test diagnostics of girdled orientations."""
        rng = rn.default_rng()
        a = np.zeros(1000)
        b = np.zeros(1000)
        c = rng.normal(0, 1.0, size=1000)
        d = rng.normal(0, 1.0, size=1000)
        orientations = Rotation.from_quat(np.column_stack([a, b, c, d])).as_matrix()
        np.testing.assert_allclose(
            _diagnostics.symmetry_pgr(orientations, axis="a"), (0, 1, 0), atol=0.1
        )


class TestVolumeWeighting:
    """Tests for volumetric resampling of orientation data."""

    def test_output_shape(self):
        """Test that we get the correct output shape."""
        orientations = [
            Rotation.random(1000).as_matrix(),
            Rotation.random(1000).as_matrix(),
        ]
        fractions = [np.full(1000, 1 / 1000), np.full(1000, 1 / 1000)]
        new_orientations, new_fractions = _stats.resample_orientations(
            orientations, fractions
        )
        np.testing.assert_array_equal(
            np.asarray(orientations).shape, new_orientations.shape
        )
        np.testing.assert_array_equal(np.asarray(fractions).shape, new_fractions.shape)
        new_orientations, new_fractions = _stats.resample_orientations(
            orientations, fractions, n_samples=500
        )
        assert new_orientations.shape[1] == 500
        assert new_fractions.shape[1] == 500

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
        new_orientations, _ = _stats.resample_orientations(
            [orientations],
            [fractions],
            25,
        )
        assert np.all(a in orientations for a in new_orientations[0])

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
        new_orientations = _stats.resample_orientations(
            [orientations],
            [fractions],
            2,
        )
        assert np.all(a in orientations for a in new_orientations[0])

    def test_common_input_errors(self):
        """Test that exceptions are raised for bad input data."""
        orientations = Rotation.from_rotvec(
            [
                [0, 0, 0],
                [0, 0, np.pi / 6],
                [np.pi / 6, 0, 0],
            ]
        )
        fractions = np.array([0.25, 0.6, 0.15])
        with pytest.raises(ValueError):
            # SciPy Rotation instance is not valid input.
            _stats.resample_orientations(orientations, fractions)
            # Input must be a stack of orientations.
            _stats.resample_orientations(orientations.as_matrix(), fractions)
            # Input must be a stack of fractions.
            _stats.resample_orientations([orientations.as_matrix()], fractions)
            # First two dimensions of inputs must match.
            _stats.resample_orientations([orientations.as_matrix()], [fractions[:-1]])

        # This is the proper way to do it:
        _stats.resample_orientations([orientations.as_matrix()], [fractions])


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

    # TODO: Add tests for other (non-orthorhombic) lattice symmetries.

    def test_texture_uniform_ortho(self, seed):
        """Test with random (uniform distribution) orthorhombic grain orientations."""
        orientations = Rotation.random(1000, random_state=seed).as_matrix()
        assert np.isclose(
            _diagnostics.misorientation_index(
                orientations, _geo.LatticeSystem.orthorhombic
            ),
            0.05,
            atol=1e-2,
            rtol=0,
        )

    def test_texture_spread10X_ortho(self, seed):
        """Test for orthorhombic grains spread within 10° of the ±X axis."""
        orientations = (
            Rotation.from_rotvec(
                np.stack(
                    [
                        [0, x * np.pi / 18 - np.pi / 36, x * np.pi / 18 - np.pi / 36]
                        for x in rn.default_rng(seed=seed).random(1000)
                    ]
                )
            )
            .inv()
            .as_matrix()
        )
        assert np.isclose(
            _diagnostics.misorientation_index(
                orientations, _geo.LatticeSystem.orthorhombic
            ),
            0.99,
            atol=1e-2,
            rtol=0,
        )

    def test_texture_spread45X_ortho(self, seed):
        """Test for orthorhombic grains spread within 45° of the ±X axis."""
        orientations = Rotation.from_rotvec(
            np.stack(
                [
                    [0, x * np.pi / 2 - np.pi / 4, x * np.pi / 2 - np.pi / 4]
                    for x in rn.default_rng(seed=seed).random(1000)
                ]
            )
        ).as_matrix()
        assert np.isclose(
            _diagnostics.misorientation_index(
                orientations, _geo.LatticeSystem.orthorhombic
            ),
            0.81,
            atol=1e-2,
            rtol=0,
        )

    def test_textures_increasing_ortho(self, seed):
        """Test M-index for textures of increasing strength."""
        M_vals = np.empty(4)
        factors = (2, 3, 4, 5)
        for i, factor in enumerate(factors):
            orientations = Rotation.from_rotvec(
                np.stack(
                    [
                        [
                            0,
                            x * np.pi / factor - np.pi / factor / 2,
                            x * np.pi / factor - np.pi / factor / 2,
                        ]
                        for x in rn.default_rng(seed=seed).random(1000)
                    ]
                )
            ).as_matrix()
            M_vals[i] = _diagnostics.misorientation_index(
                orientations, _geo.LatticeSystem.orthorhombic
            )
        assert np.all(
            np.diff(M_vals) >= 0
        ), f"M-index values {M_vals} are not increasing"

    def test_texture_girdle_ortho(self, seed):
        """Test M-index for girdled texture."""
        rng = rn.default_rng(seed=seed)
        a = np.zeros(1000)
        b = np.zeros(1000)
        c = rng.normal(0, 1.0, size=1000)
        d = rng.normal(0, 1.0, size=1000)
        orientations = Rotation.from_quat(np.column_stack([a, b, c, d])).as_matrix()
        assert np.isclose(
            _diagnostics.misorientation_index(
                orientations, _geo.LatticeSystem.orthorhombic
            ),
            0.67,
            atol=1e-2,
        )
