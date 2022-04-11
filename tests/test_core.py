import pytest
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import linalg as la

import pydrex.core as _core
import pydrex.minerals as _minerals
import pydrex.deformation_mechanism as _defmech


#def test_update_strain():
#    ...


class TestSimpleShearSingleGrains:
    """Single-grain analytical re-orientation rate tests."""

    # Kaminski used passive rotations (see Kaminski 2001 eq. 1) and ZXZ convention.
    # Scipy gives us active rotations by default,
    # and for composite rotations the `from_rotvec` method uses the XYZ convention.
    # Therefore, for composite rotations we should use `from_euler("ZXZ", ...)`.
    # This will give us active rotations in the ZXZ convention, i.e.
    #
    #            c1c3 - c2s1s3 |  -c1s3 - c2c3s1 |   s1s2
    # Z1 X1 Z2 = c3s1 + c2c3s1 |   c1c2c3 - s1s3 |  -c1s2
    #            s2s3          |   c3s2          |   c2
    #
    # where ci = cos(angle of i'th elementary rotation), si = sin(...),
    # which is the same as the passive version but transposed.

    def test_simple_shear_initP10Z(self):
        # Single grain of olivine A-type, simple shear with:
        #     0 0 0  .   0 1 0      0 -1 0
        # L = 2 0 0  ε = 1 0 0  Ω = 1  0 0
        #     0 0 0      0 0 0      0  0 0
        nondim_velocity_gradient = np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        # Grain initialised with rotation around Z (active rotation convention).
        θ = np.deg2rad(10)
        initial_orientations = Rotation.from_rotvec([[0, 0, θ]])
        orientations_diff, fractions_diff = _core.derivatives(
            phase=_minerals.MineralPhase.olivine,
            fabric=_minerals.OlivineFabric.A,
            n_grains=1,
            orientations=initial_orientations.as_matrix(),
            fractions=np.array([1.0]),
            strain_rate=nondim_strain_rate,
            velocity_gradient=nondim_velocity_gradient,
            stress_exponent=3.5,
            dislocation_exponent=1.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2*θ)
        sinθ = np.sin(θ)
        print("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array([
            [sinθ * (1 + cos2θ), cosθ * (1 + cos2θ), 0],
            [cosθ * (- 1 - cos2θ), sinθ * (1 + cos2θ), 0],
            [0, 0, 0],
        ])
        print("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        print("calculated grain volume change:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_simple_shear_initN10Z(self):
        # Single grain of olivine A-type, simple shear with:
        #     0 0 0  .   0 1 0      0 -1 0
        # L = 2 0 0  ε = 1 0 0  Ω = 1  0 0
        #     0 0 0      0 0 0      0  0 0
        nondim_velocity_gradient = np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        # Grain initialised with rotation around Z (active rotation convention).
        θ = np.deg2rad(-10)
        initial_orientations = Rotation.from_rotvec([[0, 0, θ]])
        orientations_diff, fractions_diff = _core.derivatives(
            phase=_minerals.MineralPhase.olivine,
            fabric=_minerals.OlivineFabric.A,
            n_grains=1,
            orientations=initial_orientations.as_matrix(),
            fractions=np.array([1.0]),
            strain_rate=nondim_strain_rate,
            velocity_gradient=nondim_velocity_gradient,
            stress_exponent=3.5,
            dislocation_exponent=1.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2*θ)
        sinθ = np.sin(θ)
        print("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array([
            [sinθ * (1 + cos2θ), cosθ * (1 + cos2θ), 0],
            [cosθ * (- 1 - cos2θ), sinθ * (1 + cos2θ), 0],
            [0, 0, 0],
        ])
        print("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        print("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_simple_shear_initP45Z(self):
        # Single grain of olivine A-type, simple shear with:
        #     0 0 0  .   0 1 0      0 -1 0
        # L = 2 0 0  ε = 1 0 0  Ω = 1  0 0
        #     0 0 0      0 0 0      0  0 0
        nondim_velocity_gradient = np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        # Grain initialised with rotation around Z (active rotation convention).
        θ = np.deg2rad(45)
        initial_orientations = Rotation.from_rotvec([[0, 0, θ]])
        orientations_diff, fractions_diff = _core.derivatives(
            phase=_minerals.MineralPhase.olivine,
            fabric=_minerals.OlivineFabric.A,
            n_grains=1,
            orientations=initial_orientations.as_matrix(),
            fractions=np.array([1.0]),
            strain_rate=nondim_strain_rate,
            velocity_gradient=nondim_velocity_gradient,
            stress_exponent=3.5,
            dislocation_exponent=1.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2*θ)
        sinθ = np.sin(θ)
        print("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array([
            [sinθ * (1 + cos2θ), cosθ * (1 + cos2θ), 0],
            [cosθ * (- 1 - cos2θ), sinθ * (1 + cos2θ), 0],
            [0, 0, 0],
        ])
        print("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        print("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_simple_shear_initP90Z(self):
        # Single grain of olivine A-type, simple shear with:
        #     0 0 0  .   0 1 0      0 -1 0
        # L = 2 0 0  ε = 1 0 0  Ω = 1  0 0
        #     0 0 0      0 0 0      0  0 0
        nondim_velocity_gradient = np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        # Grain initialised with rotation around Z (active rotation convention).
        θ = np.deg2rad(90)
        initial_orientations = Rotation.from_rotvec([[0, 0, θ]])
        orientations_diff, fractions_diff = _core.derivatives(
            phase=_minerals.MineralPhase.olivine,
            fabric=_minerals.OlivineFabric.A,
            n_grains=1,
            orientations=initial_orientations.as_matrix(),
            fractions=np.array([1.0]),
            strain_rate=nondim_strain_rate,
            velocity_gradient=nondim_velocity_gradient,
            stress_exponent=3.5,
            dislocation_exponent=1.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2*θ)
        sinθ = np.sin(θ)
        print("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array([
            [sinθ * (1 + cos2θ), cosθ * (1 + cos2θ), 0],
            [cosθ * (- 1 - cos2θ), sinθ * (1 + cos2θ), 0],
            [0, 0, 0],
        ])
        print("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        print("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_simple_shear_initP10Y(self):
        # Single grain of olivine A-type, simple shear with:
        #     0 0 2  .   0 0 1       0 0 1
        # L = 0 0 0  ε = 0 0 0  Ω =  0 0 0
        #     0 0 0      1 0 0      -1 0 0
        nondim_velocity_gradient = np.array([[0, 0, 2.0], [0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])
        # Grain initialised with rotation around Y (active rotation convention).
        θ = np.deg2rad(10)
        initial_orientations = Rotation.from_rotvec([[0, θ, 0]])
        orientations_diff, fractions_diff = _core.derivatives(
            phase=_minerals.MineralPhase.olivine,
            fabric=_minerals.OlivineFabric.A,
            n_grains=1,
            orientations=initial_orientations.as_matrix(),
            fractions=np.array([1.0]),
            strain_rate=nondim_strain_rate,
            velocity_gradient=nondim_velocity_gradient,
            stress_exponent=3.5,
            dislocation_exponent=1.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2*θ)
        sinθ = np.sin(θ)
        print("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array([
            [sinθ * (1 - cos2θ), 0, cosθ * (cos2θ - 1)],
            [0, 0, 0],
            [cosθ * (1 - cos2θ), 0, sinθ * (1 - cos2θ)],
        ])
        print("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        print("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_simple_shear_initN10Y(self):
        # Single grain of olivine A-type, simple shear with:
        #     0 0 2  .   0 0 1       0 0 1
        # L = 0 0 0  ε = 0 0 0  Ω =  0 0 0
        #     0 0 0      1 0 0      -1 0 0
        nondim_velocity_gradient = np.array([[0, 0, 2.0], [0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])
        # Grain initialised with rotation around Y (active rotation convention).
        θ = np.deg2rad(-10)
        initial_orientations = Rotation.from_rotvec([[0, θ, 0]])
        orientations_diff, fractions_diff = _core.derivatives(
            phase=_minerals.MineralPhase.olivine,
            fabric=_minerals.OlivineFabric.A,
            n_grains=1,
            orientations=initial_orientations.as_matrix(),
            fractions=np.array([1.0]),
            strain_rate=nondim_strain_rate,
            velocity_gradient=nondim_velocity_gradient,
            stress_exponent=3.5,
            dislocation_exponent=1.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2*θ)
        sinθ = np.sin(θ)
        print("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array([
            [sinθ * (1 - cos2θ), 0, cosθ * (cos2θ - 1)],
            [0, 0, 0],
            [cosθ * (1 - cos2θ), 0, sinθ * (1 - cos2θ)],
        ])
        print("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        print("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_simple_shear_initP45Y(self):
        # Single grain of olivine A-type, simple shear with:
        #     0 0 2  .   0 0 1       0 0 1
        # L = 0 0 0  ε = 0 0 0  Ω =  0 0 0
        #     0 0 0      1 0 0      -1 0 0
        nondim_velocity_gradient = np.array([[0, 0, 2.0], [0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])
        # Grain initialised with rotation around Y (active rotation convention).
        θ = np.deg2rad(45)
        initial_orientations = Rotation.from_rotvec([[0, θ, 0]])
        orientations_diff, fractions_diff = _core.derivatives(
            phase=_minerals.MineralPhase.olivine,
            fabric=_minerals.OlivineFabric.A,
            n_grains=1,
            orientations=initial_orientations.as_matrix(),
            fractions=np.array([1.0]),
            strain_rate=nondim_strain_rate,
            velocity_gradient=nondim_velocity_gradient,
            stress_exponent=3.5,
            dislocation_exponent=1.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2*θ)
        sinθ = np.sin(θ)
        print("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array([
            [sinθ * (1 - cos2θ), 0, cosθ * (cos2θ - 1)],
            [0, 0, 0],
            [cosθ * (1 - cos2θ), 0, sinθ * (1 - cos2θ)],
        ])
        print("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        print("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_simple_shear_initP90Y(self):
        # Single grain of olivine A-type, simple shear with:
        #     0 0 2  .   0 0 1       0 0 1
        # L = 0 0 0  ε = 0 0 0  Ω =  0 0 0
        #     0 0 0      1 0 0      -1 0 0
        nondim_velocity_gradient = np.array([[0, 0, 2.0], [0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])
        # Grain initialised with rotation around Y (active rotation convention).
        θ = np.deg2rad(90)
        initial_orientations = Rotation.from_rotvec([[0, θ, 0]])
        orientations_diff, fractions_diff = _core.derivatives(
            phase=_minerals.MineralPhase.olivine,
            fabric=_minerals.OlivineFabric.A,
            n_grains=1,
            orientations=initial_orientations.as_matrix(),
            fractions=np.array([1.0]),
            strain_rate=nondim_strain_rate,
            velocity_gradient=nondim_velocity_gradient,
            stress_exponent=3.5,
            dislocation_exponent=1.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2*θ)
        sinθ = np.sin(θ)
        print("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array([
            [sinθ * (1 - cos2θ), 0, cosθ * (cos2θ - 1)],
            [0, 0, 0],
            [cosθ * (1 - cos2θ), 0, sinθ * (1 - cos2θ)],
        ])
        print("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        print("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_simple_shear_random(self):
        # Single grain of olivine A-type with random initial orientation:
        #     0 0 0  .   0 1 0      0 -1 0
        # L = 2 0 0  ε = 1 0 0  Ω = 1  0 0
        #     0 0 0      0 0 0      0  0 0
        nondim_velocity_gradient = np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        # Grain initialised with random rotation.
        initial_orientations = Rotation.random(1, random_state=1)
        orientations_diff, fractions_diff = _core.derivatives(
            phase=_minerals.MineralPhase.olivine,
            fabric=_minerals.OlivineFabric.A,
            n_grains=1,
            orientations=initial_orientations.as_matrix(),
            fractions=np.array([1.0]),
            strain_rate=nondim_strain_rate,
            velocity_gradient=nondim_velocity_gradient,
            stress_exponent=3.5,
            dislocation_exponent=1.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        # Check that we are moving towards a 'point' symmetry (λ[0] is largest).
        # See Vollmer 1990:
        # <https://doi.org/10.1130/0016-7606(1990)102%3C0786:aaoemt%3E2.3.co;2>.
        orientations_new = initial_orientations.as_matrix()[0] + orientations_diff[0]
        λ = np.abs(la.eigvals(orientations_new))
        assert λ[0] > λ[1] and np.isclose(λ[1], λ[2])
        assert np.isclose(np.sum(fractions_diff), 0.0)
