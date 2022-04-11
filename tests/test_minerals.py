import pytest
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import linalg as la

import pydrex.minerals as _minerals
import pydrex.deformation_mechanism as _defmech


class TestSimpleShearOlivineA:
    """Tests for A-type Olivine in simple shear."""

    def test_random_single(self, params_Kaminski2001_fig5_shortdash):
        # Single grain of olivine A-type with random initial orientation.
        #     0 0 0  .   0 1 0      0 -1 0
        # L = 2 0 0  ε = 1 0 0  Ω = 1  0 0
        #     0 0 0      0 0 0      0  0 0
        n_grains = 1
        max_strain_rate = 1e-5
        mineral = _minerals.Mineral(
            _minerals.MineralPhase.olivine,
            _minerals.OlivineFabric.A,
            _defmech.Regime.dislocation,
            n_grains=n_grains,
        )
        velocity_gradient = np.array([[0, 0, 0], [2.0 * max_strain_rate, 0, 0], [0, 0, 0]])
        deformation_gradient = np.eye(3)  # Start with undeformed mineral.
        for dt in np.linspace(0, 2.5 / max_strain_rate, 11):
            deformation_gradient = mineral.update_orientations(
                params_Kaminski2001_fig5_shortdash,
                deformation_gradient,
                velocity_gradient,
                dt,
            )
        # Check that we are moving towards a 'point' symmetry
        # (one eigenvalue is largest). See Vollmer 1990:
        # <https://doi.org/10.1130/0016-7606(1990)102%3C0786:aaoemt%3E2.3.co;2>.
        orientations_final = mineral._orientations[-1][0]
        fractions_final = mineral._fractions[-1]
        print(orientations_final)
        λ = np.sort(np.abs(la.eigvals(orientations_final)))
        assert λ[2] > λ[1] and np.isclose(λ[1], λ[0])
        print(fractions_final)
        assert np.isclose(np.sum(fractions_final), 1.0)


    def test_random_1000(self, params_Kaminski2001_fig5_longdash):
        # 1000 grains of olivine A-type with random initial orientations.
        #     0 0 2  .   0 0 1       0 0 1
        # L = 0 0 0  ε = 0 0 0  Ω =  0 0 0
        #     0 0 0      1 0 0      -1 0 0
        n_grains = 1000
        max_strain_rate = 1e-5
        mineral = _minerals.Mineral(
            _minerals.MineralPhase.olivine,
            _minerals.OlivineFabric.A,
            _defmech.Regime.dislocation,
            n_grains=n_grains,
        )
        velocity_gradient = np.array([[0, 0, 2.0 * max_strain_rate], [0, 0, 0], [0, 0, 0]])
        deformation_gradient  = np.eye(3)  # Start with undeformed mineral.
        for dt in np.linspace(0, 1.0 / max_strain_rate, 6):
            deformation_gradient = mineral.update_orientations(
                params_Kaminski2001_fig5_longdash,
                deformation_gradient,
                velocity_gradient,
                dt,
            )
        # Check that we are moving towards a 'point' symmetry
        # (one eigenvalue is largest). See Vollmer 1990:
        # <https://doi.org/10.1130/0016-7606(1990)102%3C0786:aaoemt%3E2.3.co;2>.
        fractions_final = mineral._fractions[-1]
        orientations_final = (
            Rotation.from_matrix(mineral._orientations[-1]).mean(fractions_final).as_matrix()
        )
        λ = np.sort(np.abs(la.eigvals(orientations_final)))
        print(mineral.orientations_init)
        print(orientations_final)
        assert λ[2] > λ[1] and np.isclose(λ[1], λ[0])
        print(fractions_final)
        assert np.isclose(np.sum(fractions_final), 1.0)
