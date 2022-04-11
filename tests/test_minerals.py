import pytest
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import linalg as la

import pydrex.minerals as _minerals
import pydrex.deformation_mechanism as _defmech


class TestSimpleShearOlivineA:
    """Tests for A-type Olivine in simple shear."""

    def test_random_singleY(self, params_Kaminski2001_fig5_shortdash):
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
        for time in np.linspace(0, 1.0 / max_strain_rate, 11):
            # TODO: Choose n_iter based on time step and some resolution parameter?
            #       Need n_iter to increase for larger timesteps, but not too much...
            deformation_gradient = mineral._update_orientations_drex2004(
                params_Kaminski2001_fig5_shortdash,
                deformation_gradient,
                velocity_gradient,
                n_iter=10,  # Passes for `n_iter=6` and `25 > n_iter > 10`
            )
        # Check that we are moving towards a 'point' symmetry
        # (one eigenvalue is largest). See Vollmer 1990:
        # <https://doi.org/10.1130/0016-7606(1990)102%3C0786:aaoemt%3E2.3.co;2>.
        orientations_final = mineral._orientations[-1][0]
        fractions_final = mineral._fractions[-1]
        msg = f"final grain sizes:\n{fractions_final}"
        assert np.isclose(np.sum(fractions_final), 1.0), msg
        λ = np.sort(np.abs(la.eigvals(orientations_final)))
        msg = f"eigenvalues of final orientation matrix:\n{λ}"
        assert not np.isclose(λ[2], λ[1], atol=0.1), msg
        assert λ[2] > λ[1] and np.isclose(λ[1], λ[0]), msg

