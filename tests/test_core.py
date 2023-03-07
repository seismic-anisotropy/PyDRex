"""> PyDRex: tests for core D-Rex routines."""
import numpy as np
from scipy.spatial.transform import Rotation

from pydrex import core as _core
from pydrex import minerals as _minerals
from pydrex import logger as _log


class TestSimpleShearSingleAType:
    """Single-grain A-type olivine analytical re-orientation rate tests."""

    def test_initP10Z(self):
        r"""Single grain of olivine A-type, simple shear with:

        $$
        \bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\dot{\varepsilon}} = \begin{bmatrix} 0 & 1 & 0 \cr 1 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\Omega} = \begin{bmatrix} 0 & -1 & 0 \cr 1 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}
        $$

        Initial rotation 10° anti-clockwise around Z (looking down the axis).

        """
        nondim_velocity_gradient = np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        # Grain initialised with rotation around Z (anti-clockwise).
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
            stress_exponent=1.5,
            deformation_exponent=3.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2 * θ)
        sinθ = np.sin(θ)
        _log.debug("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array(
            [
                [sinθ * (1 + cos2θ), cosθ * (1 + cos2θ), 0],
                [cosθ * (-1 - cos2θ), sinθ * (1 + cos2θ), 0],
                [0, 0, 0],
            ]
        )
        _log.debug("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        _log.debug("calculated grain volume change:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_initN10Z(self):
        r"""Single grain of olivine A-type, simple shear with:

        $$
        \bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\dot{\varepsilon}} = \begin{bmatrix} 0 & 1 & 0 \cr 1 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\Omega} = \begin{bmatrix} 0 & -1 & 0 \cr 1 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}
        $$

        Initial rotation 10° clockwise around Z (looking down the axis).

        """
        nondim_velocity_gradient = np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        # Grain initialised with rotation around Z (clockwise).
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
            stress_exponent=1.5,
            deformation_exponent=3.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2 * θ)
        sinθ = np.sin(θ)
        _log.debug("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array(
            [
                [sinθ * (1 + cos2θ), cosθ * (1 + cos2θ), 0],
                [cosθ * (-1 - cos2θ), sinθ * (1 + cos2θ), 0],
                [0, 0, 0],
            ]
        )
        _log.debug("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        _log.debug("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_initP45Z(self):
        r"""Single grain of olivine A-type, simple shear with:

        $$
        \bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\dot{\varepsilon}} = \begin{bmatrix} 0 & 1 & 0 \cr 1 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\Omega} = \begin{bmatrix} 0 & -1 & 0 \cr 1 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}
        $$

        Initial rotation 45° anti-clockwise around Z (looking down the axis).

        """
        nondim_velocity_gradient = np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        # Grain initialised with rotation around Z (anti-clockwise).
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
            stress_exponent=1.5,
            deformation_exponent=3.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2 * θ)
        sinθ = np.sin(θ)
        _log.debug("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array(
            [
                [sinθ * (1 + cos2θ), cosθ * (1 + cos2θ), 0],
                [cosθ * (-1 - cos2θ), sinθ * (1 + cos2θ), 0],
                [0, 0, 0],
            ]
        )
        _log.debug("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        _log.debug("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_initP90Z(self):
        r"""Single grain of olivine A-type, simple shear with:

        $$
        \bm{L} = \begin{bmatrix} 0 & 0 & 0 \cr 2 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\dot{\varepsilon}} = \begin{bmatrix} 0 & 1 & 0 \cr 1 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\Omega} = \begin{bmatrix} 0 & -1 & 0 \cr 1 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix}
        $$

        Initial rotation 90° anti-clockwise around Z (looking down the axis).

        """
        nondim_velocity_gradient = np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        # Grain initialised with rotation around Z (anti-clockwise).
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
            stress_exponent=1.5,
            deformation_exponent=3.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2 * θ)
        sinθ = np.sin(θ)
        _log.debug("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array(
            [
                [sinθ * (1 + cos2θ), cosθ * (1 + cos2θ), 0],
                [cosθ * (-1 - cos2θ), sinθ * (1 + cos2θ), 0],
                [0, 0, 0],
            ]
        )
        _log.debug("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        _log.debug("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_initP10Y(self):
        r"""Single grain of olivine A-type, simple shear with:

        $$
        \bm{L} = \begin{bmatrix} 0 & 0 & 2 \cr 0 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\dot{\varepsilon}} = \begin{bmatrix} 0 & 0 & 1 \cr 0 & 0 & 0 \cr 1 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\Omega} = \begin{bmatrix} 0 & 0 & 1 \cr 0 & 0 & 0 \cr -1 & 0 & 0 \end{bmatrix}
        $$

        Initial rotation 10° anti-clockwise around Y (looking down the axis).

        """
        nondim_velocity_gradient = np.array([[0, 0, 2.0], [0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])
        # Grain initialised with rotation around Y (anti-clockwise).
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
            stress_exponent=1.5,
            deformation_exponent=3.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2 * θ)
        sinθ = np.sin(θ)
        _log.debug("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array(
            [
                [sinθ * (1 - cos2θ), 0, cosθ * (cos2θ - 1)],
                [0, 0, 0],
                [cosθ * (1 - cos2θ), 0, sinθ * (1 - cos2θ)],
            ]
        )
        _log.debug("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        _log.debug("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_initN10Y(self):
        r"""Single grain of olivine A-type, simple shear with:

        $$
        \bm{L} = \begin{bmatrix} 0 & 0 & 2 \cr 0 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\dot{\varepsilon}} = \begin{bmatrix} 0 & 0 & 1 \cr 0 & 0 & 0 \cr 1 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\Omega} = \begin{bmatrix} 0 & 0 & 1 \cr 0 & 0 & 0 \cr -1 & 0 & 0 \end{bmatrix}
        $$

        Initial rotation 10° clockwise around Y (looking down the axis).

        """
        nondim_velocity_gradient = np.array([[0, 0, 2.0], [0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])
        # Grain initialised with rotation around Y (clockwise).
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
            stress_exponent=1.5,
            deformation_exponent=3.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2 * θ)
        sinθ = np.sin(θ)
        _log.debug("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array(
            [
                [sinθ * (1 - cos2θ), 0, cosθ * (cos2θ - 1)],
                [0, 0, 0],
                [cosθ * (1 - cos2θ), 0, sinθ * (1 - cos2θ)],
            ]
        )
        _log.debug("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        _log.debug("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_initP45Y(self):
        r"""Single grain of olivine A-type, simple shear with:

        $$
        \bm{L} = \begin{bmatrix} 0 & 0 & 2 \cr 0 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\dot{\varepsilon}} = \begin{bmatrix} 0 & 0 & 1 \cr 0 & 0 & 0 \cr 1 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\Omega} = \begin{bmatrix} 0 & 0 & 1 \cr 0 & 0 & 0 \cr -1 & 0 & 0 \end{bmatrix}
        $$

        Initial rotation 45° anti-clockwise around Y (looking down the axis).

        """
        nondim_velocity_gradient = np.array([[0, 0, 2.0], [0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])
        # Grain initialised with rotation around Y (anti-clockwise).
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
            stress_exponent=1.5,
            deformation_exponent=3.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2 * θ)
        sinθ = np.sin(θ)
        _log.debug("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array(
            [
                [sinθ * (1 - cos2θ), 0, cosθ * (cos2θ - 1)],
                [0, 0, 0],
                [cosθ * (1 - cos2θ), 0, sinθ * (1 - cos2θ)],
            ]
        )
        _log.debug("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        _log.debug("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)

    def test_initP90Y(self):
        r"""Single grain of olivine A-type, simple shear with:

        $$
        \bm{L} = \begin{bmatrix} 0 & 0 & 2 \cr 0 & 0 & 0 \cr 0 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\dot{\varepsilon}} = \begin{bmatrix} 0 & 0 & 1 \cr 0 & 0 & 0 \cr 1 & 0 & 0 \end{bmatrix},
        \quad
        \bm{\Omega} = \begin{bmatrix} 0 & 0 & 1 \cr 0 & 0 & 0 \cr -1 & 0 & 0 \end{bmatrix}
        $$

        Initial rotation 90° anti-clockwise around Y (looking down the axis).

        """
        nondim_velocity_gradient = np.array([[0, 0, 2.0], [0, 0, 0], [0, 0, 0]])
        nondim_strain_rate = np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]])
        # Grain initialised with rotation around Y (anti-clockwise).
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
            stress_exponent=1.5,
            deformation_exponent=3.5,
            nucleation_efficiency=5,
            gbm_mobility=125,
            volume_fraction=1.0,
        )
        cosθ = np.cos(θ)
        cos2θ = np.cos(2 * θ)
        sinθ = np.sin(θ)
        _log.debug("calculated re-orientation rate:\n", orientations_diff)
        target_orientations_diff = np.array(
            [
                [sinθ * (1 - cos2θ), 0, cosθ * (cos2θ - 1)],
                [0, 0, 0],
                [cosθ * (1 - cos2θ), 0, sinθ * (1 - cos2θ)],
            ]
        )
        _log.debug("target re-orientation rate:\n", target_orientations_diff)
        assert np.allclose(orientations_diff, target_orientations_diff)
        _log.debug("calculated grain volume changes:\n", fractions_diff)
        assert np.isclose(np.sum(fractions_diff), 0.0)
