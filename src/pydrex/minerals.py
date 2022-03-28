"""PyDRex: Lagrangian data structures and fabric/RRSS selection.

Acronyms:
- RRSS = Reference Resolved Shear Stress,
    i.e. components of stress acting on each slip system in the grain reference frame

"""
from dataclasses import dataclass
from enum import IntEnum, unique

import numpy as np
import numba as nb
from scipy.spatial.transform import Rotation

import pydrex.core as _core


@unique
class OlivineFabric(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4


@unique
class EnstatiteFabric(IntEnum):
    A = 0  # Just to make it consistent.


@unique
class MineralPhase(IntEnum):
    """Supported mineral phases:

    - olivine (0)
    - enstatite (1)

    """

    olivine = 0
    enstatite = 1


@nb.njit
def get_rrss(phase, fabric):
    if phase == MineralPhase.olivine:
        if fabric == OlivineFabric.A:
            return np.array([1, 2, 3, np.inf])
        elif fabric == OlivineFabric.B:
            return np.array([3, 2, 1, np.inf])
        elif fabric == OlivineFabric.C:
            return np.array([3, 2, np.inf, 1])
        elif fabric == OlivineFabric.D:
            return np.array([1, 1, 3, np.inf])
        elif fabric == OlivineFabric.E:
            return np.array([3, 1, 2, np.inf])
        else:
            assert False  # Should never happen.
    elif phase == MineralPhase.enstatite:
        if fabric == EnstatiteFabric.A:
            return np.array([np.inf, np.inf, np.inf, 1])
        else:
            assert False  # Should never happen.
    else:
        assert False  # Should never happen.


@dataclass
class Mineral:
    """Class for storing mineral CPO.

    A `Mineral` stores CPO data for an aggregate of grains*.
    Additionally, mineral fabric type and deformation regime
    are also tracked. See `pydrex.fabric` and `pydrex.deformation_mechanism`.

    Attributes:
    - `phase` (int) — ordinal number of the mineral phase, see `MineralPhase`
    - `fabric` (int) — ordinal number of the fabric type, see `pydrex.fabric`
    - `regime` (int) — ordinal number of the deformation regime,
        see `pydrex.deformation_mechanism.Regime`
    - `n_grains` (int) — number of grains in the aggregate
    - `fractions` (array, optional) — dimensionless volumes of the grains
    - `orientations` (array, optional) — orientation matrices of the grains

    By default, a uniform volume distribution of random orientations is generated.

    *Note that "grains" is here an approximate term,
    and the stored objects do not fully correspond to physical grains.
    For example, the number of grains is fixed despite inclusion of
    grain nucleation in the modelling. It is assumed that new grains
    do not grow large enough to require independent rotation tracking.
    The DRex model is also unsuitable if static recrystallization is significant.

    """

    phase: int
    fabric: int
    regime: int
    n_grains: int
    fractions: np.ndarray = None
    orientations: np.ndarray = None
    # Private copy of the initial CPO values, needed for grain boundary sliding.
    _fractions_init: np.ndarray = None
    _orientations_init: np.ndarray = None

    def __post_init__(self):
        """Initialise orientations and grain volume fractions."""
        if self.fractions is None:
            self.fractions = np.ones(self.n_grains) / self.n_grains
        if self.orientations is None:
            self.orientations = Rotation.random(
                self.n_grains, random_state=1
            ).as_matrix()
        # Store initial values in case we need them for non-rotating grains.
        self._fractions_init = self.fractions.copy()
        self._orientations_init = self.orientations.copy()

    def update_orientations(
        self, strain_rate, strain_rate_max, velocity_gradient, config, dt
    ):
        """Update CPO orientations and their volume distribution using the RK4 scheme.

        Args:
        - `strain_rate` (array) — 3x3 dimensionless macroscopic strain rate tensor
        - `strain_rate_max` (float) — strain rate scale (max. eigenvalue of strain rate)
        - `velocity_gradient` (array) — 3x3 dimensionless velocity gradient tensor
        - `config` (dict) — PyDRex configuration dictionary
        - `dt` (float) — advection time step

        """
        # TODO: Investigate improvements to the DRex model.
        # If the velocity gradient does not have a rotational component,
        # the DRex model seems to fall over. Manually skip the mutations.
        if np.allclose(velocity_gradient, velocity_gradient.transpose()):
            return
        stress_exponent = config["stress_exponent"]
        dislocation_exponent = config["dislocation_exponent"]
        gbm_mobility = config["gbm_mobility"]
        gbs_threshold = config["gbs_threshold"]
        nucleation_efficiency = config["nucleation_efficiency"]
        if self.phase == MineralPhase.olivine:
            volume_fraction = config["olivine_fraction"]
        elif self.phase == MineralPhase.enstatite:
            volume_fraction = config["enstatite_fraction"]
        else:
            assert False  # Should never happen.

        # ========== RK step  1 ==========
        rotation_rates, fractions_diff = _core.derivatives(
            phase=self.phase,
            fabric=self.fabric,
            n_grains=self.n_grains,
            orientations=self.orientations,
            fractions=self.fractions,
            strain_rate=strain_rate,
            velocity_gradient=velocity_gradient,
            stress_exponent=stress_exponent,
            dislocation_exponent=dislocation_exponent,
            nucleation_efficiency=nucleation_efficiency,
            gbm_mobility=gbm_mobility,
            volume_fraction=volume_fraction,
        )
        orientations_1 = rotation_rates * dt * strain_rate_max
        orientations_iter = self.orientations + 0.5 * orientations_1
        orientations_iter.clip(-1, 1)
        fractions_1 = fractions_diff * dt * strain_rate_max
        fractions_iter = self.fractions + 0.5 * fractions_1
        fractions_iter.clip(0, None)
        fractions_iter /= fractions_iter.sum()

        # ========== RK step  2 ==========
        rotation_rates, fractions_diff = _core.derivatives(
            phase=self.phase,
            fabric=self.fabric,
            n_grains=self.n_grains,
            orientations=self.orientations,
            fractions=self.fractions,
            strain_rate=strain_rate,
            velocity_gradient=velocity_gradient,
            stress_exponent=stress_exponent,
            dislocation_exponent=dislocation_exponent,
            nucleation_efficiency=nucleation_efficiency,
            gbm_mobility=gbm_mobility,
            volume_fraction=volume_fraction,
        )
        orientations_2 = rotation_rates * dt * strain_rate_max
        orientations_iter = self.orientations + 0.5 * orientations_2
        orientations_iter.clip(-1, 1)
        fractions_2 = fractions_diff * dt * strain_rate_max
        fractions_iter = self.fractions + 0.5 * fractions_2
        fractions_iter.clip(0, None)
        fractions_iter /= fractions_iter.sum()

        # ========== RK step 3 ==========
        rotation_rates, fractions_diff = _core.derivatives(
            phase=self.phase,
            fabric=self.fabric,
            n_grains=self.n_grains,
            orientations=self.orientations,
            fractions=self.fractions,
            strain_rate=strain_rate,
            velocity_gradient=velocity_gradient,
            stress_exponent=stress_exponent,
            dislocation_exponent=dislocation_exponent,
            nucleation_efficiency=nucleation_efficiency,
            gbm_mobility=gbm_mobility,
            volume_fraction=volume_fraction,
        )
        orientations_3 = rotation_rates * dt * strain_rate_max
        orientations_iter = self.orientations + orientations_3
        orientations_iter.clip(-1, 1)
        fractions_3 = fractions_diff * dt * strain_rate_max
        fractions_iter = self.fractions + fractions_3
        fractions_iter.clip(0, None)
        fractions_iter /= fractions_iter.sum()

        # ========== RK step 4 ==========
        rotation_rates, fractions_diff = _core.derivatives(
            phase=self.phase,
            fabric=self.fabric,
            n_grains=self.n_grains,
            orientations=self.orientations,
            fractions=self.fractions,
            strain_rate=strain_rate,
            velocity_gradient=velocity_gradient,
            stress_exponent=stress_exponent,
            dislocation_exponent=dislocation_exponent,
            nucleation_efficiency=nucleation_efficiency,
            gbm_mobility=gbm_mobility,
            volume_fraction=volume_fraction,
        )
        orientations_4 = rotation_rates * dt * strain_rate_max
        self.orientations = (
            self.orientations
            + (
                orientations_1 / 2
                + orientations_2
                + orientations_3
                + orientations_4 / 2
            )
            / 3
        )
        self.orientations.clip(-1, 1)
        fractions_4 = fractions_diff * dt * strain_rate_max
        self.fractions += (
            fractions_1 / 2 + fractions_2 + fractions_3 + fractions_4 / 2
        ) / 3
        self.fractions /= self.fractions.sum()

        # Grain boundary sliding for small grains.
        mask = self.fractions < gbs_threshold / self.n_grains
        self.orientations[mask, :, :] = self._orientations_init[mask, :, :]
        self.fractions[mask] = gbs_threshold / self.n_grains
        self.fractions /= self.fractions.sum()

    def save(self, filename):
        """Save CPO data to a `numpy` NPZ file.

        Raises a `ValueError` if the data shapes are not compatible.

        See also: `numpy.savez`.

        """
        if self.fractions.shape[0] == self.orientations.shape[0] == self.n_grains:
            meta = np.array([self.fabric, self.regime], dtype=np.uint8)
            np.savez(
                filename,
                meta=meta,
                fractions=self.fractions,
                orientations=self.orientations,
            )
        else:
            raise ValueError(
                "Size of CPO data arrays must match number of grains."
                + " You've supplied corrupted data with:"
                + f" `n_grains = {self.n_grains}`,"
                + f" `fractions.shape = {self.fractions.shape}`,"
                + f" `orientations.shape = {self.orientations.shape}`."
            )

    def load(self, filename):
        """Load CPO data from a `numpy` NPZ file.

        See also: `Mineral.save`.

        """
        if not filename.endswith(".npz"):
            raise ValueError(
                f"Must only load from numpy NPZ format. Cannot load from {filename}."
            )
        data = np.load(filename)
        fabric, regime = data["meta"]
        self.fabric = fabric
        self.regime = regime
        self.fractions = data["fractions"]
        self.orientations = data["orientations"]
