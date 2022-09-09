"""PyDRex: Lagrangian data structures and fabric/RRSS selection.

Acronyms:
- RRSS = Reference Resolved Shear Stress,
    i.e. components of stress acting on each slip system in the grain reference frame

"""
from dataclasses import dataclass, field
from enum import IntEnum, unique

import numba as nb
import numpy as np
from scipy import linalg as la
from scipy.integrate import RK45
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
    """Get Reference Resolved Shear Stress for the mineral `phase` and `fabric`.

    Returns an array of the components of stress acting on each slip system
    in the grain-local reference frame.

    """
    if phase == MineralPhase.olivine:
        match fabric:
            case OlivineFabric.A:
                return np.array([1, 2, 3, np.inf])
            case OlivineFabric.B:
                return np.array([3, 2, 1, np.inf])
            case OlivineFabric.C:
                return np.array([3, 2, np.inf, 1])
            case OlivineFabric.D:
                return np.array([1, 1, 3, np.inf])
            case OlivineFabric.E:
                return np.array([3, 1, 2, np.inf])
            case _:
                raise ValueError("fabric must be a valid `OlivineFabric`")
    elif phase == MineralPhase.enstatite:
        if fabric == EnstatiteFabric.A:
            return np.array([np.inf, np.inf, np.inf, 1])
        raise ValueError("fabric must be a valid `EnstatiteFabric`")
    raise ValueError("phase must be a valid `MineralPhase`")


def get_primary_axis(fabric):
    """Get primary slip axis name for the given olivine `fabric`."""
    match fabric:
        case OlivineFabric.A:
            return "a"
        case OlivineFabric.B:
            return "c"
        case OlivineFabric.C:
            return "c"
        case OlivineFabric.D:
            return "a"
        case OlivineFabric.E:
            return "a"
    raise ValueError(f"fabric must be a valid `OlivineFabric`, not {fabric}")


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
    - `fractions_init` (array, optional) — initial dimensionless grain volumes
    - `orientations_init` (array, optional) — initial grain orientation matrices

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
    # Initial condition, randomised if not given.
    fractions_init: np.ndarray = None
    orientations_init: np.ndarray = None
    # Private members to store values for all successive simulation steps.
    # Lists of numpy arrays will be `np.stack`ed when writing the NPZ file.
    _fractions: list = field(default_factory=list)
    _orientations: list = field(default_factory=list)

    def __post_init__(self):
        """Initialise random orientations and grain volume fractions."""
        if self.fractions_init is None:
            self.fractions_init = np.ones(self.n_grains) / self.n_grains
        if self.orientations_init is None:
            self.orientations_init = Rotation.random(
                self.n_grains, random_state=1
            ).as_matrix()

        # Copy the initial values to the storage lists.
        self._fractions.append(self.fractions_init)
        self._orientations.append(self.orientations_init)

    def update_orientations(
        self, config, finite_strain_ell, velocity_gradient, integration_time
    ):
        """Update orientations and volume distribution for the `Mineral`.

        Update crystalline orientations and grain volume distribution
        for minerals undergoing plastic deformation.
        Only 3D simulations are currently supported.

        Args:
        - `config` (dict) — PyDRex configuration dictionary
        - `finite_strain_ell` (array) — 3x3 finite strain ellipsoid (initial strain)
        - `velocity_gradient` (array) — 3x3 velocity gradient matrix
        - `integration_time` (float) — total time of integrated dislocation creep

        Array values must provide a NumPy-compatible interface:
        <https://numpy.org/doc/stable/user/whatisnumpy.html>

        """

        def extract_vars(y):
            finite_strain_ell = y[:9].copy().reshape(3, 3)
            orientations = (
                y[9 : self.n_grains * 9 + 9]
                .copy()
                .reshape(self.n_grains, 3, 3)
                .clip(-1, 1)
            )
            fractions = (
                y[self.n_grains * 9 + 9 : self.n_grains * 10 + 9].copy().clip(0, None)
            )
            fractions /= fractions.sum()
            return finite_strain_ell, orientations, fractions

        def rhs(t, y):
            finite_strain_ell, orientations, fractions = extract_vars(y)
            orientations_diff, fractions_diff = _core.derivatives(
                phase=self.phase,
                fabric=self.fabric,
                n_grains=self.n_grains,
                orientations=orientations,
                fractions=fractions,
                strain_rate=nondim_strain_rate,
                velocity_gradient=nondim_velocity_gradient,
                stress_exponent=config["stress_exponent"],
                dislocation_exponent=config["dislocation_exponent"],
                nucleation_efficiency=config["nucleation_efficiency"],
                gbm_mobility=config["gbm_mobility"],
                volume_fraction=volume_fraction,
            )
            return np.hstack(
                (
                    np.dot(velocity_gradient, finite_strain_ell).flatten(),
                    orientations_diff.flatten() * strain_rate_max,
                    fractions_diff * strain_rate_max,
                )
            )

        # Imposed macroscopic strain rate tensor.
        strain_rate = (velocity_gradient + velocity_gradient.transpose()) / 2
        # Strain rate scale (max. eigenvalue of strain rate).
        strain_rate_max = np.abs(la.eigvalsh(strain_rate)).max()
        # Dimensionless strain rate and velocity gradient.
        nondim_strain_rate = strain_rate / strain_rate_max
        nondim_velocity_gradient = velocity_gradient / strain_rate_max
        # Volume fraction of the mineral phase.
        if self.phase == MineralPhase.olivine:
            volume_fraction = config["olivine_fraction"]
        elif self.phase == MineralPhase.enstatite:
            volume_fraction = config["enstatite_fraction"]
        else:
            assert False  # Should never happen.

        sol = RK45(
            rhs,
            0,
            np.hstack(
                (
                    finite_strain_ell.flatten(),
                    self._orientations[-1].flatten(),
                    self._fractions[-1],
                )
            ),
            integration_time,
        )

        while sol.status == "running":
            sol.step()
            finite_strain_ell, orientations, fractions = extract_vars(sol.y)
            # Grain boundary sliding for small grains.
            mask = fractions < config["gbs_threshold"] / self.n_grains
            # No rotation: carry over previous orientations.
            orientations[mask, :, :] = self._orientations[0][mask, :, :]
            fractions[mask] = config["gbs_threshold"] / self.n_grains
            fractions /= fractions.sum()
            sol.y[9:] = np.hstack((orientations.flatten(), fractions))

        # Extract final values for this simulation step, append to storage.
        finite_strain_ell, orientations, fractions = extract_vars(sol.y.squeeze())
        self._orientations.append(orientations)
        self._fractions.append(fractions)
        return finite_strain_ell

    def save(self, filename):
        """Save CPO data for all stored timesteps to a `numpy` NPZ file.

        Raises a `ValueError` if the data shapes are not compatible.

        See also: `numpy.savez`.

        """
        if len(self._fractions) != len(self._orientations):
            raise ValueError(
                "Length of stored results must match."
                + " You've supplied currupted data with:\n"
                + f"- {len(self._fractions)} grain size results, and\n"
                + f"- {len(self._orientations)} orientation results."
            )
        if (
            self._fractions[0].shape[0]
            == self._orientations[0].shape[0]
            == self.n_grains
        ):
            meta = np.array([self.phase, self.fabric, self.regime], dtype=np.uint8)
            np.savez(
                filename,
                meta=meta,
                fractions=np.stack(self._fractions),
                orientations=np.stack(self._orientations),
            )
        else:
            raise ValueError(
                "Size of CPO data arrays must match number of grains."
                + " You've supplied corrupted data with:\n"
                + f"- `n_grains = {self.n_grains}`,\n"
                + f"- `_fractions[0].shape = {self._fractions[0].shape}`, and\n"
                + f"- `_orientations[0].shape = {self._orientations[0].shape}`."
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
        phase, fabric, regime = data["meta"]
        self.phase = phase
        self.fabric = fabric
        self.regime = regime
        self._fractions = list(data["fractions"])
        self._orientations = list(data["orientations"])
        self.orientations_init = self._orientations[0]
        self.fractions_init = self._fractions[0]
