"""> PyDRex: Lagrangian data structures and mineral fabric definitions.

**Acronyms:**
- CRSS = Critical Resolved Shear Stress,
    i.e. threshold stress required to initiate slip on a slip system,
    normalised to the stress required to initiate slip on the softes slip system

"""
import io
import pathlib as pl
from dataclasses import dataclass, field
from enum import IntEnum, unique
from zipfile import ZipFile

import numba as nb
import numpy as np
from scipy import linalg as la
from scipy.integrate import RK45
from scipy.spatial.transform import Rotation

from pydrex import core as _core
from pydrex import deformation_mechanism as _defmech
from pydrex import exceptions as _err
from pydrex import logger as _log


@unique
class OlivineFabric(IntEnum):
    """Enumeration type for olivine fabrics A-E."""

    A = 0
    B = 1
    C = 2
    D = 3
    E = 4


@unique
class EnstatiteFabric(IntEnum):
    """Enumeration type with a single member A which is the only enstatite fabric."""

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
def get_crss(phase, fabric):
    """Get Critical Resolved Shear Stress for the mineral `phase` and `fabric`.

    Returns an array of the normalised threshold stresses required to activate slip
    on each slip system.

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
    are also tracked. See `pydrex.deformation_mechanism`.

    Attributes:
    - `phase` (int) — ordinal number of the mineral phase, see `MineralPhase`
    - `fabric` (int) — ordinal number of the fabric type, see `OlivineFabric`
      and `EnstatiteFabric`
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

    phase: int = MineralPhase.olivine
    fabric: int = OlivineFabric.A
    regime: int = _defmech.Regime.dislocation
    n_grains: int = 1000
    # Initial condition, randomised if not given.
    fractions_init: np.ndarray = None
    orientations_init: np.ndarray = None
    fractions: list = field(default_factory=list)
    orientations: list = field(default_factory=list)

    def __str__(self):
        # String output, used for str(self) and f"{self}", etc.
        if hasattr(self.fractions[0], "shape"):
            shape_of_fractions = str(self.fractions[0].shape)
        else:
            shape_of_fractions = "(?)"

        if hasattr(self.orientations[0], "shape"):
            shape_of_orientations = str(self.orientations[0].shape)
        else:
            shape_of_orientations = "(?)"

        return (
            self.__class__.__qualname__
            + f"(phase={self.phase!s}, "
            + f"fabric={self.fabric!s}, "
            + f"regime={self.regime!s}, "
            + f"n_grains={self.n_grains!s}, "
            + f"fractions=<{self.fractions.__class__.__qualname__}"
            + f" of {self.fractions[0].__class__.__qualname__} {shape_of_fractions}>, "
            + f"orientations=<{self.orientations.__class__.__qualname__}"
            + f" of {self.orientations[0].__class__.__qualname__} {shape_of_orientations}>)"
        )

    def _repr_pretty_(self, p, cycle):
        # Format to use when printing to IPython or other interactive console.
        p.text(self.__str__() if not cycle else self.__class__.__qualname__ + "(...)")

    def __post_init__(self):
        """Initialise random orientations and grain volume fractions."""
        if self.fractions_init is None:
            self.fractions_init = np.full(self.n_grains, 1.0 / self.n_grains)
        if self.orientations_init is None:
            self.orientations_init = Rotation.random(
                self.n_grains, random_state=1
            ).as_matrix()

        # Copy the initial values to the storage lists.
        self.fractions.append(self.fractions_init)
        self.orientations.append(self.orientations_init)

        # Delete the initial value duplicates to avoid confusion.
        del self.fractions_init
        del self.orientations_init

        _log.info("created %s", self)

    def update_orientations(
        self,
        config,
        deformation_gradient,
        velocity_gradient,
        integration_time,
        pathline=None,
    ):
        """Update orientations and volume distribution for the `Mineral`.

        Update crystalline orientations and grain volume distribution
        for minerals undergoing plastic deformation.

        Args:
        - `config` (dict) — PyDRex configuration dictionary
        - `deformation_gradient` (array) — 3x3 initial deformation gradient tensor
        - `velocity_gradient` (array or function) — 3x3 velocity gradient matrix,
          or an interpolator that returns the 3x3 matrix when evaluated at a point
          along the provided pathline
        - `integration_time` (float) — total time of integrated dislocation
          creep (if `pathline` is not None, this is used as the maximum
          integration timestep during CPO calculation)
        - `pathline` (tuple, optional) — tuple consisting of:
            1. the time at which to start the CPO integration
            2. the time at which to stop the CPO integration
            3. a callable that accepts a time value and returns the position of
               the mineral

        Array values must provide a NumPy-compatible interface:
        <https://numpy.org/doc/stable/user/whatisnumpy.html>

        """

        # Set up callables for the ODE, some variables come from enclosing scope.
        def extract_vars(y):
            # TODO: Check if we can avoid .copy() here.
            deformation_gradient = y[:9].copy().reshape(3, 3)
            orientations = (
                y[9 : self.n_grains * 9 + 9]
                .copy()
                .reshape(self.n_grains, 3, 3)
                .clip(-1, 1)
            )
            fractions = (
                y[self.n_grains * 9 + 9 : self.n_grains * 10 + 9].copy().clip(0, None)
            )
            # FIXME: Guard against zero-division? Why is .sum ≈ 0?
            fractions /= fractions.sum()
            return deformation_gradient, orientations, fractions

        def eval_rhs(t, y):
            """Evaluate right hand side of the D-Rex PDE."""
            deformation_gradient, orientations, fractions = extract_vars(y)
            # Uses nondimensional values of strain rate and velocity gradient.
            orientations_diff, fractions_diff = _core.derivatives(
                phase=self.phase,
                fabric=self.fabric,
                n_grains=self.n_grains,
                orientations=orientations,
                fractions=fractions,
                strain_rate=strain_rate / strain_rate_max,
                velocity_gradient=_velocity_gradient / strain_rate_max,
                stress_exponent=config["stress_exponent"],
                deformation_exponent=config["deformation_exponent"],
                nucleation_efficiency=config["nucleation_efficiency"],
                gbm_mobility=config["gbm_mobility"],
                volume_fraction=volume_fraction,
            )
            return np.hstack(
                (
                    (_velocity_gradient @ deformation_gradient).flatten(),
                    orientations_diff.flatten() * strain_rate_max,
                    fractions_diff * strain_rate_max,
                )
            )

        def apply_gbs(orientations, fractions, config):
            # Grain boundary sliding for small grains.
            mask = fractions < config["gbs_threshold"] / self.n_grains
            _log.debug(
                "grain boundary sliding activity (volume percentage): %s",
                len(np.nonzero(mask)) / len(fractions),
            )
            # No rotation: carry over previous orientations.
            orientations[mask, :, :] = self.orientations[0][mask, :, :]
            fractions[mask] = config["gbs_threshold"] / self.n_grains
            # FIXME: Guard against zero-division? Why is .sum ≈ 0?
            fractions /= fractions.sum()
            _log.debug(
                "grain volume fractions: median=%e, min=%e, max=%e, sum=%e",
                np.median(fractions),
                np.min(fractions),
                np.max(fractions),
                np.sum(fractions),
            )
            return orientations, fractions

        # Set up pathline or time integral bounds and initial condition.
        if callable(velocity_gradient):
            if pathline is None:
                raise ValueError(
                    "unable to evaluate velocity gradient callable."
                    + " You must provide the `pathline` to use a velocity gradient callable."
                )
            time_start, time_end, get_position = pathline
            _velocity_gradient = velocity_gradient(
                np.atleast_2d(get_position(time_start))
            )[0]
            _log.info(
                "calculating CPO at %s (t=%e) with velocity gradient %s",
                get_position(time_start),
                time_start,
                _velocity_gradient.flatten(),
            )
            max_step = integration_time
        else:
            _velocity_gradient = velocity_gradient
            time_start = 0
            time_end = integration_time
            _log.info(
                "calculating CPO for Δt=%e with velocity gradient %s",
                time_end - time_start,
                _velocity_gradient.flatten(),
            )
            max_step = time_end - time_start

        strain_rate = (_velocity_gradient + _velocity_gradient.transpose()) / 2
        strain_rate_max = np.abs(la.eigvalsh(strain_rate)).max()
        # max_step = min(max_step, 1e-2 / strain_rate_max)
        # max_step = 1e6

        if self.phase == MineralPhase.olivine:
            volume_fraction = config["olivine_fraction"]
        elif self.phase == MineralPhase.enstatite:
            volume_fraction = config["enstatite_fraction"]
        else:
            assert False  # Should never happen.

        # Initialise solver and perform first step.
        solver = RK45(
            eval_rhs,
            time_start,
            np.hstack(
                (
                    deformation_gradient.flatten(),
                    self.orientations[-1].flatten(),
                    self.fractions[-1],
                )
            ),
            time_end,
            # first_step=max_step / 4,  # TODO: Move divisor to config?
            max_step=max_step,
            # atol=np.inf,  # FIXME: Different solver or smart way to set tolerance.
        )
        message = solver.step()
        if message is not None and solver.status == "failed":
            raise _err.IterationError(message)
        _log.debug(
            "%s step_size=%e (max_step=%e)",
            solver.__class__.__qualname__,
            solver.step_size,
            solver.max_step,
        )

        deformation_gradient, orientations, fractions = extract_vars(solver.y)
        orientations, fractions = apply_gbs(orientations, fractions, config)
        solver.y[9:] = np.hstack((orientations.flatten(), fractions))

        # Solve ODE using numerical iteration scheme.
        while solver.status == "running":
            if callable(velocity_gradient):
                _velocity_gradient = velocity_gradient(
                    np.atleast_2d(get_position(solver.t))
                )[0]
                _log.info(
                    "calculating CPO at %s (t=%e) with velocity gradient %s",
                    get_position(solver.t),
                    solver.t,
                    _velocity_gradient.flatten(),
                )
            else:
                _velocity_gradient = velocity_gradient

            strain_rate = (_velocity_gradient + _velocity_gradient.transpose()) / 2
            strain_rate_max = np.abs(la.eigvalsh(strain_rate)).max()
            # solver.max_step = min(solver.max_step, 1e-2 / strain_rate_max)

            message = solver.step()
            if message is not None and solver.status == "failed":
                raise _err.IterationError(message)
            _log.debug(
                "%s step_size=%e (max_step=%e)",
                solver.__class__.__qualname__,
                solver.step_size,
                solver.max_step,
            )

            deformation_gradient, orientations, fractions = extract_vars(solver.y)
            orientations, fractions = apply_gbs(orientations, fractions, config)
            solver.y[9:] = np.hstack((orientations.flatten(), fractions))

        # Extract final values for this simulation step, append to storage.
        deformation_gradient, orientations, fractions = extract_vars(solver.y.squeeze())
        self.orientations.append(orientations)
        self.fractions.append(fractions)
        return deformation_gradient

    def save(self, filename, postfix=None):
        """Save CPO data for all stored timesteps to a `numpy` NPZ file.

        If `postfix` is not `None`, the data is appended to the NPZ file
        in fields ending with "`_postfix`".

        Raises a `ValueError` if the data shapes are not compatible.

        See also: `numpy.savez`, `Mineral.load`, `Mineral.from_file`.

        """
        if len(self.fractions) != len(self.orientations):
            raise ValueError(
                "Length of stored results must match."
                + " You've supplied currupted data with:\n"
                + f"- {len(self.fractions)} grain size results, and\n"
                + f"- {len(self.orientations)} orientation results."
            )
        if self.fractions[0].shape[0] == self.orientations[0].shape[0] == self.n_grains:
            data = {
                "meta": np.array(
                    [self.phase, self.fabric, self.regime], dtype=np.uint8
                ),
                "fractions": np.stack(self.fractions),
                "orientations": np.stack(self.orientations),
            }
            path = pl.Path(filename)
            # Create parent directories if needed.
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append to file, requires postfix (unique name).
            if postfix is not None:
                archive = ZipFile(filename, mode="a", allowZip64=True)
                for key in data.keys():
                    with archive.open(
                        f"{key}_{postfix}", "w", force_zip64=True
                    ) as file:
                        buffer = io.BytesIO()
                        np.save(buffer, data[key])
                        file.write(buffer.getvalue())
                        buffer.close()
            else:
                np.savez(filename, **data)
        else:
            raise ValueError(
                "Size of CPO data arrays must match number of grains."
                + " You've supplied corrupted data with:\n"
                + f"- `n_grains = {self.n_grains}`,\n"
                + f"- `fractions[0].shape = {self.fractions[0].shape}`, and\n"
                + f"- `orientations[0].shape = {self.orientations[0].shape}`."
            )

    def load(self, filename, postfix=None):
        """Load CPO data from a `numpy` NPZ file.

        If `postfix` is not `None`, data is read from fields ending with "`_postfix`".

        See also: `Mineral.save`, `Mineral.from_file`.

        """
        if not filename.endswith(".npz"):
            raise ValueError(
                f"Must only load from numpy NPZ format. Cannot load from {filename}."
            )
        data = np.load(filename)
        if postfix is not None:
            phase, fabric, regime = data[f"meta_{postfix}"]
            self.fractions = list(data[f"fractions_{postfix}"])
            self.orientations = list(data[f"orientations_{postfix}"])
        else:
            phase, fabric, regime = data["meta"]
            self.fractions = list(data["fractions"])
            self.orientations = list(data["orientations"])

        self.phase = phase
        self.fabric = fabric
        self.regime = regime
        self.orientations_init = self.orientations[0]
        self.fractions_init = self.fractions[0]

    @classmethod
    def from_file(cls, filename, postfix=None):
        """Construct a `Mineral` instance using data from a `numpy` NPZ file.

        If `postfix` is not `None`, data is read from fields ending with “`_postfix`”.

        See also: `Mineral.save`, `Mineral.load`.

        """
        if not filename.endswith(".npz"):
            raise ValueError(
                f"Must only load from numpy NPZ format. Cannot load from {filename}."
            )
        data = np.load(filename)
        if postfix is not None:
            phase, fabric, regime = data[f"meta_{postfix}"]
            fractions = list(data[f"fractions_{postfix}"])
            orientations = list(data[f"orientations_{postfix}"])
        else:
            phase, fabric, regime = data["meta"]
            fractions = list(data["fractions"])
            orientations = list(data["orientations"])

        mineral = cls(
            phase,
            fabric,
            regime,
            n_grains=len(fractions[0]),
            fractions_init=fractions[0],
            orientations_init=orientations[0],
        )
        mineral.fractions = fractions
        mineral.orientations = orientations
        return mineral
