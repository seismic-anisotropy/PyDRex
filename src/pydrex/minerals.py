"""> PyDRex: Lagrangian data structures and mineral fabric definitions.

**Acronyms:**
- CRSS = Critical Resolved Shear Stress,
    i.e. threshold stress required to initiate slip on a slip system,
    normalised to the stress required to initiate slip on the softest slip system

"""
import io
from dataclasses import dataclass, field
from zipfile import ZipFile

import numpy as np
from scipy import linalg as la
from scipy.integrate import LSODA
from scipy.spatial.transform import Rotation

from pydrex import core as _core
from pydrex import deformation_mechanism as _defmech
from pydrex import exceptions as _err
from pydrex import io as _io
from pydrex import logger as _log


OLIVINE_STIFFNESS = np.array(
    [
        [320.71, 69.84, 71.22, 0.0, 0.0, 0.0],
        [69.84, 197.25, 74.8, 0.0, 0.0, 0.0],
        [71.22, 74.8, 234.32, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 63.77, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 77.67, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 78.36],
    ]
)
"""Stiffness tensor for olivine, with units of GPa.

The source of the values used here is unknown, but they are copied
from the original DRex code: <http://www.ipgp.fr/~kaminski/web_doudoud/DRex.tar.gz> [88K download]

"""


ENSTATITE_STIFFNESS = np.array(
    [
        [236.9, 79.6, 63.2, 0.0, 0.0, 0.0],
        [79.6, 180.5, 56.8, 0.0, 0.0, 0.0],
        [63.2, 56.8, 230.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 84.3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 79.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 80.1],
    ]
)
"""Stiffness tensor for enstatite, with units of GPa.

The source of the values used here is unknown, but they are copied
from the original DRex code: <http://www.ipgp.fr/~kaminski/web_doudoud/DRex.tar.gz> [88K download]

"""


OLIVINE_PRIMARY_AXIS = {
    _core.MineralFabric.olivine_A: "a",
    _core.MineralFabric.olivine_B: "c",
    _core.MineralFabric.olivine_C: "c",
    _core.MineralFabric.olivine_D: "a",
    _core.MineralFabric.olivine_E: "a",
}
"""Primary slip axis name for for the given olivine `fabric`."""


OLIVINE_SLIP_SYSTEMS = (
    ([0, 1, 0], [1, 0, 0]),
    ([0, 0, 1], [1, 0, 0]),
    ([0, 1, 0], [0, 0, 1]),
    ([1, 0, 0], [0, 0, 1]),
)
"""Slip systems for olivine in conventional order.

Tuples contain the slip plane normal and slip direction vectors.
The order of slip systems returned matches the order of critical shear stresses
returned by `get_crss`.

"""


@dataclass
class Mineral:
    """Class for storing mineral CPO.

    A `Mineral` stores CPO data for an aggregate of grains*.
    Additionally, mineral fabric type and deformation regime
    are also tracked. See `pydrex.deformation_mechanism`.

    Attributes:
    - `phase` (int) — ordinal number of the mineral phase, see `MineralPhase`
    - `fabric` (int) — ordinal number of the fabric type, see `MineralFabric.olivine_
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

    phase: int = _core.MineralPhase.olivine
    fabric: int = _core.MineralFabric.olivine_A
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
        get_velocity_gradient,
        pathline,
        **kwargs,
    ):
        """Update orientations and volume distribution for the `Mineral`.

        Update crystalline orientations and grain volume distribution
        for minerals undergoing plastic deformation.

        Args:
        - `config` (dict) — PyDRex configuration dictionary
        - `deformation_gradient` (array) — 3x3 initial deformation gradient tensor
        - `get_velocity_gradient` (function) — callable with signature f(x) that returns
          a 3x3 velocity gradient matrix at position x (vector)
        - `pathline` (tuple) — tuple consisting of:
            1. the time at which to start the CPO integration (t_start)
            2. the time at which to stop the CPO integration (t_end)
            3. callable with signature f(t) that returns the position of the mineral at
                time t ∈ [t_start, t_end]

        Any additional (optional) keyword arguments are passed to
        `scipy.integrate.LSODA`.

        Array values must provide a NumPy-compatible interface:
        <https://numpy.org/doc/stable/user/whatisnumpy.html>

        """

        # ===== Set up callables for the ODE solver and internal processing =====

        def extract_vars(y):
            # TODO: Check if we can avoid .copy() here.
            deformation_gradient = y[:9].copy().reshape((3, 3))
            orientations = (
                y[9 : self.n_grains * 9 + 9]
                .copy()
                .reshape((self.n_grains, 3, 3))
                .clip(-1, 1)
            )
            # TODO: Check if we can avoid .copy() here.
            fractions = (
                y[self.n_grains * 9 + 9 : self.n_grains * 10 + 9].copy().clip(0, None)
            )
            fractions /= fractions.sum()
            return deformation_gradient, orientations, fractions

        def eval_rhs(t, y):
            """Evaluate right hand side of the D-Rex PDE."""
            # assert not np.any(np.isnan(y)), y[np.isnan(y)].shape
            position = get_position(t)
            velocity_gradient = get_velocity_gradient(position)
            _log.debug(
                "calculating CPO at %s (t=%e) with velocity gradient %s",
                position,
                t,
                velocity_gradient.flatten(),
            )

            strain_rate = (velocity_gradient + velocity_gradient.transpose()) / 2
            strain_rate_max = np.abs(la.eigvalsh(strain_rate)).max()
            deformation_gradient, orientations, fractions = extract_vars(y)
            # Uses nondimensional values of strain rate and velocity gradient.
            orientations_diff, fractions_diff = _core.derivatives(
                phase=self.phase,
                fabric=self.fabric,
                n_grains=self.n_grains,
                orientations=orientations,
                fractions=fractions,
                strain_rate=strain_rate / strain_rate_max,
                velocity_gradient=velocity_gradient / strain_rate_max,
                stress_exponent=config["stress_exponent"],
                deformation_exponent=config["deformation_exponent"],
                nucleation_efficiency=config["nucleation_efficiency"],
                gbm_mobility=config["gbm_mobility"],
                volume_fraction=volume_fraction,
            )
            return np.hstack(
                (
                    (velocity_gradient @ deformation_gradient).flatten(),
                    orientations_diff.flatten() * strain_rate_max,
                    fractions_diff * strain_rate_max,
                )
            )

        def apply_gbs(orientations, fractions, config):
            """Apply grain boundary sliding for small grains."""
            mask = fractions < config["gbs_threshold"] / self.n_grains
            _log.debug(
                "grain boundary sliding activity (volume percentage): %s",
                len(np.nonzero(mask)) / len(fractions),
            )
            # No rotation: carry over previous orientations.
            # TODO: Should we really be resetting to initial orientations here?
            orientations[mask, :, :] = self.orientations[0][mask, :, :]
            fractions[mask] = config["gbs_threshold"] / self.n_grains
            fractions /= fractions.sum()
            _log.debug(
                "grain volume fractions: median=%e, min=%e, max=%e, sum=%e",
                np.median(fractions),
                np.min(fractions),
                np.max(fractions),
                np.sum(fractions),
            )
            return orientations, fractions

        def perform_step(solver):
            """Perform SciPy solver step and appropriate processing."""
            message = solver.step()
            if message is not None and solver.status == "failed":
                raise _err.IterationError(message)
            _log.debug(
                "%s step_size=%e", solver.__class__.__qualname__, solver.step_size
            )

            deformation_gradient, orientations, fractions = extract_vars(solver.y)
            orientations, fractions = apply_gbs(orientations, fractions, config)
            solver.y[9:] = np.hstack((orientations.flatten(), fractions))

        # ===== Initialise and run the solver using the above callables =====

        time_start, time_end, get_position = pathline
        if not callable(get_velocity_gradient):
            raise ValueError(
                "unable to evaluate velocity gradient callable."
                + " You must provide a callable with signature f(x)"
                + " that returns a 3x3 matrix."
            )
        if not callable(get_position):
            raise ValueError(
                "unable to evaluate position callable."
                + " You must provide a callable with signature f(t)"
                + " that returns a 3-component array."
            )

        if self.phase == _core.MineralPhase.olivine:
            volume_fraction = config["olivine_fraction"]
        elif self.phase == _core.MineralPhase.enstatite:
            volume_fraction = config["enstatite_fraction"]
        else:
            assert False  # Should never happen.

        y_start = np.hstack(
            (
                deformation_gradient.flatten(),
                self.orientations[-1].flatten(),
                self.fractions[-1],
            )
        )
        solver = LSODA(
            eval_rhs,
            time_start,
            y_start,
            time_end,
            atol=kwargs.pop("atol", np.abs(y_start * 1e-6) + 1e-12),
            rtol=kwargs.pop("rtol", 1e-6),
            first_step=kwargs.pop("first_step", np.abs(time_end - time_start) * 1e-1),
            # max_step=kwargs.pop("max_step", np.abs(time_end - time_start)),
            **kwargs,
        )
        perform_step(solver)
        while solver.status == "running":
            perform_step(solver)

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
            # Create parent directories, resolve relative paths.
            _io.resolve_path(filename)
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
