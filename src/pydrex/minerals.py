"""> PyDRex: Computations of mineral texture and elasticity.

**Acronyms:**
- CPO = Crystallographic preferred orientation,
    i.e. preferential clustering of polycrystal grain orientations in SO(3),
    leading to an overall anisotropic orientation distribution

"""

import io
from dataclasses import dataclass, field
from zipfile import ZipFile

import numpy as np
from numpy import asarray as Ŋ
from scipy import linalg as la
from scipy.integrate import LSODA
from scipy.spatial.transform import Rotation

from pydrex import core as _core
from pydrex import exceptions as _err
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import tensors as _tensors
from pydrex import utils as _utils

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
"""Stiffness tensor for olivine (Voigt representation), with units of GPa.

The source of the values used here is unknown, but they are copied
from the original DRex code: <http://www.ipgp.fr/~kaminski/web_doudoud/DRex.tar.gz>
[88K download]

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
"""Stiffness tensor for enstatite (Voigt representation), with units of GPa.

The source of the values used here is unknown, but they are copied
from the original DRex code: <http://www.ipgp.fr/~kaminski/web_doudoud/DRex.tar.gz>
[88K download]

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
returned by `pydrex.core.get_crss`.

"""


# TODO: Compare to [Man & Huang, 2011](https://doi.org/10.1007/s10659-011-9312-y).
def voigt_averages(minerals, weights):
    """Calculate elastic tensors as the Voigt averages of a collection of `mineral`s.

    Args:
    - `minerals` — list of `pydrex.minerals.Mineral` instances storing orientations and
      fractional volumes of the grains within each distinct mineral phase
    - `weights` (dict) — dictionary containing weights of each mineral
      phase, as a fraction of 1, in keys named "<phase>_fraction",
      e.g. "olivine_fraction"

    Raises a ValueError if the minerals contain an unequal number of grains or stored
    texture results.

    """
    n_grains = minerals[0].n_grains
    if not np.all([m.n_grains == n_grains for m in minerals[1:]]):
        raise ValueError("cannot average minerals with unequal grain counts")
    n_steps = len(minerals[0].orientations)
    if not np.all([len(m.orientations) == n_steps for m in minerals[1:]]):
        raise ValueError(
            "cannot average minerals with variable-length orientation arrays"
        )
    if not np.all([len(m.fractions) == n_steps for m in minerals]):
        raise ValueError(
            "cannot average minerals with variable-length grain volume arrays"
        )

    elastic_tensors = {}

    # TODO: Perform rotation directly on the 6x6 matrices, see Carcione 2007.
    # This trick is implemented in cpo_elastic_tensor.cc in Aspect.
    average_tensors = np.zeros((n_steps, 6, 6))
    for i in range(n_steps):
        for mineral in minerals:
            for n in range(n_grains):
                match mineral.phase:
                    case _core.MineralPhase.olivine:
                        if "olivine" not in elastic_tensors:
                            elastic_tensors["olivine"] = (
                                _tensors.voigt_to_elastic_tensor(OLIVINE_STIFFNESS)
                            )
                        average_tensors[i] += _tensors.elastic_tensor_to_voigt(
                            _tensors.rotate(
                                elastic_tensors["olivine"],
                                mineral.orientations[i][n, ...].transpose(),
                            )
                            * mineral.fractions[i][n]
                            * weights["olivine_fraction"]
                        )
                    case _core.MineralPhase.enstatite:
                        if "enstatite" not in elastic_tensors:
                            elastic_tensors["enstatite"] = (
                                _tensors.voigt_to_elastic_tensor(ENSTATITE_STIFFNESS)
                            )
                        average_tensors[i] += _tensors.elastic_tensor_to_voigt(
                            _tensors.rotate(
                                elastic_tensors["enstatite"],
                                minerals.orientations[i][n, ...].transpose(),
                            )
                            * mineral.fractions[i][n]
                            * weights["enstatite_fraction"]
                        )
                    case _:
                        raise ValueError(f"unsupported mineral phase: {mineral.phase}")
    return average_tensors


@dataclass
class Mineral:
    """Class for storing polycrystal texture for a single mineral phase.

    A `Mineral` stores texture data for an aggregate of grains*.
    Additionally, mineral fabric type and deformation regime are also tracked.
    To provide an initial texture for the mineral, use the constructor arguments
    `fractions_init` and `orientations_init`. By default,
    a uniform volume distribution of random orientations is generated.

    The `update_orientations` method computes new orientations and grain volumes
    for a given velocity gradient. These results are stored in the `.orientations` and
    `.fractions` attributes of the `Mineral` instance. The method also returns the
    updated macroscopic deformation gradient based on the provided initial deformation
    gradient.

    *Note that the "number of grains" is a static integer value that
    does not track the actual number of physical grains in the deforming polycrystal.
    Instead, this number acts as a "number of bins" for the statistical resolution of
    the crystallographic orientation distribution. The value is roughly equivalent to
    (a multiple of) the number of initial, un-recrystallised grains in the polycrystal.
    It is assumed that recrystallised grains do not grow large enough to require
    rotation tracking.

    **Examples:**

    Mineral with isotropic initial texture:
    >>> import pydrex
    >>> olA = pydrex.Mineral(
    ...     phase=pydrex.MineralPhase.olivine,
    ...     fabric=pydrex.MineralFabric.olivine_A,
    ...     regime=pydrex.DeformationRegime.dislocation,
    ...     n_grains=2000
    ... )
    >>> olA.phase
    <MineralPhase.olivine: 0>
    >>> olA.fabric
    <MineralFabric.olivine_A: 0>
    >>> olA.regime
    <DeformationRegime.dislocation: 1>
    >>> olA.n_grains
    2000

    Mineral with specified initial texture and default phase, fabric and regime settings
    which are for an olivine A-type mineral in the dislocation creep regime.
    The initial grain volume fractions should be normalised.
    >>> import numpy as np
    >>> from scipy.spatial.transform import Rotation
    >>> import pydrex
    >>> rng = np.random.default_rng()
    >>> n_grains = 2000
    >>> olA = pydrex.Mineral(
    ...     n_grains=n_grains,
    ...     fractions_init=np.full(n_grains, 1 / n_grains),
    ...     orientations_init=Rotation.from_euler(
    ...         "zxz", [
    ...             [x * np.pi / 2, np.pi / 2, np.pi / 2] for x in rng.random(n_grains)
    ...         ]
    ...     ).inv().as_matrix(),
    ... )
    >>> len(olA.orientations)
    1
    >>> type(olA.orientations)
    <class 'list'>
    >>> olA.orientations[0].shape
    (2000, 3, 3)
    >>> olA.fractions[0].shape
    (2000,)

    Note that minerals can also be constructed from serialized data,
    see `Mineral.load` and `Mineral.from_file`.

    **Attributes:**
    - `phase` (`pydrex.core.MineralPhase`) — ordinal number of the mineral phase
    - `fabric` (`pydrex.core.MineralFabric`) — ordinal number of the fabric type
    - `regime` (`pydrex.core.DeformationRegime`) — ordinal number of the deformation
      regime
    - `n_grains` (int) — number of grains in the aggregate
    - `fractions` (list of arrays) — grain volume fractions for each texture snapshot
    - `orientations` (list of arrays) — grain orientation matrices for each texture
      snapshot
    - `seed` (int) — seed used by the random number generator to set up the isotropic
      initial condition when `fractions_init` or `orientations_init` are not provided
    - `lband` (int) — passed to the `scipy.integrate.LSODA` solver
    - `uband` (int) — passed to the `scipy.integrate.LSODA` solver

    """

    phase: int = _core.MineralPhase.olivine
    fabric: int = _core.MineralFabric.olivine_A
    regime: int = _core.DeformationRegime.dislocation
    n_grains: int = _io.DEFAULT_PARAMS["number_of_grains"]
    # Initial condition, randomised if not given.
    fractions_init: np.ndarray = None
    orientations_init: np.ndarray = None
    fractions: list = field(default_factory=list)
    orientations: list = field(default_factory=list)
    seed: int = None
    lband: int = None
    uband: int = None

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

        obj = self.__class__.__qualname__
        phase = f"(phase={self.phase!r}, "
        fabric = f"fabric={self.fabric!r}, "
        regime = f"regime={self.regime!r}, "
        n_grains = f"n_grains={self.n_grains}, "
        _fclass = self.fractions.__class__.__qualname__
        _f0class = self.fractions[0].__class__.__qualname__
        frac = f"fractions=<{_fclass} of {_f0class} {shape_of_fractions}>, "
        _oclass = self.orientations.__class__.__qualname__
        _o0class = self.orientations[0].__class__.__qualname__
        orient = f"orientations=<{_oclass} of {_o0class} {shape_of_orientations}>)"
        return f"{obj}{phase}{fabric}{regime}{n_grains}{frac}{orient}"

    def _repr_pretty_(self, p, cycle):
        # Format to use when printing to IPython or other interactive console.
        p.text(self.__str__() if not cycle else self.__class__.__qualname__ + "(...)")

    def __post_init__(self):
        """Initialise random orientations and grain volume fractions."""
        if self.fractions_init is None:
            self.fractions_init = np.full(self.n_grains, 1.0 / self.n_grains)
        if self.orientations_init is None:
            self.orientations_init = Rotation.random(
                self.n_grains, random_state=self.seed
            ).as_matrix()
        # For large numbers of grains, the number of ODE's exceeds what LSODA can
        # handle. Therefore, we specify the Jacobian matrix as banded.
        # By default, we use a bandwidth o f 12000 with lband = uband = 6000.
        # This should work for up to 10000 grains.
        if self.lband is None and self.uband is None and self.n_grains > 4632:
            _log.warning(
                "using a banded Jacobian because of the large number of grains."
                + " To manually control the bandwidth, set `lband` and/or `uband`"
                + f" in calls to `{self.__class__.__qualname__}.update_orientations`."
            )
            self.lband = 6000
            self.uband = 6000

        # Copy the initial values to the storage lists.
        self.fractions.append(self.fractions_init)
        self.orientations.append(self.orientations_init)

        # Delete the initial value duplicates to avoid confusion.
        del self.fractions_init
        del self.orientations_init

        _log.info("created %s", self)

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return (
                self.phase == other.phase
                and self.fabric == other.fabric
                and self.regime == other.regime
                and self.n_grains == other.n_grains
                and len(self.fractions) == len(other.fractions)
                and np.all(
                    Ŋ([f.shape for f in self.fractions])
                    == Ŋ([f.shape for f in other.fractions])
                )
                and np.all(Ŋ(self.fractions) == Ŋ(other.fractions))
                and len(self.orientations) == len(other.orientations)
                and np.all(
                    Ŋ([f.shape for f in self.orientations])
                    == Ŋ([f.shape for f in other.orientations])
                )
                and np.all(Ŋ(self.orientations) == Ŋ(other.orientations))
                and self.seed == other.seed
                and self.lband == other.lband
                and self.uband == other.uband
            )
        return False

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
        for minerals undergoing plastic deformation. Return the updated deformation
        gradient measuring the corresponding macroscopic deformation.

        Args:
        - `config` (dict) — PyDRex configuration dictionary
        - `deformation_gradient` (array) — 3x3 initial deformation gradient tensor
        - `get_velocity_gradient` (function) — callable with signature f(t, x) that
          returns a 3x3 velocity gradient matrix at time t at position x (3D vector)
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

        def eval_rhs(t, y):
            """Evaluate right hand side of the D-Rex PDE."""
            # assert not np.any(np.isnan(y)), y[np.isnan(y)].shape
            position = get_position(t)
            velocity_gradient = get_velocity_gradient(t, position)
            # _log.debug(
            #     "calculating CPO at %s (t=%e) with velocity gradient %s",
            #     position,
            #     t,
            #     velocity_gradient.ravel(),
            # )

            if self.phase == _core.MineralPhase.olivine:
                volume_fraction = config["olivine_fraction"]
            elif self.phase == _core.MineralPhase.enstatite:
                volume_fraction = config["enstatite_fraction"]
            else:
                raise ValueError(
                    f"phase must be a valid `MineralPhase`, not {self.phase}"
                )

            strain_rate = (velocity_gradient + velocity_gradient.transpose()) / 2
            strain_rate_max = np.abs(la.eigvalsh(strain_rate)).max()
            deformation_gradient, orientations, fractions = _utils.extract_vars(
                y, self.n_grains
            )
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

        def perform_step(solver):
            """Perform SciPy solver step and appropriate processing."""
            message = solver.step()
            if message is not None and solver.status == "failed":
                raise _err.IterationError(message)
            # _log.debug(
            #     "%s step_size=%e", solver.__class__.__qualname__, solver.step_size
            # )

            deformation_gradient, orientations, fractions = _utils.extract_vars(
                solver.y, self.n_grains
            )
            orientations, fractions = _utils.apply_gbs(
                orientations,
                fractions,
                config["gbs_threshold"],
                self.orientations[-1],
                self.n_grains,
            )
            solver.y[9:] = np.hstack((orientations.flatten(), fractions))

        # ===== Initialise and run the solver using the above callables =====

        time_start, time_end, get_position = pathline
        if not callable(get_velocity_gradient):
            raise ValueError(
                "unable to evaluate velocity gradient callable."
                + " You must provide a callable with signature f(t, x)"
                + " that returns a 3x3 matrix."
            )
        if not callable(get_position):
            raise ValueError(
                "unable to evaluate position callable."
                + " You must provide a callable with signature f(t)"
                + " that returns a 3-component array."
            )
        _log.debug(
            "calculating CPO from %s (t=%s) to %s (t=%s)",
            get_position(time_start),
            time_start,
            get_position(time_end),
            time_end,
        )
        _log.debug("    with deformation gradient %s", deformation_gradient.ravel())
        _log.debug(
            "    with velocity gradient interpolated between %s and %s",
            get_velocity_gradient(time_start, get_position(time_start)).ravel(),
            get_velocity_gradient(time_end, get_position(time_end)).ravel()
        )
        _log.debug(
            "    intermediate velocity gradient = %s",
            get_velocity_gradient(
                (time_start + time_end) / 2, get_position((time_start + time_end) / 2)
            ).ravel(),
        )

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
            lband=self.lband,
            uband=self.uband,
            **kwargs,
        )
        perform_step(solver)
        while solver.status == "running":
            perform_step(solver)

        # Extract final values for this simulation step, append to storage.
        deformation_gradient, orientations, fractions = _utils.extract_vars(
            solver.y.squeeze(), self.n_grains
        )
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
                _log.info("saving Mineral to file %s (postfix: %s)", filename, postfix)
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
                _log.info("saving Mineral to file %s", filename)
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
