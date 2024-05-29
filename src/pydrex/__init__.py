r"""
#### Simulate crystallographic preferred orientation evolution in polycrystals

---

.. warning::
    **This software is currently in early development (alpha)
    and therefore subject to breaking changes without notice.**

## About

Viscoplastic deformation of minerals, e.g. in Earth's mantle, leads to distinct
signatures in the mineral texture. Many minerals naturally occur in
polycrystalline form, which means that they are composed of many grains with
different volumes and lattice orientations. Preferential alignment of the
average lattice orientation is called crystallographic preferred orientation
(CPO). PyDRex simulates the development and evolution of CPO in deforming
polycrystals, as well as tracking macroscopic finite strain measures.
Currently, the code supports olivine and enstatite mineral phases. The
following features are provided:
- JIT-compiled CPO solver, based on the D-Rex model, which updates the
  polycrystal orientation distribution depending on the macroscopic velocity
  gradients
- `Mineral` class which stores attributes of a distinct mineral phase in the
  polycrystal and its texture snapshots
- Voigt averaging to calculate the average elastic tensor of a textured,
  multiphase polycrystal
- Decomposition of average elastic tensors into components attributed to
  minerals with distinct lattice symmetries
- Crystallographic pole figure visualisation (contouring is a work in progress)
- [work in progress] Texture diagnostics: M-index, bingham average,
  Point-Girdle-Random symmetry, coaxial a.k.a "BA" index, etc.
- [work in progress] Seismic anisotropy diagnostics: % tensorial anisotropy,
  hexagonal symmetry a.k.a transverse isotropy direction, etc.

The core CPO solver is based on the original Fortran 90 implementation by Édouard Kaminski,
which can be [downloaded from this link (~90KB)](http://www.ipgp.fr/~kaminski/web_doudoud/DRex.tar.gz).
The reference papers are [Kaminski & Ribe (2001)](https://doi.org/10.1016/s0012-821x(01)00356-9)
and [Kaminski & Ribe (2004)](https://doi.org/10.1111%2Fj.1365-246x.2004.02308.x),
and an open-access paper which discusses the model is [Fraters & Billen (2021)](https://doi.org/10.1029/2021gc009846).

## Installation

The minimum required Python version is set using `requires-python` in the
[`pyproject.toml`](https://github.com/seismic-anisotropy/PyDRex/blob/main/pyproject.toml) file.
For installation instructions,
see [the README](https://github.com/seismic-anisotropy/PyDRex/blob/main/README.md) file.

## Documentation

The website menu can be used to discover the public API of this package.
Some of the tests are also documented and can serve as usage examples.
Their docstrings can be viewed [in this section](../tests.html).
Documentation is also available from the Python REPL via the `help()` method.

## The D-Rex kinematic CPO model

The D-Rex model is used to compute crystallographic preferred orientation (CPO)
for polycrystals deforming by plastic deformation and dynamic recrystallization.
Polycrystals are discretized into "grains" which represent fractional volumes
of the total crystal that are characterised by a particular crystal lattice
orientation. For numerical efficiency, the number of grains in the model does
not change, and should only be interpreted as an approximation of the number
of (unrecrystallised) physical grains. Dynamic recrystallization is modelled using
statistical expressions which approximate the interaction of each grain with an
effective medium based on the averaged dislocation energy of all other grains. Note that
the model is not suited to situations where static recrystallization processes are
significant.

The primary microphysical mechanism for plastic deformation of the polycrystal
is dislocation creep, which involves dislocation glide ("slip") along symmetry
planes of the mineral and dislocation climb, which allows for dislocations to
annihilate each other so that the number of dislocations reaches a steady-state.
The D-Rex model does not simulate dislocation climb, but implicitly assumes that
the dislocations are in steady-state so that the dislocation density of the
crystal can be described by

$$
ρ ∝ b^{-2} \left(\frac{σ}{μ}\right)^{p}
$$

where $b$ is the length of the Burgers' vector, $σ$ is the stress
and $μ$ is the shear modulus. The value of the exponent $p$ is given by the
`stress_exponent` input parameter. For an overview of available parameters,
see [the `pydrex.mock` source code, for now...]

The effects of dynamic recrystallization are twofold. Grains with a higher than
average dislocation density may be affected by either grain nucleation, which is
the formation of initially small, strain-free sub-grains, or grain boundary
migration, by which process other grains of lower strain energy annex a portion
of its volume. Nucleation occurs mostly in grains oriented favourably for
dislocation glide, and the new grains also grow by grain boundary migration.
If nucleation is too inefficient, the dislocation density in deformation-aligned
grains will remain high and these grains will therefore shrink in volume. On the
other hand, if grain boundaries are too immobile, then nucleated grains will take
longer to grow, reducing the speed of CPO development and re-orientation.
Because nucleated grains are assumed to inherit the orientation of the parent,
they do not affect the model except by reducing the average dislocation density.
A grain boundary mobility parameter of $M^{∗} = 0$ will therefore disable any
recrystallization effects. Finally, the process of grain boundary sliding can
also be included, which simply disallows rotation of grains with very small volume.
This only affects CPO evolution by introducing a latency for the onset of grain
boundary migration in nucleated grains. It also manifests as an upper bound on
texture strength.

## Parameter reference

Model parameters will eventually be provided in a `.toml` file.
For now just pass a dictionary to `config` in the `Mineral.update_orientations` method.
A draft of the input file spec is shown below:

```toml
.. include:: data/specs/spec.toml
```

"""

# Set up the top-level pydrex namespace for convenient usage.
# To keep it clean, we don't want every single symbol here, especially not those from
# `utils` or `visualisation` modules, which should be explicitly imported instead.
import pydrex.axes  # Defines the 'pydrex.polefigure' Axes subclass.
from pydrex.core import (
    DefaultParams,
    DeformationRegime,
    MineralFabric,
    MineralPhase,
    derivatives,
    get_crss,
)
from pydrex.diagnostics import (
    bingham_average,
    coaxial_index,
    elasticity_components,
    finite_strain,
    misorientation_index,
    misorientation_indices,
    smallest_angle,
    symmetry_pgr,
)
from pydrex.geometry import (
    LatticeSystem,
    lambert_equal_area,
    misorientation_angles,
    poles,
    shirley_concentric_squaredisk,
    symmetry_operations,
    to_cartesian,
    to_spherical,
)
from pydrex.io import data, read_scsv, save_scsv
from pydrex.minerals import (
    OLIVINE_PRIMARY_AXIS,
    OLIVINE_SLIP_SYSTEMS,
    StiffnessTensors,
    Mineral,
    voigt_averages,
)
from pydrex.pathlines import get_pathline
from pydrex.stats import (
    misorientation_hist,
    misorientations_random,
    resample_orientations,
)
from pydrex.tensors import (
    elastic_tensor_to_voigt,
    rotate,
    voigt_decompose,
    voigt_matrix_to_vector,
    voigt_to_elastic_tensor,
    voigt_vector_to_matrix,
)
