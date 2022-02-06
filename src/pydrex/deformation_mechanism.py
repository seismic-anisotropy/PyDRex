"""PyDRex: Deformation mechanism enums.

Public symbols:
- `Regime`

"""
__all__ = ["Regime"]

import enum as e


@e.unique
class Regime(e.Enum):
    diffusion = 0
    dislocation = 1
    byerlee = 2
    max_viscosity = 3
