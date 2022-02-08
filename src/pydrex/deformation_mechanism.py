"""PyDRex: Deformation mechanism enums.

Public symbols:
- `Regime`

"""
__all__ = ["Regime"]

from enum import Enum, unique


@unique
class Regime(Enum):
    diffusion = 0
    dislocation = 1
    byerlee = 2
    max_viscosity = 3
