"""PyDRex: Deformation mechanism enums."""
from enum import Enum, unique


@unique
class Regime(Enum):
    diffusion = 0
    dislocation = 1
    byerlee = 2
    max_viscosity = 3
