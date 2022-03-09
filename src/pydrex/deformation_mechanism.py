"""PyDRex: Deformation mechanism enums."""
from enum import IntEnum, unique


@unique
class Regime(IntEnum):
    """Deformation mechanism regimes.

    - diffusion creep (0)
    - dislocation creep (1)
    - Byerlee's law (2)
    - maximum viscosity region (3)

    """
    diffusion = 0
    dislocation = 1
    byerlee = 2
    max_viscosity = 3
