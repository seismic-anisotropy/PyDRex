"""PyDRex: Fabric and RRSS selection and helpers.

Acronyms:
- RRSS = Reference Resolved Shear Stress,
    i.e. components of stress acting on each slip system in the grain reference frame

"""
from enum import Enum, unique

import numpy as np

import pydrex.minerals as _minerals


@unique
class OlivineFabric(Enum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4


@unique
class EnstatiteFabric(Enum):
    A = 0  # Just to make it consistent.


RRSS_OLIVINE = {
    OlivineFabric.A: np.array([1, 2, 3, np.inf]),
    OlivineFabric.B: np.array([3, 2, 1, np.inf]),
    OlivineFabric.C: np.array([3, 2, np.inf, 1]),
    OlivineFabric.D: np.array([1, 1, 3, np.inf]),
    OlivineFabric.E: np.array([3, 1, 2, np.inf]),
}

RRSS_ENSTATITE = {
    EnstatiteFabric.A: np.array([np.inf, np.inf, np.inf, 1]),
}

RRSS = {
    _minerals.MineralPhase.olivine: RRSS_OLIVINE,
    _minerals.MineralPhase.enstatite: RRSS_ENSTATITE,
}
