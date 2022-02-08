"""PyDRex: Fabric selection and helpers.

Acronyms:
- RRSS = Reference Resolved Shear Stress, i.e. components of stress acting on each slip system in the grain reference frame

"""
import numpy as np


RRSS_OLIVINE_A = np.array([1, 2, 3, np.inf])
RRSS_OLIVINE_B = np.array([3, 2, 1, np.inf])
RRSS_OLIVINE_C = np.array([3, 2, np.inf, 1])
RRSS_OLIVINE_D = np.array([1, 1, 3, np.inf])
RRSS_OLIVINE_E = np.array([3, 1, 2, np.inf])


RRSS_ENSTATITE = np.array([np.inf, np.inf, np.inf, 1])
