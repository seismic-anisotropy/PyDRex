#!/usr/bin/env python3

# Note: To benefit from SVML support available in Numba, install icc_rt package
# and locate libsvml.so; append the corresponding directory to the environment
# variable LD_LIBRARY_PATH.

# Global parameters used in PyDRex.py
from numpy import array, diag

gridRes = array([8e3, 2e4, 2e3])  # X, [Y], Z
gridMin = array([0, 2e5, 0])  # X, [Y], Z
gridMax = array([1.2e6, 3e5, 4e5]) + gridRes  # X, [Y], Z

checkpoint = 2000

Xol = 0.7  # Fraction of olivine in the aggregate
tau = array([1, 2, 3, 1e6])  # Olivine
tau_ens = 1  # Enstatite
mob = 125  # Grain boundary mobility
chi = 0.2  # Threshold volume fraction for GBS
lamb = 5  # Nucleation parameter set to 5 as in Kaminski and Ribe (2001)
size = 13 ** 3  # Initial size | Number of points in the Eulerian space
stressexp = 3.5  # Stress exponent for olivine and enstatite
# Stiffness matrix for Olivine (GPa)
S0 = diag((320.71, 197.25, 234.32, 63.77, 77.67, 78.36))
S0[0, 1] = 69.84
S0[0, 2] = 71.22
S0[1, 2] = 74.8
# Stiffness matrix for Enstatite (GPa)
S0_ens = diag((236.9, 180.5, 230.4, 84.3, 79.4, 80.1))
S0_ens[0, 1] = 79.6
S0_ens[0, 2] = 63.2
S0_ens[1, 2] = 56.8
