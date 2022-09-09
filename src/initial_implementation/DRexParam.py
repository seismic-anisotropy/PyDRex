#!/usr/bin/env python3
from numpy import amax, amin, array, diag, hstack, inf, linspace

name = "Tag"  # A little tag to keep track

dim = 2  # Number of dimensions; must be 2 or 3
assert dim in [2, 3]

# Generate a regular grid
uniformGridSpacing = True
if uniformGridSpacing:  # As many entries as dimensions: X, [Y], Z
    gridRes = array([4e3, 4e3])
    gridMin = array([0, 0])
    gridMax = array([1.2e6, 4e5]) + gridRes
    gridNodes = ((gridMax - gridMin) / gridRes + 1).astype(int)
    gridCoords = [linspace(x, y, z) for x, y, z in zip(gridMin, gridMax, gridNodes)]
else:  # As many variables as dimensions: X, [Y], Z
    xCoords = hstack(
        (
            linspace(5e5, 1.3e6, 161),
            linspace(1.3025e6, 1.7e6, 160),
            linspace(1.705e6, 2.5e6, 160),
        )
    )
    yCoords = hstack(
        (
            linspace(0, 3e5, 13),
            linspace(3.2e5, 4e5, 5),
            linspace(4.1e5, 4.5e5, 5),
            linspace(4.55e5, 4.8e5, 6),
            linspace(4.825e5, 5.2e5, 16),
            linspace(5.25e5, 5.5e5, 6),
            linspace(5.6e5, 6e5, 5),
            linspace(6.2e5, 7e5, 5),
            linspace(7.25e5, 1e6, 12),
        )
    )
    zCoords = hstack(
        (
            linspace(0, 1e5, 51),
            linspace(1.04e5, 2e5, 25),
            linspace(2.1e5, 3e5, 10),
            linspace(3.2e5, 4e5, 5),
        )
    )
    gridCoords = [xCoords, yCoords, zCoords]
    gridNodes = [arr.size for arr in gridCoords]
    gridMin = [amin(arr) for arr in gridCoords]
    gridMax = [amax(arr) for arr in gridCoords]
assert dim == len(gridCoords)

checkpoint = 3000  # Number of nodes after which a checkpoint should be dumped

Xol = 0.7  # Fraction of olivine in the aggregate
tau = array([1, 2, 3, inf])  # Olivine
tau_ens = 1  # Enstatite
mob = 125  # Grain boundary mobility
chi = 0.3  # Threshold volume fraction for grain boundary sliding
lamb = 5  # Nucleation parameter set to 5 as in Kaminski and Ribe (2001)
size = 13**3  # Initial size | Number of points in the Eulerian space
stressexp = 3.5  # Stress exponent for olivine and enstatite
# Stiffness matrix for Olivine (GPa)
S0 = diag((320.71, 197.25, 234.32, 63.77, 77.67, 78.36))
S0[0, 1] = S0[1, 0] = 69.84
S0[0, 2] = S0[2, 0] = 71.22
S0[1, 2] = S0[2, 1] = 74.8
# Stiffness matrix for Enstatite (GPa)
S0_ens = diag((236.9, 180.5, 230.4, 84.3, 79.4, 80.1))
S0_ens[0, 1] = S0_ens[1, 0] = 79.6
S0_ens[0, 2] = S0_ens[2, 0] = 63.2
S0_ens[1, 2] = S0_ens[2, 1] = 56.8
