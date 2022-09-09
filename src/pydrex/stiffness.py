"""PyDRex: Stiffness matrices for minerals.

The provided matrices encode the stiffness tensor in the compact Voigt notation.
The stiffness matrices contain values of the elastic constants in GPa.
Currently, stiffness matrices are provided for olivine and enstatite.

The source of the values used here is unknown, but they are copied
from the original DRex code:

<http://www.ipgp.fr/~kaminski/web_doudoud/DRex.tar.gz> [88K download]

"""
import numpy as np

OLIVINE = np.array(
    [
        [320.71, 69.84, 71.22, 0.0, 0.0, 0.0],
        [69.84, 197.25, 74.8, 0.0, 0.0, 0.0],
        [71.22, 74.8, 234.32, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 63.77, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 77.67, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 78.36],
    ]
)

ENSTATITE = np.array(
    [
        [236.9, 79.6, 63.2, 0.0, 0.0, 0.0],
        [79.6, 180.5, 56.8, 0.0, 0.0, 0.0],
        [63.2, 56.8, 230.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 84.3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 79.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 80.1],
    ]
)
