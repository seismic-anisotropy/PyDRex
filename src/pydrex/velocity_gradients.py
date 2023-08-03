"""> PyDRex: Steady-state solutions of velocity gradients for various flows."""
import functools as ft

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def _simple_shear_2d(x, direction=1, deformation_plane=0, strain_rate=1):
    grad_v = np.zeros((3, 3))
    grad_v[direction, deformation_plane] = 2 * strain_rate
    return grad_v


def simple_shear_2d(direction, deformation_plane, strain_rate):
    """Return simple shear velocity gradient callable f(x) for the given parameters."""
    shear = (direction, deformation_plane)
    match shear:
        case ("X", "Y"):
            _shear = (0, 1)
        case ("X", "Z"):
            _shear = (0, 2)
        case ("Y", "X"):
            _shear = (1, 0)
        case ("Y", "Z"):
            _shear = (1, 2)
        case ("Z", "X"):
            _shear = (2, 0)
        case ("Z", "Y"):
            _shear = (2, 1)
        case _:
            raise ValueError(
                "unsupported shear type with"
                + f" direction = {direction}, deformation_plane = {deformation_plane}"
            )

    return ft.partial(
        _simple_shear_2d,
        direction=_shear[0],
        deformation_plane=_shear[1],
        strain_rate=strain_rate,
    )
