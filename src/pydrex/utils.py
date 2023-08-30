"""> PyDRex: Miscellaneous utility methods."""
from datetime import datetime

import numba as nb
import numpy as np


def remove_nans(a):
    """Remove NaN values from array."""
    a = np.asarray(a)
    return a[~np.isnan(a)]


def readable_timestamp(timestamp, tformat="%H:%M:%S"):
    """Convert timestamp in fractional seconds to human readable format."""
    return datetime.fromtimestamp(timestamp).strftime(tformat)


def get_steps(a):
    """Get forward difference of 2D array `a`, with repeated last elements.

    The repeated last elements ensure that output and input arrays have equal shape.

    Examples:

    >>> _get_steps(np.array([1, 2, 3, 4, 5]))
    array([[1, 1, 1, 1, 1]])

    >>> _get_steps(np.array([[1, 2, 3, 4, 5], [1, 3, 6, 9, 10]]))
    array([[1, 1, 1, 1, 1],
           [2, 3, 3, 1, 1]])

    >>> _get_steps(np.array([[1, 2, 3, 4, 5], [1, 3, 6, 9, 10], [1, 0, 0, 0, np.inf]]))
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 2.,  3.,  3.,  1.,  1.],
           [-1.,  0.,  0., inf, nan]])

    """
    a2 = np.atleast_2d(a)
    return np.diff(
        a2, append=np.reshape(a2[:, -1] + (a2[:, -1] - a2[:, -2]), (a2.shape[0], 1))
    )


def angle_fse_simpleshear(strain):
    """Get angle of FSE long axis anticlockwise from the X axis in simple shear."""
    return np.rad2deg(np.arctan(np.sqrt(strain**2 + 1) + strain))


def lag_2d_corner_flow(θ):
    # Predicted grain orientation lag for 2D corner flow, eq. 11 in Kaminski 2002.
    _θ = np.ma.masked_less(θ, 1e-15)
    return (_θ * (_θ**2 + np.cos(_θ) ** 2)) / (
        np.tan(_θ) * (_θ**2 + np.cos(_θ) ** 2 - _θ * np.sin(2 * _θ))
    )


@nb.njit(fastmath=True)
def quat_product(q1, q2):
    """Quaternion product, q1, q2 and output are in scalar-last (x,y,z,w) format."""
    return [
        *q1[-1] * q2[:3] + q2[-1] * q1[:3] + np.cross(q1[:3], q1[:3]),
        q1[-1] * q2[-1] - np.dot(q1[:3], q2[:3]),
    ]


def __run_doctests():
    import doctest

    return doctest.testmod()
