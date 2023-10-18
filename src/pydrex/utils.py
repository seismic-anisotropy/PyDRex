"""> PyDRex: Miscellaneous utility methods."""
from datetime import datetime
import subprocess
import os
import platform

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def strain_increment(dt, velocity_gradient):
    """Calculate strain increment for a given time increment and velocity gradient.

    Returns “tensorial” strain increment ε, which is equal to 2 × γ where γ is the
    “(engineering) shear strain” increment.

    """
    return (
        np.abs(dt)
        * np.abs(
            np.linalg.eigvalsh((velocity_gradient + velocity_gradient.transpose()) / 2)
        ).max()
    )


def remove_nans(a):
    """Remove NaN values from array."""
    a = np.asarray(a)
    return a[~np.isnan(a)]


def readable_timestamp(timestamp, tformat="%H:%M:%S"):
    """Convert timestamp in fractional seconds to human readable format."""
    return datetime.fromtimestamp(timestamp).strftime(tformat)


def default_ncpus():
    """Get a safe default number of CPUs available for multiprocessing.

    On Linux platforms that support it, the method `os.sched_getaffinity()` is used.
    On Mac OS, the command `sysctl -n hw.ncpu` is used.
    On Windows, the environment variable `NUMBER_OF_PROCESSORS` is queried.
    If any of these fail, a fallback of 1 is used and a warning is logged.

    """
    try:
        match platform.system():
            case "Linux":
                return len(os.sched_getaffinity()) - 1  # May raise AttributeError.
            case "Darwin":
                # May raise CalledProcessError.
                out = subprocess.run(
                    ["sysctl", "-n", "hw.ncpu"], capture_output=True, check=True
                )
                return int(out.stdout.strip()) - 1
            case "Windows":
                return int(os.environ["NUMBER_OF_PROCESSORS"]) - 1
    except AttributeError or CalledProcessError or KeyError:
        return 1


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
    """Get predicted grain orientation lag for 2D corner flow.

    See eq. 11 in [Kaminski & Ribe (2002)](https://doi.org/10.1029/2001GC000222).

    """
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


def redraw_legend(ax):
    """Redraw legend on matplotlib axis with new labels since last `ax.legend()`."""
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.legend()


def __run_doctests():
    import doctest

    return doctest.testmod()
