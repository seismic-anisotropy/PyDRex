"""> PyDRex: Miscellaneous utility methods."""

import os
import platform
import subprocess

import numba as nb
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerLine2D, HandlerPathCollection
from matplotlib.pyplot import Line2D


@nb.njit(fastmath=True)
def strain_increment(dt, velocity_gradient):
    """Calculate strain increment for a given time increment and velocity gradient.

    Returns “tensorial” strain increment ε, which is equal to γ/2 where γ is the
    “(engineering) shear strain” increment.

    """
    return (
        np.abs(dt)
        * np.abs(
            np.linalg.eigvalsh((velocity_gradient + velocity_gradient.transpose()) / 2)
        ).max()
    )


@nb.njit
def apply_gbs(orientations, fractions, gbs_threshold, orientations_prev, n_grains):
    """Apply grain boundary sliding for small grains."""
    mask = fractions < (gbs_threshold / n_grains)
    # _log.debug(
    #     "grain boundary sliding activity (volume percentage): %s",
    #     len(np.nonzero(mask)) / len(fractions),
    # )
    # No rotation: carry over previous orientations.
    orientations[mask, :, :] = orientations_prev[mask, :, :]
    fractions[mask] = gbs_threshold / n_grains
    fractions /= fractions.sum()
    # _log.debug(
    #     "grain volume fractions: median=%e, min=%e, max=%e, sum=%e",
    #     np.median(fractions),
    #     np.min(fractions),
    #     np.max(fractions),
    #     np.sum(fractions),
    # )
    return orientations, fractions


@nb.njit
def extract_vars(y, n_grains):
    """Extract deformation gradient, orientation matrices and grain sizes from y."""
    deformation_gradient = y[:9].reshape((3, 3))
    orientations = y[9 : n_grains * 9 + 9].reshape((n_grains, 3, 3)).clip(-1, 1)
    fractions = y[n_grains * 9 + 9 : n_grains * 10 + 9].clip(0, None)
    fractions /= fractions.sum()
    return deformation_gradient, orientations, fractions


def remove_nans(a):
    """Remove NaN values from array."""
    a = np.asarray(a)
    return a[~np.isnan(a)]


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
                return len(os.sched_getaffinity(0)) - 1  # May raise AttributeError.
            case "Darwin":
                # May raise CalledProcessError.
                out = subprocess.run(
                    ["sysctl", "-n", "hw.ncpu"], capture_output=True, check=True
                )
                return int(out.stdout.strip()) - 1
            case "Windows":
                return int(os.environ["NUMBER_OF_PROCESSORS"]) - 1
    except AttributeError or subprocess.CalledProcessError or KeyError:
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


def redraw_legend(ax, fig=None, remove_all=True, **kwargs):
    """Redraw legend on matplotlib axis or figure.

    Transparency is removed from legend symbols.
    If `fig` is not None and `remove_all` is True,
    all legends are first removed from the parent figure.
    Optional keyword arguments are passed to `matplotlib.axes.Axes.legend` by default,
    or `matplotlib.figure.Figure.legend` if `fig` is not None.

    .. warning::
        Note that if `fig` is not `None`, the legend may be cropped from the saved
        figure due to a Matplotlib bug. In this case, it is required to add the
        arguments `bbox_extra_artists=(legend,)` and `bbox_inches="tight"` to `savefig`,
        where `legend` is the object returned by this function. To prevent the legend
        from consuming axes/subplot space, it is further required to add the lines:
        `legend.set_in_layout(False)`, `fig.canvas.draw()`, `legend.set_layout(True)`
        and `fig.set_layout_engine("none")` before saving the figure.

    """
    handler_map = {
        PathCollection: HandlerPathCollection(
            update_func=_remove_legend_symbol_transparency
        ),
        Line2D: HandlerLine2D(update_func=_remove_legend_symbol_transparency),
    }
    if fig is None:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        return ax.legend(handler_map=handler_map, **kwargs)
    else:
        for legend in fig.legends:
            if legend is not None:
                legend.remove()
        if remove_all:
            for ax in fig.axes:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
        return fig.legend(handler_map=handler_map, **kwargs)


def _remove_legend_symbol_transparency(handle, orig):
    """Remove transparency from symbols used in a Matplotlib legend."""
    # https://stackoverflow.com/a/59629242/12519962
    handle.update_from(orig)
    handle.set_alpha(1)


def __run_doctests():
    import doctest

    return doctest.testmod()
