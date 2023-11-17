"""> PyDRex: Miscellaneous utility methods."""
from datetime import datetime
import subprocess
import os
import platform

from matplotlib.pyplot import Line2D
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
from matplotlib import transforms as mtrans
import numba as nb
import numpy as np

from pydrex import logger as _log


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


def redraw_legend(ax, fig=None, legendax=None, remove_all=True, **kwargs):
    """Redraw legend on matplotlib axis or figure.

    Transparency is removed from legend symbols.
    If `fig` is not None and `remove_all` is True,
    all legends are first removed from the parent figure.
    Optional keyword arguments are passed to `matplotlib.axes.Axes.legend` by default,
    or `matplotlib.figure.Figure.legend` if `fig` is not None.

    If `legendax` is not None, the axis legend will be redrawn using the `legendax` axes
    instead of taking up space in the original axes. This option requires `fig=None`.

    .. warning::
        Note that if `fig` is not `None`, the legend may be cropped from the saved
        figure due to a Matplotlib bug. In this case, it is required to add the
        arguments `bbox_extra_artists=(legend,)` and `bbox_inches="tight"` to `savefig`,
        where `legend` is the object returned by this function. To prevent the legend
        from consuming axes/subplot space, it is further required to add the lines:
        `legend.set_in_layout(False)`, `fig.canvas.draw()`, `legend.set_layout(True)`
        and `fig.set_layout_engine("none")` before saving the figure.

    """
    handler_map={
        PathCollection: HandlerPathCollection(
            update_func=_remove_legend_symbol_transparency
        ),
        Line2D: HandlerLine2D(update_func=_remove_legend_symbol_transparency)
    }
    if fig is None:
        legend = ax.get_legend()
        if legend is not None:
            handles, labels = ax.get_legend_handles_labels()
            legend.remove()
        if legendax is not None:
            legendax.axis("off")
            return legendax.legend(handles, labels, handler_map=handler_map, **kwargs)
        return ax.legend(handler_map=handler_map, **kwargs)
    else:
        if legendax is None:
            _log.warning("ignoring `legendax` argument which requires `fig=None`")
        for legend in fig.legends:
            if legend is not None:
                legend.remove()
        if remove_all:
            for ax in fig.axes:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
        return fig.legend(handler_map=handler_map, **kwargs)


def add_subplot_labels(axs, labelmap=None, loc="left", fontsize="medium", **kwargs):
    """Add subplot labels to axes mosaic, using `ax.title()`.

    Use `labelmap` to specify a dictionary that maps keys in `axs` to subplot labels.
    If `labelmap` is None, the keys in `axs` will be used as the labels by default.

    Any axes in `axs` corresponding to the special key `"legend"` are skipped.

    """
    for txt, ax in axs.items():
        if txt.lower() == "legend":
            continue
        if labelmap is not None:
            _txt = labelmap[txt]
        else:
            _txt = txt
        ax.set_title(_txt, loc=loc, fontsize=fontsize, **kwargs)


def _remove_legend_symbol_transparency(handle, orig):
    """Remove transparency from symbols used in a Matplotlib legend."""
    # https://stackoverflow.com/a/59629242/12519962
    handle.update_from(orig)
    handle.set_alpha(1)


def __run_doctests():
    import doctest

    return doctest.testmod()
