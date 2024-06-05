"""> PyDRex: Miscellaneous utility methods."""

import os
import platform
import subprocess

import dill
from functools import wraps
import numba as nb
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerLine2D, HandlerPathCollection
from matplotlib.pyplot import Line2D
from matplotlib.transforms import ScaledTranslation

from pydrex import logger as _log


def import_proc_pool():
    """Import either `ray.util.multiprocessing.Pool` or `multiprocessing.Pool`.

    Import a process `Pool` object either from Ray of from Python's stdlib.
    Both offer the same API, the Ray implementation will be preferred if available.
    Using the `Pool` provided by Ray allows for distributed memory multiprocessing.

    Returns a tuple containing the `Pool` object and a boolean flag which is `True` if
    Ray is available.

    """
    try:
        from ray.util.multiprocessing import Pool

        has_ray = True
    except ImportError:
        from multiprocessing import Pool

        has_ray = False
    return Pool, has_ray


class SerializedCallable:
    """A serialized version of the callable f.

    Serialization is performed using the dill library. The object is safe to pass into
    `multiprocessing.Pool.map` and its alternatives.

    .. note:: To serialize a lexical closure (i.e. a function defined inside a
        function), use the `serializable` decorator.

    """
    def __init__(self, f):
        self._f = dill.dumps(f, protocol=5, byref=True)

    def __call__(self, *args, **kwargs):
        return dill.loads(self._f)(*args, **kwargs)


def serializable(f):
    """Make wrapped function serializable."""
    return SerializedCallable(f)


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


def remove_dim(a, dim):
    """Remove all values corresponding to dimension `dim` from an array.

    Note that a `dim` of 0 refers to the “x” values.

    Examples:

    >>> a = [1, 2, 3]
    >>> remove_dim(a, 0)
    array([2, 3])
    >>> remove_dim(a, 1)
    array([1, 3])
    >>> remove_dim(a, 2)
    array([1, 2])

    >>> a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> remove_dim(a, 0)
    array([[5, 6],
           [8, 9]])
    >>> remove_dim(a, 1)
    array([[1, 3],
           [7, 9]])
    >>> remove_dim(a, 2)
    array([[1, 2],
           [4, 5]])

    """
    _a = np.asarray(a)
    for i, _ in enumerate(_a.shape):
        _a = np.delete(_a, [dim], axis=i)
    return _a


def add_dim(a, dim, val=0):
    """Add entries of `val` corresponding to dimension `dim` to an array.

    Note that a `dim` of 0 refers to the “x” values.

    Examples:

    >>> a = [1, 2]
    >>> add_dim(a, 0)
    array([0, 1, 2])
    >>> add_dim(a, 1)
    array([1, 0, 2])
    >>> add_dim(a, 2)
    array([1, 2, 0])

    >>> add_dim([1.0, 2.0], 2)
    array([1., 2., 0.])

    >>> a = [[1, 2], [3, 4]]
    >>> add_dim(a, 0)
    array([[0, 0, 0],
           [0, 1, 2],
           [0, 3, 4]])
    >>> add_dim(a, 1)
    array([[1, 0, 2],
           [0, 0, 0],
           [3, 0, 4]])
    >>> add_dim(a, 2)
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 0]])

    """
    _a = np.asarray(a)
    for i, _ in enumerate(_a.shape):
        _a = np.insert(_a, [dim], 0, axis=i)
    return _a


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


def diff_like(a):
    """Get forward difference of 2D array `a`, with repeated last elements.

    The repeated last elements ensure that output and input arrays have equal shape.

    Examples:

    >>> diff_like(np.array([1, 2, 3, 4, 5]))
    array([[1, 1, 1, 1, 1]])

    >>> diff_like(np.array([[1, 2, 3, 4, 5], [1, 3, 6, 9, 10]]))
    array([[1, 1, 1, 1, 1],
           [2, 3, 3, 1, 1]])

    >>> diff_like(np.array([[1, 2, 3, 4, 5], [1, 3, 6, 9, 10], [1, 0, 0, 0, np.inf]]))
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
    handler_map = {
        PathCollection: HandlerPathCollection(
            update_func=_remove_legend_symbol_transparency
        ),
        Line2D: HandlerLine2D(update_func=_remove_legend_symbol_transparency),
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
        if legendax is not None:
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


def add_subplot_labels(
    mosaic, labelmap=None, loc="left", fontsize="medium", internal=False, **kwargs
):
    """Add subplot labels to axes mosaic.

    Use `labelmap` to specify a dictionary that maps keys in `mosaic` to subplot labels.
    If `labelmap` is None, the keys in `axs` will be used as the labels by default.

    If `internal` is `False` (default), the axes titles will be used.
    Otherwise, internal labels will be drawn with `ax.text`,
    in which case `loc` must be a tuple of floats.

    Any axes in `axs` corresponding to the special key `legend` are skipped.

    """
    for txt, ax in mosaic.items():
        if txt.lower() == "legend":
            continue
        _txt = labelmap[txt] if labelmap is not None else txt
        if internal:
            trans = ScaledTranslation(10 / 72, -5 / 72, ax.figure.dpi_scale_trans)
            if isinstance(loc, str):
                raise ValueError(
                    "'loc' argument must be a sequence of float when 'internal' is 'True'"
                )
            ax.text(
                *loc,
                _txt,
                transform=ax.transAxes + trans,
                fontsize=fontsize,
                bbox={
                    "facecolor": (1.0, 1.0, 1.0, 0.3),
                    "edgecolor": "none",
                    "pad": 3.0,
                },
            )
        else:
            ax.set_title(_txt, loc=loc, fontsize=fontsize, **kwargs)


def _remove_legend_symbol_transparency(handle, orig):
    """Remove transparency from symbols used in a Matplotlib legend."""
    # https://stackoverflow.com/a/59629242/12519962
    handle.update_from(orig)
    handle.set_alpha(1)
