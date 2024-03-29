"""> PyDRex: Functions for pathline construction."""

import numpy as np
from scipy import integrate as si

from pydrex import logger as _log
from pydrex import utils as _utils


def get_pathline(
    final_location,
    get_velocity,
    get_velocity_gradient,
    min_coords,
    max_coords,
    max_strain,
    regular_steps=None,
    **kwargs,
):
    """Determine the pathline for a particle in a steady state flow.

    The pathline will terminate at the given `final_location` and follow a curve
    determined by the velocity gradient. It works for both 2D (rectangular) and 3D
    (orthopiped¹) domains, so long as the provided callables expect/return arrays of the
    appropriate dimension.

    .. note::
        The pathline is calculated backwards in time (t < 0) from the given endpoint.
        Therefore, the returned position callable should be evaluated at negative times.

    Args:
    - `final_location` (array) — coordinates of the final location
    - `get_velocity` (callable) — returns velocity vector at a point
    - `get_velocity_gradient` (callable) — returns velocity gradient matrix at a point
    - `min_coords` (array) — lower bound coordinates of the box
    - `max_coords` (array) — upper bound coordinates of the box
    - `max_strain` (float) — target strain (given as “tensorial” strain ε) at the final
      location, useful if the pathline never inflows into the domain (the pathline will
      only be traced backwards until a strain of 0 is reached, unless a domain boundary
      is reached first)
    - `regular_steps` (float, optional) — number of time steps to use for regular
      resampling between the start (t << 0) and end (t <= 0) of the pathline
      (if `None`, which is the default, then the timestamps obtained from
      `scipy.integrate.solve_ivp` are returned instead)

    Optional keyword arguments will be passed to `scipy.integrate.solve_ivp`. However,
    some of the arguments to the `solve_ivp` call may not be modified, and a warning
    will be raised if they are provided.

    Returns a tuple containing the time points and an interpolant that can be used
    to evaluate the pathline position (see `scipy.integrate.OdeSolution`).

    ¹An “orthopiped” is a 3D rectangle (called a “box” when we are in a hurry), see
    <https://www.whatistoday.net/2020/04/cuboid-dilemma.html>.

    """

    def _terminate(
        time, point, get_velocity, get_velocity_gradient, min_coords, max_coords
    ):
        # Track “previous” (last seen) timestamp and total strain value.
        nonlocal _time_prev, _strain

        if _is_inside(point, min_coords, max_coords):
            dε = _utils.strain_increment(
                time - _time_prev, get_velocity_gradient(np.nan, point)
            )
            if time > _time_prev:  # Timestamps jump around for SciPy to find the root.
                _strain += dε
            else:  # Subtract strain increment because we are going backwards in time.
                _strain -= dε
            _time_prev = time
            return _strain
        # If we are outside the domain, always terminate.
        return 0

    _terminate.terminal = True
    _strain = max_strain
    _time_prev = 0
    _event_flag = False

    # Illegal keyword args, check the call below. Remove them and warn about it.
    for key in ("events", "jac", "dense_output", "args"):
        try:
            kwargs.pop(key)
        except KeyError:
            continue
        else:
            _log.warning("ignoring illegal keyword argument: %s", key)

    # We don't want to stop at a particular time,
    # so integrate time for 100 Myr, in seconds (“forever”).
    path = si.solve_ivp(
        _ivp_func,
        [0, -100e6 * 365.25 * 8.64e4],
        final_location,
        method=kwargs.pop("method", "LSODA"),
        events=[_terminate],
        args=(get_velocity, get_velocity_gradient, min_coords, max_coords),
        dense_output=True,
        jac=_ivp_jac,
        atol=kwargs.pop("atol", 1e-8),
        rtol=kwargs.pop("rtol", 1e-5),
        **kwargs,
    )
    _log.info(
        "calculated pathline from %s (t = %e) to %s (t = %e)",
        path.sol(path.t[0]),
        path.t[0],
        path.sol(path.t[-1]),
        path.t[-1],
    )

    if regular_steps is None:
        return path.t[::-1], path.sol
    else:
        return np.linspace(path.t[-1], path.t[0], regular_steps + 1), path.sol


def _ivp_func(time, point, get_velocity, get_velocity_gradient, min_coords, max_coords):
    """Internal use only, must have the same signature as `get_pathline`."""
    if _is_inside(point, min_coords, max_coords):
        return get_velocity(np.nan, point)
    return np.zeros_like(point)


def _ivp_jac(time, point, get_velocity, get_velocity_gradient, min_coords, max_coords):
    """Internal use only, must have the same signature as `_ivp_func`."""
    if _is_inside(point, min_coords, max_coords):
        return get_velocity_gradient(np.nan, point)
    return np.zeros((np.array(point).size,) * 2)


def _is_inside(point, min_coords, max_coords):
    """Check if the point lies within the numerical domain."""
    assert np.array(point).size == len(min_coords) == len(max_coords)
    if np.any(np.array(point) < min_coords) or np.any(np.array(point) > max_coords):
        return False
    return True
