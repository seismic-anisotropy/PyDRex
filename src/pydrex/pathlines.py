"""> PyDRex: Functions for pathline construction."""
import numpy as np
from scipy import integrate as si
from scipy import linalg as la

from pydrex import logger as _log


def get_pathline(
    final_location,
    get_velocity,
    get_velocity_gradient,
    min_coords,
    max_coords,
    max_strain=10,
):
    """Determine the pathline for a particle in a steady state flow.

    The pathline will intersect the given `point` and follow a curve determined by
    the interpolated/prescribed velocity gradient.

    Args:
        `final_location` (NumPy array) — coordinates of the target final location,
                                         which may not be reached if `max_strain` is
                                         reached first
        `get_velocity` (callback) — returns velocity vector at a point
        `get_velocity_gradient` (callback) — returns ∇v (3x3 matrix) at a point
        `min_coords` (iterable) — lower bound coordinate of the domain
        `max_coords` (iterable) — upper bound coordinate of the domain
        `max_strain` (optional) — strain limit at which to terminate the pathline,
                                  useful if the pathline does not exit the domain

    Returns a tuple containing the time points and an interpolant that can be used
    to evaluate the pathline position (see `scipy.integrate.OdeSolution`).

    """

    def _max_strain(
        time, point, get_velocity, get_velocity_gradient, min_coords, max_coords
    ):
        nonlocal event_time, event_time_prev, event_strain_prev, event_strain
        nonlocal event_strain_prev, event_flag
        if event_flag:
            return (
                event_strain if time == event_time else event_strain_prev
            ) - max_strain

        if _is_inside(point, min_coords, max_coords):
            velocity_gradient = get_velocity_gradient(point)
            # Imposed macroscopic strain rate tensor.
            strain_rate = (velocity_gradient + velocity_gradient.transpose()) / 2
            # Strain rate scale (max. eigenvalue of strain rate).
            strain_rate_max = np.abs(la.eigvalsh(strain_rate)).max()
            event_strain_prev = event_strain
            event_strain += abs(time - event_time) * strain_rate_max
            if event_strain >= max_strain:
                event_flag = True
            event_time_prev = event_time
            event_time = time
            return event_strain - max_strain

        return 0

    _max_strain.terminal = True
    event_strain = event_time = 0
    event_strain_prev = event_time_prev = None
    event_flag = False

    # Initial condition is the final position of the crystal.
    # Solve backwards in time until outside the domain or max. strain reached.
    # We don't want to stop at a particular time,
    # so integrate time for 100 Myr, in seconds (forever).
    path = si.solve_ivp(
        _ivp_func,
        [0, -100e6 * 365.25 * 8.64e4],
        final_location,
        method="LSODA",
        # first_step=1e10,
        # max_step=np.inf,
        # max_step=1e6,
        # t_eval=None,
        events=[_max_strain],
        args=(get_velocity, get_velocity_gradient, min_coords, max_coords),
        dense_output=True,
        jac=_ivp_jac,
        atol=1e-8,
        rtol=1e-5,
    )
    _log.info(
        "calculated pathline from %s (t=%e) to %s (t=%e)",
        path.sol(path.t[0]),
        path.t[0],
        path.sol(path.t[-2]),
        path.t[-2],
    )

    # Remove the last timestep, because the position will be outside the domain.
    # The integration only stops AFTER the event is triggered.
    return path.t[:-1], path.sol


def _ivp_func(
    time, point, get_velocity, get_velocity_gradient, min_coords, max_coords
):
    """Internal use only, must have the same signature as `get_pathline`."""
    if _is_inside(point, min_coords, max_coords):
        return get_velocity(point)
    return np.zeros_like(point)


def _ivp_jac(
    time, point, get_velocity, get_velocity_gradient, min_coords, max_coords
):
    """Internal use only, must have the same signature as `_ivp_func`."""
    if _is_inside(point, min_coords, max_coords):
        return get_velocity_gradient(point)
    return np.zeros((np.array(point).size,) * 2)


def _is_inside(point, min_coords, max_coords):
    """Check if the point lies within the numerical domain."""
    assert np.array(point).size == len(min_coords) == len(max_coords)
    if np.any(np.array(point) < min_coords) or np.any(np.array(point) > max_coords):
        return False
    return True
