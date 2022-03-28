"""PyDRex: Functions for pathline construction."""


def get_pathline(point, interpolators, min_coords, max_coords):
    """Determine the pathline for a crystal aggregate in a steady state flow.

    The pathline will intersect the given `point` and follow a curve determined by
    the velocity gradient interpolators.

    Args:
        `point` (NumPy array) — coordinates of the point
        `interpolators` (dict) — dictionary with velocity gradient interpolators
        `min_coords` (iterable) — lower bound coordinate of the interpolation grid
        `max_coords` (iterable) — upper bound coordinate of the interpolation grid

    Returns a tuple containing the time points and an interpolant that can be used
    to evaluate the pathline position (see `scipy.integrate.OdeSolution`).

    """

    def _max_strain(time, point, interpolators, min_coords, max_coords):
        nonlocal event_time, event_time_prev, event_strain_prev, event_strain, event_strain_prev, event_flag
        # TODO: Refactor, move 10 "max strain" parameter to config?
        if event_flag:
            return (event_strain if time == event_time else event_strain_prev) - 10

        if _is_inside(point, min_coords, max_coords):
            velocity_gradient = _interp.get_velocity_gradient(point, interpolators)
            # Imposed macroscopic strain rate tensor.
            strain_rate = (velocity_gradient + velocity_gradient.transpose()) / 2
            # Strain rate scale (max. eigenvalue of strain rate).
            strain_rate_max = np.abs(la.eigvalsh(strain_rate)).max()
            event_strain_prev = event_strain
            event_strain += abs(time - event_time) * strain_rate_max
            if event_strain >= 10:
                event_flag = True
            event_time_prev = event_time
            event_time = time
            return event_strain - 10

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
        point,
        method="RK45",
        first_step=1e10,
        max_step=np.inf,
        events=[_max_strain],
        args=(interpolators, min_coords, max_coords),
        dense_output=True,
        jac=_ivp_jac,
        atol=1e-8,
        rtol=1e-5,
    )
    return path.t, path.sol


def _ivp_func(time, point, interpolators, min_coords, max_coords):
    """Internal use only, must have the same signature as `get_pathline`."""
    if _is_inside(point, min_coords, max_coords):
        return _interp.get_velocity(point, interpolators)
    return np.zeros_like(point)


def _ivp_jac(time, point, interpolators, min_coords, max_coords):
    """Internal use only, must have the same signature as `_ivp_func`."""
    if _is_inside(point, min_coords, max_coords):
        return _interp.get_velocity_gradient(point, interpolators)
    return np.zeros((point.size,) * 2)


def _is_inside(point, min_coords, max_coords):
    """Check if the point lies within the numerical domain."""
    assert point.size == len(min_coords) == len(max_coords)
    if np.any(point < min_coords) or np.any(point > max_coords):
        return False
    return True
