import configparser
import pathlib

import numpy as np
import numba as nb

config = configparser.ConfigParser()
config.read(pathlib.PurePath(__file__).with_suffix(".ini"))
init = config["initial conditions"]


def gen_init_positions():
    """Generate initial particle positions."""
    init_horiz = np.array([float(x) for x in init["INIT_HORIZ"].split()])
    init_depth = float(init["INIT_DEPTH"])
    return np.array([[x, -init_depth] for x in init_horiz])


def gen_interp_position(position_prev, position, t, dt):
    """Generate position interpolator for CPO integration from `position_prev` to `position`."""
    @nb.njit(fastmath=True)
    def _interp_position(int_time):
        return position_prev + (int_time - t + dt) / dt * (position - position_prev)
    return _interp_position


def gen_interp_velocity_grad(velocity_grad_prev, velocity_grad, t, dt):
    """Generate ∇v interpolator for CPO integration between `velocity_grad_prev` and `velocity_grad`."""
    @nb.njit(fastmath=True)
    def _interp_vel_grad(int_time):
        return velocity_grad_prev + (int_time - t + dt) / dt * (
            velocity_grad - velocity_grad_prev
        )
    return _interp_vel_grad


# Convert cm/yr to m/s.
PLATE_SPEED = float(init["PLATE_SPEED"]) / (100.0 * 365.0 * 86400.0)
# Initial position of the particle.
INITIAL_POSITIONS = gen_init_positions()
# Recrystallisation parameter configuration.
PARAMS = {
    "stress_exponent": float(init["STRESS_EXPONENT"]),
    "deformation_exponent": float(init["DEFORMATION_EXPONENT"]),
    "gbm_mobility": float(init["GBM_MOBILITY"]),
    "nucleation_efficiency": float(init["NUCLEATION_EFFICIENCY"]),
    "gbs_threshold": float(init["GBS_THRESHOLD"]),
    "minerals": ("olivine",),
    "olivine_fraction": 1,
}

@nb.njit(fastmath=True)
def velocity(coords, time):
    """Return prescribed velocity for 2D corner flow."""
    if coords[0] == coords[1] == 0:
        return np.array([PLATE_SPEED, 0])

    return (2.0 * PLATE_SPEED / np.pi) * np.array(
        [
            np.arctan2(coords[0], -coords[1])
            + coords[0] * coords[1] / (coords[0] ** 2 + coords[1] ** 2),
            coords[1] ** 2 / (coords[0] ** 2 + coords[1] ** 2),
        ]
    )


@nb.njit(fastmath=True)
def velocity_gradient(coords, time):
    """Return prescribed velocity for 2D corner flow."""
    if coords[0] == coords[1] == 0:
        return np.nan * np.ones((2, 2))

    return (
        4.0 * PLATE_SPEED / (np.pi * (coords[0] ** 2 + coords[1] ** 2) ** 2)
    ) * np.array(
        [
            [-(coords[0] ** 2) * coords[1], coords[0] ** 3],
            [-coords[0] * coords[1] ** 2, coords[0] ** 2 * coords[1]],
        ]
    )
