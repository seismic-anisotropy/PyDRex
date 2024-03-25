import configparser
import pathlib

import numba as nb
import numpy as np

from pydrex.utils import add_dim, remove_dim
from pydrex.velocity import simple_shear_2d

config = configparser.ConfigParser()
config.read(pathlib.PurePath(__file__).with_suffix(".ini"))
init = config["initial conditions"]


def init_positions():
    """Return initial particle positions."""
    init_horiz = np.array([float(x) for x in init["INIT_HORIZ"].split()])
    init_vert = float(init["INIT_VERT"])
    return np.array([[x, init_vert] for x in init_horiz])


def cb_interp_position(position_prev, position, t, dt):
    """Return position interpolator for CPO integration from `position_prev` to `position`."""

    @nb.njit(fastmath=True)
    def _interp_position(int_time):
        return position_prev + (int_time - t + dt) / dt * (position - position_prev)

    return _interp_position


def cb_interp_velocity_grad(velocity_grad_prev, velocity_grad, t, dt):
    """Return ∇v interpolator for CPO integration between `velocity_grad_prev` and `velocity_grad`."""

    @nb.njit(fastmath=True)
    def _interp_vel_grad(int_time):
        return velocity_grad_prev + (int_time - t + dt) / dt * (
            velocity_grad - velocity_grad_prev
        )

    return _interp_vel_grad


STRAIN_RATE = float(init["STRAIN_RATE"])
# Initial position of the particle.
INITIAL_POSITIONS = init_positions()
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

_velocity, _velocity_grad = simple_shear_2d("X", "Z", STRAIN_RATE)


def velocity(X, t):
    return remove_dim(_velocity(add_dim(X, 1)), 1)


def velocity_gradient(X, t):
    return remove_dim(_velocity_grad(add_dim(X, 1)), 1)
