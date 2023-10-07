"""> PyDRex: Steady-state solutions of velocity (gradients) for various flows."""
import functools as ft

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def _simple_shear_2d_grad(x, direction=1, deformation_plane=0, strain_rate=1):
    grad_v = np.zeros((3, 3))
    grad_v[direction, deformation_plane] = 2 * strain_rate
    return grad_v


@nb.njit(fastmath=True)
def _simple_shear_2d(x, direction=1, deformation_plane=0, strain_rate=1):
    v = np.zeros(3)
    v[direction] = x[deformation_plane] * strain_rate
    return v


@nb.njit(fastmath=True)
def _cell_2d_grad(x, horizontal=0, vertical=2, velocity_edge=1):
    cos_πx_div2 = np.cos(0.5 * np.pi * x[horizontal])
    cos_πz_div2 = np.cos(0.5 * np.pi * x[vertical])
    sin_πx_div2 = np.sin(0.5 * np.pi * x[horizontal])
    sin_πz_div2 = np.sin(0.5 * np.pi * x[vertical])
    grad_v = np.zeros((3, 3))
    grad_v[horizontal, horizontal] = -0.5 * np.pi * sin_πz_div2 * sin_πx_div2
    grad_v[horizontal, vertical] = 0.5 * np.pi * cos_πz_div2 * cos_πx_div2
    grad_v[vertical, vertical] = -0.5 * np.pi * cos_πz_div2 * cos_πx_div2
    grad_v[vertical, horizontal] = 0.5 * np.pi * sin_πz_div2 * sin_πx_div2
    return grad_v


@nb.njit(fastmath=True)
def _cell_2d(x, horizontal=0, vertical=2, velocity_edge=1):
    cos_πx_div2 = np.cos(0.5 * np.pi * x[horizontal])
    cos_πz_div2 = np.cos(0.5 * np.pi * x[vertical])
    sin_πx_div2 = np.sin(0.5 * np.pi * x[horizontal])
    sin_πz_div2 = np.sin(0.5 * np.pi * x[vertical])
    v = np.zeros(3)
    v[horizontal] = cos_πx_div2 * sin_πz_div2
    v[vertical] = -sin_πx_div2 * cos_πz_div2
    return v


def simple_shear_2d(direction, deformation_plane, strain_rate):
    """Return simple shear velocity and velocity gradient callables.

    The returned callables have signature f(x) where x is a position vector.

    Args:
    - `direction` (one of {"X", "Y", "Z"}) — velocity vector direction
    - `deformation_plane` (one of {"X", "Y", "Z"}) — direction of velocity gradient
    - `strain_rate` (float) — 1/2 × strength of velocity gradient

    """
    shear = (direction, deformation_plane)
    match shear:
        case ("X", "Y"):
            _shear = (0, 1)
        case ("X", "Z"):
            _shear = (0, 2)
        case ("Y", "X"):
            _shear = (1, 0)
        case ("Y", "Z"):
            _shear = (1, 2)
        case ("Z", "X"):
            _shear = (2, 0)
        case ("Z", "Y"):
            _shear = (2, 1)
        case _:
            raise ValueError(
                "unsupported shear type with"
                + f" direction = {direction}, deformation_plane = {deformation_plane}"
            )

    return (
        ft.partial(
            _simple_shear_2d,
            direction=_shear[0],
            deformation_plane=_shear[1],
            strain_rate=strain_rate,
        ),
        ft.partial(
            _simple_shear_2d_grad,
            direction=_shear[0],
            deformation_plane=_shear[1],
            strain_rate=strain_rate,
        ),
    )


def cell_2d(horizontal, vertical, velocity_edge):
    """Return velocity and velocity gradient callable for a steady-state 2D Stokes cell.

    The returned callables have signature f(x) where x is a position vector.

    Args:
    - `horizontal` (one of {"X", "Y", "Z"}) — horizontal direction
    - `vertical` (one of {"X", "Y", "Z"}) — vertical direction
    - `velocity_edge` (float) — velocity magnitude at the center of the cell edge

    """
    geometry = (horizontal, vertical)
    match geometry:
        case ("X", "Y"):
            _geometry = (0, 1)
        case ("X", "Z"):
            _geometry = (0, 2)
        case ("Y", "X"):
            _geometry = (1, 0)
        case ("Y", "Z"):
            _geometry = (1, 2)
        case ("Z", "X"):
            _geometry = (2, 0)
        case ("Z", "Y"):
            _geometry = (2, 1)
        case _:
            raise ValueError(
                "unsupported convection cell geometry with"
                + f" horizontal = {horizontal}, vertical = {vertical}"
            )

    return (
        ft.partial(
            _cell_2d,
            horizontal=_geometry[0],
            vertical=_geometry[1],
            velocity_edge=velocity_edge,
        ),
        ft.partial(
            _cell_2d_grad,
            horizontal=_geometry[0],
            vertical=_geometry[1],
            velocity_edge=velocity_edge,
        ),
    )
