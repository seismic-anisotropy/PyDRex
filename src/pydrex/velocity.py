"""> PyDRex: Steady-state solutions of velocity (gradients) for various flows.

For the sake of consistency, all callables returned from methods in this module expect a
3D position vector as input. They also return 3D tensors in all cases. This means they
can be directly used as arguments to e.g. `pydrex.minerals.Mineral.update_orientations`.

"""
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


@nb.njit(fastmath=True)
def _corner_2d(x, horizontal=0, vertical=2, plate_speed=1):
    h = x[horizontal]
    v = x[vertical]
    out = np.zeros(3)
    prefactor = 2 * plate_speed / np.pi
    out[horizontal] = prefactor * (np.arctan2(h, -v) + h * v / (h**2 + v**2))
    out[vertical] = prefactor * v**2 / (h**2 + v**2)
    return out


@nb.njit(fastmath=True)
def _corner_2d_grad(x, horizontal=0, vertical=2, plate_speed=1):
    h = x[horizontal]
    v = x[vertical]
    grad_v = np.zeros((3, 3))
    prefactor = 4 * plate_speed / (np.pi * (h**2 + v**2)**2)
    grad_v[horizontal, horizontal] = -(h**2) * v
    grad_v[horizontal, vertical] = h**3
    grad_v[vertical, horizontal] = -h * v**2
    grad_v[vertical, vertical] = h**2 * v
    return prefactor * grad_v


def simple_shear_2d(direction, deformation_plane, strain_rate):
    """Return simple shear velocity and velocity gradient callables.

    The returned callables have signature f(x) where x is a 3D position vector.

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
    r"""Return velocity and velocity gradient callable for a steady-state 2D Stokes cell.

    The velocity field is defined by:
    $$
    \bm{u} = \cos(π x/2)\sin(π x/2) \bm{\hat{h}} - \sin(π x/2)\cos(π x/2) \bm{\hat{v}}
    $$
    where $\bm{\hat{h}}$ and $\bm{\hat{v}}$ are unit vectors in the chosen horizontal
    and vertical directions, respectively.

    The returned callables have signature f(x) where x is a 3D position vector.

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


def corner_2d(horizontal, vertical, plate_speed):
    r"""Return velocity and velocity gradient callable for a steady-state 2D Stokes cell.

    The velocity field is defined by:
    $$
    \bm{u} = \frac{dr}{dt} \bm{\hat{r}} + r \frac{dθ}{dt} \bm{\hat{θ}}
    = \frac{2 U}{π}(θ\sinθ - \cosθ) ⋅ \bm{\hat{r}} + \frac{2 U}{π}θ\cosθ ⋅ \bm{\hat{θ}}
    $$
    where $θ = 0$ points vertically downwards along the ridge axis
    and $θ = π/2$ points along the surface. $U$ is the half spreading velocity.
    Streamlines for the flow obey:
    $$
    ψ = \frac{2 U r}{π}θ\cosθ
    $$
    and are related to the velocity through:
    $$
    \bm{u} = -\frac{1}{r} ⋅ \frac{dψ}{dθ} ⋅ \bm{\hat{r}} + \frac{dψ}{dr}\bm{\hat{θ}}
    $$
    Conversion to Cartesian ($x,y,z$) coordinates yields:
    $$
    \bm{u} = \frac{2U}{π} \left[
    \tan^{-1}\left(\frac{x}{-z}\right) + \frac{xz}{x^{2} + z^{2}} \right] \bm{\hat{x}} +
    \frac{2U}{π} \frac{z^{2}}{x^{2} + z^{2}} \bm{\hat{z}}
    $$
    where
    \begin{align\*}
    x &= r \sinθ \cr
    z &= -r \cosθ
    \end{align\*}
    and the velocity gradient is:
    $$
    L = \frac{4 U}{π{(x^{2}+z^{2})}^{2}} ⋅
    \begin{bmatrix}
        -x^{2}z & 0 & x^{3} \cr
        0 & 0 & 0 \cr
        -xz^{2} & 0 & x^{2}z
    \end{bmatrix}
    $$
    See also Fig. 5 in [Kaminski & Ribe, 2002](https://doi.org/10.1029/2001GC000222).

    The returned callables have signature f(x) where x is a 3D position vector.

    Args:
    - `horizontal` (one of {"X", "Y", "Z"}) — horizontal direction
    - `vertical` (one of {"X", "Y", "Z"}) — vertical direction
    - `plate_speed` (float) — speed of the “plate” i.e. upper boundary

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
            _corner_2d,
            horizontal=_geometry[0],
            vertical=_geometry[1],
            plate_speed=plate_speed,
        ),
        ft.partial(
            _corner_2d_grad,
            horizontal=_geometry[0],
            vertical=_geometry[1],
            plate_speed=plate_speed,
        ),
    )
