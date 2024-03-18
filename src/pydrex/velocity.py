"""> PyDRex: Steady-state solutions of velocity (gradients) for various flows.

For the sake of consistency, all callables returned from methods in this module expect a
3D position vector as input. They also return 3D tensors in all cases. This means they
can be directly used as arguments to e.g. `pydrex.minerals.Mineral.update_orientations`.

"""

import functools as ft

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def _simple_shear_2d_grad(x, direction, deformation_plane, strain_rate):
    grad_v = np.zeros((3, 3))
    grad_v[direction, deformation_plane] = 2 * strain_rate
    return grad_v


@nb.njit(fastmath=True)
def _simple_shear_2d(x, direction, deformation_plane, strain_rate):
    v = np.zeros(3)
    v[direction] = x[deformation_plane] * strain_rate
    return v


@nb.njit(fastmath=True)
def _cell_2d_grad(x, horizontal, vertical, velocity_edge, edge_length):
    _lim = edge_length / 2
    if np.abs(x[horizontal]) > _lim or np.abs(x[vertical]) > _lim:
        # NOTE: At the moment, this prints type info rather than values. Probably a
        # numba bug, version 0.59 shouldn't require these to be compile-time constants.
        raise ValueError(
            f"position ({x[horizontal]}, {x[vertical]}) is outside domain with xᵢ ∈ [-{_lim}, {_lim}]"
        )
    cos_πx_divd = np.cos(np.pi * x[horizontal] / edge_length)
    cos_πz_divd = np.cos(np.pi * x[vertical] / edge_length)
    sin_πx_divd = np.sin(np.pi * x[horizontal] / edge_length)
    sin_πz_divd = np.sin(np.pi * x[vertical] / edge_length)
    grad_v = np.zeros((3, 3))
    grad_v[horizontal, horizontal] = -np.pi / edge_length * sin_πz_divd * sin_πx_divd
    grad_v[horizontal, vertical] = np.pi / edge_length * cos_πz_divd * cos_πx_divd
    grad_v[vertical, vertical] = -np.pi / edge_length * cos_πz_divd * cos_πx_divd
    grad_v[vertical, horizontal] = np.pi / edge_length * sin_πz_divd * sin_πx_divd
    return velocity_edge * grad_v


@nb.njit(fastmath=True)
def _cell_2d(x, horizontal, vertical, velocity_edge, edge_length):
    _lim = edge_length / 2
    if np.abs(x[horizontal]) > _lim or np.abs(x[vertical]) > _lim:
        # NOTE: At the moment, this prints type info rather than values. Probably a
        # numba bug, version 0.59 shouldn't require these to be compile-time constants.
        raise ValueError(
            f"position ({x[horizontal]}, {x[vertical]}) is outside domain with xᵢ ∈ [-{_lim}, {_lim}]"
        )
    cos_πx_divd = np.cos(np.pi * x[horizontal] / edge_length)
    cos_πz_divd = np.cos(np.pi * x[vertical] / edge_length)
    sin_πx_divd = np.sin(np.pi * x[horizontal] / edge_length)
    sin_πz_divd = np.sin(np.pi * x[vertical] / edge_length)
    v = np.zeros(3)
    v[horizontal] = velocity_edge * cos_πx_divd * sin_πz_divd
    v[vertical] = -velocity_edge * sin_πx_divd * cos_πz_divd
    return v


@nb.njit(fastmath=True)
def _corner_2d(x, horizontal, vertical, plate_speed):
    h = x[horizontal]
    v = x[vertical]
    if np.abs(h) < 1e-15 and np.abs(v) < 1e-15:
        return np.full(3, np.nan)
    out = np.zeros(3)
    prefactor = 2 * plate_speed / np.pi
    out[horizontal] = prefactor * (np.arctan2(h, -v) + h * v / (h**2 + v**2))
    out[vertical] = prefactor * v**2 / (h**2 + v**2)
    return out


@nb.njit(fastmath=True)
def _corner_2d_grad(x, horizontal, vertical, plate_speed):
    h = x[horizontal]
    v = x[vertical]
    if np.abs(h) < 1e-15 and np.abs(v) < 1e-15:
        return np.full((3, 3), np.nan)
    grad_v = np.zeros((3, 3))
    prefactor = 4 * plate_speed / (np.pi * (h**2 + v**2) ** 2)
    grad_v[horizontal, horizontal] = -(h**2) * v
    grad_v[horizontal, vertical] = h**3
    grad_v[vertical, horizontal] = -h * v**2
    grad_v[vertical, vertical] = h**2 * v
    return prefactor * grad_v


def _to_indices(horizontal, vertical):
    geometry = (horizontal.upper(), vertical.upper())
    match geometry:
        case ("X", "Y"):
            indices = (0, 1)
        case ("X", "Z"):
            indices = (0, 2)
        case ("Y", "X"):
            indices = (1, 0)
        case ("Y", "Z"):
            indices = (1, 2)
        case ("Z", "X"):
            indices = (2, 0)
        case ("Z", "Y"):
            indices = (2, 1)
        case _:
            raise ValueError
    return indices


def simple_shear_2d(direction, deformation_plane, strain_rate):
    """Return simple shear velocity and velocity gradient callables.

    The returned callables have signature f(x) where x is a 3D position vector.

    Args:
    - `direction` (one of {"X", "Y", "Z"}) — velocity vector direction
    - `deformation_plane` (one of {"X", "Y", "Z"}) — direction of velocity gradient
    - `strain_rate` (float) — 1/2 × strength of velocity gradient (i.e. magnitude of the
      velocity at a unit distance from the shear plane)

    .. note::
        Input arrays to the returned callables must have homogeneous element types.
        Arrays with e.g. both floating point and integer values are not supported.

    Examples:

    >>> u, L = simple_shear_2d("X", "Z", 1e-4)
    >>> u([0, 0, 0])
    array([0.e+00, 0.e+00, 0.e+00])
    >>> u([0, 0, 1])
    array([1.e-04, 0.e+00, 0.e+00])
    >>> u([0.0, 0.0, 2.0])
    array([2.e-04, 0.e+00, 0.e+00])
    >>> L([0, 0, 0])
    array([[0.e+00, 0.e+00, 2.e-04],
        [0.e+00, 0.e+00, 0.e+00],
        [0.e+00, 0.e+00, 0.e+00]])
    >>> L([0.0, 0.0, 1.0])
    array([[0.e+00, 0.e+00, 2.e-04],
        [0.e+00, 0.e+00, 0.e+00],
        [0.e+00, 0.e+00, 0.e+00]])

    """
    try:
        indices = _to_indices(direction, deformation_plane)
    except ValueError:
        raise ValueError(
            "unsupported shear type with"
            + f" direction = {direction}, deformation_plane = {deformation_plane}"
        )

    return (
        ft.partial(
            _simple_shear_2d,
            direction=indices[0],
            deformation_plane=indices[1],
            strain_rate=strain_rate,
        ),
        ft.partial(
            _simple_shear_2d_grad,
            direction=indices[0],
            deformation_plane=indices[1],
            strain_rate=strain_rate,
        ),
    )


def cell_2d(horizontal, vertical, velocity_edge, edge_length=2):
    r"""Get velocity and velocity gradient callables for a steady-state 2D Stokes cell.

    The cell is centered at (0,0) and the velocity field is defined by:
    $$
    \bm{u} = U\cos(π x/d)\sin(π z/d) \bm{\hat{h}} - U\sin(π x/d)\cos(π z/d) \bm{\hat{v}}
    $$
    where $\bm{\hat{h}}$ and $\bm{\hat{v}}$ are unit vectors in the chosen horizontal
    and vertical directions, respectively. The velocity at the cell edge has a magnitude
    of $U$ and $d$ is the length of a cell edge.

    The returned callables have signature f(x) where x is a 3D position vector.

    Args:
    - `horizontal` (one of {"X", "Y", "Z"}) — horizontal direction
    - `vertical` (one of {"X", "Y", "Z"}) — vertical direction
    - `velocity_edge` (float) — velocity magnitude at the center of the cell edge
    - `edge_length` (float, optional) — the edge length of the cell (= 2 by default)

    Examples:

    >>> u, L = cell_2d("X", "Z", 1)
    >>> u([0, 0, 0])
    array([0.e+00, 0.e+00, -0.e+00])
    >>> u([0, 0, 1])
    array([1.e+00, 0.e+00, -0.e+00])
    >>> u([0, 1, 0])  # Y-value is not used.
    array([0.e+00, 0.e+00, -0.e+00])
    >>> u([0, 0, -1])
    array([-1.e+00, 0.e+00, -0.e+00])
    >>> u([1, 0, 0])
    array([0.e+00, 0.e+00, -1.e+00])
    >>> u([-0.5, 0.0, 0.0])
    array([0.e+00, 0.e+00, 7.071067811865475e-01])
    >>> L([0, 0, 0])
    array([[-0.e+00, 0.e+00, 1.5707963267948966e+00],
        [0.e+00, 0.e+00, 0.e+00],
        [0.e+00, 0.e+00, -1.5707963267948966e+00]])
    >>> L([1, 0, 0])
    array([[-0.e+00, 0.e+00, 9.618353468608949e-17],
        [0.e+00, 0.e+00, 0.e+00],
        [0.e+00, 0.e+00, -9.618353468608949e-17]])
    >>> L([0, 0, 0]) == L([0, 1, 0])  # Y-value is not used.
    array([[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]])
    >>> L([1, 0, 0]) == L([0, 0, 1])
    array([[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]])
    >>> L([1, 0, 0]) == L([-1, 0, 0])
    array([[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]])
    >>> L([1, 0, 0]) == L([0, 0, -1])
    array([[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]])
    >>> L([0.5, 0.0, 0.5])
    array([[-7.853981633974481e-01, 0.e+00, 7.853981633974485e-01],
        [0.e+00, 0.e+00, 0.e+00],
        [7.853981633974481e-01, 0.e+00, -7.853981633974485e-01]])

    >>> u, L = cell_2d("X", "Z", 6.3e-10, 1e5)
    >>> u([0, 0, 0])
    array([0.e+00, 0.e+00, -0.e+00])
    >>> u([0.0, 0.0, -5e4])
    array([-6.3e-10, 0.e+00, -0.e+00])
    >>> u([2e2, 0e0, 0e0])
    array([0.e+00, 0.e+00, -3.958380698302139e-12])

    """
    if edge_length < 0:
        raise ValueError(f"edge length of 2D cell must be positive, not {edge_length}")
    try:
        indices = _to_indices(horizontal, vertical)
    except ValueError:
        raise ValueError(
            "unsupported convection cell geometry with"
            + f" horizontal = {horizontal}, vertical = {vertical}"
        )

    return (
        ft.partial(
            _cell_2d,
            horizontal=indices[0],
            vertical=indices[1],
            velocity_edge=velocity_edge,
            edge_length=edge_length,
        ),
        ft.partial(
            _cell_2d_grad,
            horizontal=indices[0],
            vertical=indices[1],
            velocity_edge=velocity_edge,
            edge_length=edge_length,
        ),
    )


def corner_2d(horizontal, vertical, plate_speed):
    r"""Get velocity and velocity gradient callables for a steady-state 2D corner flow.

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
    try:
        indices = _to_indices(horizontal, vertical)
    except ValueError:
        raise ValueError(
            "unsupported convection cell geometry with"
            + f" horizontal = {horizontal}, vertical = {vertical}"
        )

    return (
        ft.partial(
            _corner_2d,
            horizontal=indices[0],
            vertical=indices[1],
            plate_speed=plate_speed,
        ),
        ft.partial(
            _corner_2d_grad,
            horizontal=indices[0],
            vertical=indices[1],
            plate_speed=plate_speed,
        ),
    )


def __run_doctests():
    import doctest

    return doctest.testmod()
