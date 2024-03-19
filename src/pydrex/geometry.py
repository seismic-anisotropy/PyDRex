"""> PyDRex: Functions for geometric coordinate conversions and projections."""

from enum import Enum, unique

import numpy as np
from scipy import linalg as la
from scipy.spatial.transform import Rotation


@unique
class LatticeSystem(Enum):
    """Crystallographic lattice systems supported by postprocessing methods.

    The value of a member is `(a, b)` with `a` and `b` as given in the table below.
    The additional row lists the maximum misorientation angle between two crystallites
    for the given lattice system.

                triclinic  monoclinic  orthorhombic  rhombohedral tetragonal hexagonal
        ------------------------------------------------------------------------------
        a       1          2           2             3            4          6
        b       1          2           4             6            8          12
        θmax    180°       180°        120°          120°         90°        90°

    This is identically Table 1 in [Grimmer (1979)](https://doi.org/10.1016/0036-9748(79)90058-9).

    """

    triclinic = (1, 1)
    monoclinic = (2, 2)
    orthorhombic = (2, 4)
    rhombohedral = (3, 6)
    tetragonal = (4, 8)
    hexagonal = (6, 12)


def to_cartesian(ϕ, θ, r=1):
    """Convert spherical to cartesian coordinates in ℝ³.

    Spherical coordinate convention:
    - ϕ is the longitude (“azimuth”) ∈ [0, 2π)
    - θ is the colatitude (“inclination”) ∈ [0, π)

    By default, a radius of `r = 1` is used for the sphere.
    Returns a tuple containing arrays of x, y, and z values.

    """
    ϕ = np.atleast_1d(ϕ).astype(float)
    θ = np.atleast_1d(θ).astype(float)
    r = np.atleast_1d(r).astype(float)
    return (r * np.sin(θ) * np.cos(ϕ), r * np.sin(θ) * np.sin(ϕ), r * np.cos(θ))


def to_spherical(x, y, z):
    """Convert cartesian coordinates in ℝ³ to spherical coordinates.

    Spherical coordinate convention:
    - ϕ is the longitude (“azimuth”) ∈ [0, 2π)
    - θ is the colatitude (“inclination”) ∈ [0, π)

    Returns a tuple containing arrays of r, ϕ and θ values.

    """
    x = np.atleast_1d(x).astype(float)
    y = np.atleast_1d(y).astype(float)
    z = np.atleast_1d(z).astype(float)
    r = np.sqrt(x**2 + y**2 + z**2)
    return (r, np.arctan2(y, x), np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2)))


def misorientation_angles(q1_array, q2_array):
    """Calculate minimum misorientation angles for collections of rotation quaternions.

    Calculate the smallest angular distance between any quaternions `q1_array[:, i]` and
    `q2_array[:, j]`, where i == j and the first dimensions of `q1_array` and `q2_array`
    are of equal length (the output will also be this long):

        q1_array.shape      q2_array.shape      len(output)
        ---------------------------------------------------
        NxAx4               NxBx4               N

    .. warning::
        This method must be able to allocate a floating point array of shape Nx(A*B)

    Uses ~25% less memory than the same operation with rotation matrices.

    See also:
    - <https://math.stackexchange.com/questions/90081/quaternion-distance>
    - [Huynh (2009)](https://link.springer.com/article/10.1007/s10851-009-0161-2)


    """
    if q1_array.shape[0] != q2_array.shape[0]:
        raise ValueError(
            "the first dimensions of q1_array and q2_array must be of equal length"
            + f" but {q1_array.shape[0]} != {q2_array.shape[0]}"
        )
    angles = np.empty((q1_array.shape[0], q1_array.shape[1] * q2_array.shape[1]))
    k = 0
    for i in range(q1_array.shape[1]):
        for j in range(q2_array.shape[1]):
            angles[:, k] = 2 * np.rad2deg(
                np.arccos(
                    np.abs(
                        np.clip(
                            np.sum(q1_array[:, i] * q2_array[:, j], axis=1),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            k += 1
    return np.min(angles, axis=1)


def symmetry_operations(system: LatticeSystem):
    """Get sequence of symmetry operations for the given `LatticeSystem`.

    Returned transforms are either quaternions (for rotations of the lattice) or 4x4
    matrices which pre-multiply a quaternion to give a reflected variant (reflections
    are *improper* rotations and cannot be represented as quaternions or SciPy rotation
    matrices).

    """
    match system:
        case LatticeSystem.triclinic:  # Triclinic lattice, no additional symmetry.
            return [Rotation.identity().as_quat()]
        case LatticeSystem.monoclinic | LatticeSystem.orthorhombic:
            # Rotation by π around any of the Cartesian axes.
            rotations = [
                Rotation.from_rotvec(np.pi * np.asarray(vector)).as_quat()
                for vector in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            ]
            # Reflections around any of the Cartesian planes.
            # Quaternion space matrix equivalents for the R^3 operations:
            #   np.diag(x) for x in ([1, 1, -1], [1, -1, 1], [-1, 1, 1])
            # https://stackoverflow.com/a/33999726/12519962
            # NOTE: Don't use scipy Rotation, it silently converts to proper rotations.
            # There is maybe <https://github.com/matthew-brett/transforms3d>,
            # but another dependency just for this might not be worth it.
            reflections = [
                np.diag(x) for x in ([1, -1, -1, 1], [1, -1, 1, -1], [1, 1, -1, -1])
            ]
            return [Rotation.identity().as_quat(), *rotations, *reflections]
        case LatticeSystem.rhombohedral:
            # Rotations by n * π/3, n ∈ {1, 2} around any of the Cartesian axes.
            rotations = [
                Rotation.from_rotvec(i * np.pi / 3 * np.asarray(vector)).as_quat()
                for vector in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
                for i in (1, 2)
            ]
            return [Rotation.identity().as_quat(), *rotations]
        case LatticeSystem.tetragonal:
            # Rotations by n * π/2, n ∈ {1, 2, 3} around any of the Cartesian axes.
            rotations = [
                Rotation.from_rotvec(i * np.pi / 2 * np.asarray(vector)).as_quat()
                for vector in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
                for i in (1, 2, 3)
            ]
            return [Rotation.identity().as_quat(), *rotations]
        case LatticeSystem.hexagonal:
            # Rotations by n * π/3, n ∈ {1, 2} around any of the Cartesian axes.
            rotations3 = [
                Rotation.from_rotvec(i * np.pi / 3 * np.asarray(vector)).as_quat()
                for vector in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
                for i in (1, 2)
            ]
            # Rotations by n * π/6, n ∈ {1, 3, 5} around any of the Cartesian axes.
            # The other three π/6 rotations are identical to the π/3 rotations.
            rotations6 = [
                Rotation.from_rotvec(i * np.pi / 6 * np.asarray(vector)).as_quat()
                for vector in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
                for i in (1, 3, 5)
            ]
            return [Rotation.identity().as_quat(), *rotations3, *rotations6]
        case _:
            raise ValueError(f"unsupported lattice system: {system}")


def poles(orientations, ref_axes="xz", hkl=[1, 0, 0]):
    """Extract 3D vectors of crystallographic directions from orientation matrices.

    Expects `orientations` to be an array with shape (N, 3, 3).
    The optional arguments `ref_axes` and `hkl` can be used to change
    the global reference axes and the crystallographic direction respectively.
    The reference axes should be given as a string of two letters,
    e.g. "xz" (default), which specify the second and third axes
    of the global right-handed reference frame. The third letter in the set "xyz"
    determines the first axis. The `ref_axes` will therefore become the
    horizontal and vertical axes of pole figures used to plot the directions.

    """
    _ref_axes = ref_axes.lower()
    upward_axes = (set("xyz") - set(_ref_axes)).pop()
    axes_map = {"x": 0, "y": 1, "z": 2}

    # Get directions in the right-handed frame.
    directions = np.tensordot(orientations.transpose([0, 2, 1]), hkl, axes=(2, 0))
    directions_norm = la.norm(directions, axis=1)
    directions /= directions_norm.reshape(-1, 1)

    # Rotate into the chosen reference frame.
    zvals = directions[:, axes_map[upward_axes]]
    yvals = directions[:, axes_map[_ref_axes[1]]]
    xvals = directions[:, axes_map[_ref_axes[0]]]
    return xvals, yvals, zvals


def lambert_equal_area(xvals, yvals, zvals):
    """Project axial data from a 3D sphere onto a 2D disk.

    Project points from a 3D sphere of radius 1, given in Cartesian coordinates,
    to points on a 2D disk using a Lambert equal area azimuthal projection.
    Returns arrays of the X and Y coordinates in the unit disk.

    This implementation first maps all points onto the same hemisphere,
    and then projects that hemisphere onto the disk.

    """
    xvals = np.atleast_1d(xvals).astype(float)
    yvals = np.atleast_1d(yvals).astype(float)
    zvals = np.atleast_1d(zvals).astype(float)
    # One reference for the equation is Mardia & Jupp 2009 (Directional Statistics),
    # where it appears as eq. 9.1.1 in spherical coordinates,
    #   [sinθcosφ, sinθsinφ, cosθ].

    # First we move all points into the upper hemisphere.
    # See e.g. page 186 of Snyder 1987 (Map Projections— A Working Manual).
    # This is done by taking θ' = π - θ where θ is the colatitude (inclination).
    # Equivalently, in Cartesian coords we just take abs of the z values.
    zvals = abs(zvals)
    # When x and y are both 0, we would have a zero-division.
    # These points are always projected onto [0, 0] (the centre of the disk).
    condition = np.logical_and(np.abs(xvals) < 1e-16, np.abs(yvals) < 1e-16)
    x_masked = np.ma.masked_where(condition, xvals)
    y_masked = np.ma.masked_where(condition, yvals)
    x_masked.fill_value = np.nan
    y_masked.fill_value = np.nan
    # The equations in Mardia & Jupp 2009 and Snyder 1987 both project the hemisphere
    # onto a disk of radius sqrt(2), so we drop the sqrt(2) factor that appears
    # after converting to Cartesian coords to get a radius of 1.
    prefactor = np.sqrt((1 - zvals) / (x_masked**2 + y_masked**2))
    prefactor.fill_value = 0.0
    return prefactor.filled() * xvals, prefactor.filled() * yvals


def shirley_concentric_squaredisk(xvals, yvals):
    """Project points from a square onto a disk using the concentric Shirley method.

    The concentric method of [Shirley & Chiu (1997)](https://doi.org/10.1080/10867651.1997.10487479)
    is optimised to preserve area. See also: <http://marc-b-reynolds.github.io/math/2017/01/08/SquareDisc.html>.

    This can be used to set up uniform grids on a disk, e.g.

    >>> a = [x / 5.0 for x in range(-5, 6)]
    >>> x = [[x] * len(a) for x in a]
    >>> y = [a for _ in a]
    >>> x_flat = [j for i in x for j in i]
    >>> y_flat = [j for i in y for j in i]
    >>> x_disk, y_disk = shirley_concentric_squaredisk(x_flat, y_flat)
    >>> r = x_disk**2 + y_disk**2
    >>> r.reshape((len(a), len(a)))
    array([[1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ],
           [1.  , 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 1.  ],
           [1.  , 0.64, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.64, 1.  ],
           [1.  , 0.64, 0.36, 0.16, 0.16, 0.16, 0.16, 0.16, 0.36, 0.64, 1.  ],
           [1.  , 0.64, 0.36, 0.16, 0.04, 0.04, 0.04, 0.16, 0.36, 0.64, 1.  ],
           [1.  , 0.64, 0.36, 0.16, 0.04, 0.  , 0.04, 0.16, 0.36, 0.64, 1.  ],
           [1.  , 0.64, 0.36, 0.16, 0.04, 0.04, 0.04, 0.16, 0.36, 0.64, 1.  ],
           [1.  , 0.64, 0.36, 0.16, 0.16, 0.16, 0.16, 0.16, 0.36, 0.64, 1.  ],
           [1.  , 0.64, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.64, 1.  ],
           [1.  , 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 1.  ],
           [1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ]])
    >>> from math import atan2
    >>> θ = [atan2(y, x) for y, x in zip(y_disk, x_disk)]
    >>> max(θ)
    3.141592653589793
    >>> min(θ)
    -2.9845130209101467

    """

    def _shirley_concentric_squaredisc_xgty(xvals, yvals):
        ratios = yvals / (xvals + 1e-12)
        return xvals * np.cos(np.pi / 4 * ratios), xvals * np.sin(np.pi / 4 * ratios)

    def _shirley_concentric_squaredisc_xlty(xvals, yvals):
        ratios = xvals / (yvals + 1e-12)
        return yvals * np.sin(np.pi / 4 * ratios), yvals * np.cos(np.pi / 4 * ratios)

    xvals = np.atleast_1d(xvals).astype(float)
    yvals = np.atleast_1d(yvals).astype(float)
    conditions = [
        np.repeat(np.atleast_2d(np.abs(xvals) >= np.abs(yvals)), 2, axis=0).transpose(),
        np.repeat(np.atleast_2d(np.abs(xvals) < np.abs(yvals)), 2, axis=0).transpose(),
    ]
    x_disk, y_disk = np.piecewise(
        np.column_stack([xvals, yvals]),
        conditions,
        [
            lambda xy: np.column_stack(
                _shirley_concentric_squaredisc_xgty(*xy.reshape(-1, 2).transpose())
            ).ravel(),
            lambda xy: np.column_stack(
                _shirley_concentric_squaredisc_xlty(*xy.reshape(-1, 2).transpose())
            ).ravel(),
        ],
    ).transpose()
    return x_disk, y_disk
