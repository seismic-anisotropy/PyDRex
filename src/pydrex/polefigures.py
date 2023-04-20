"""PyDRex: Functions for creating crystallographic pole figures.

This module exists because pole figures are more complicated than they seem.
FOr now, only limited types of pole figures are properly supported.
Eventually, inverse pole figures or related helper functions may be added here.

The high level visualisation functions can be found in `pydrex.visualisation`.

"""
import numpy as np
from scipy import linalg as la


def poles(orientations, ref_axes="xz", hkl=[1, 0, 0]):
    """Calculate stereographic poles from 3D orientation matrices.

    Expects `orientations` to be an array with shape (N, 3, 3).
    The optional arguments `ref_axes` and `hkl` can be used to specify
    the stereograph axes and the crystallographic axis respectively.
    The stereograph axes should be given as a string of two letters,
    e.g. "xz" (default), and the third letter in the set "xyz" is used
    as the 'northward' pointing axis for the Lambert equal area projection.

    See also: `lambert_equal_area`.

    """
    upward_axes = next((set("xyz") - set(ref_axes)).__iter__())
    axes_map = {"x": 0, "y": 1, "z": 2}
    directions = np.tensordot(orientations.transpose([0, 2, 1]), hkl, axes=(2, 0))
    directions_norm = la.norm(directions, axis=1)
    directions[:, 0] /= directions_norm
    directions[:, 1] /= directions_norm
    directions[:, 2] /= directions_norm

    zvals = directions[:, axes_map[upward_axes]]
    yvals = directions[:, axes_map[ref_axes[1]]]
    xvals = directions[:, axes_map[ref_axes[0]]]
    return xvals, yvals, zvals


def lambert_equal_area(xvals, yvals, zvals):
    """Project axial data from a 3D sphere onto a 2D disk.

    Project points from a 3D sphere, given in Cartesian coordinates,
    to points on a 2D disk using a Lambert equal area azimuthal projection.
    Returns arrays of the X and Y coordinates in the unit disk.

    This implementation first maps all points onto the same hemisphere,
    and then projects that hemisphere onto the disk.

    """
    # One reference for the equation is Mardia & Jupp 2009 (Directional Statistics),
    # where it appears as eq. 9.1.1 in spherical coordinates,
    #   [sinθcosφ, sinθsinφ, cosθ].
    # They project onto a disk of radius 2, but this is not necessary
    # if we are only projecting poionts from one hemisphere.

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
    assert np.any(xvals > 0)
    assert np.any(yvals > 0)
    assert np.any(xvals != yvals)
    return prefactor.filled() * xvals, prefactor.filled() * yvals


def shirley_concentric_squaredisk(xvals, yvals):
    """Project points from a square onto a disk using the concentric Shirley method.

    The concentric method of Shirley & Chiu 1997 is optimised to preserve area.
    See also: <http://marc-b-reynolds.github.io/math/2017/01/08/SquareDisc.html>.

    """
    def _shirley_concentric_squaredisc_xgty(xvals, yvals):
        ratios = yvals / (xvals + 1e-12)
        return xvals * np.cos(np.pi / 4 * ratios), xvals * np.sin(np.pi / 4 * ratios)

    def _shirley_concentric_squaredisc_xlty(xvals, yvals):
        ratios = xvals / (yvals + 1e-12)
        return yvals * np.sin(np.pi / 4 * ratios), yvals * np.cos(np.pi / 4 * ratios)

    conditions = [
        np.repeat(np.atleast_2d(np.abs(xvals) >= np.abs(yvals)), 2, axis=0).transpose(),
        np.repeat(np.atleast_2d(np.abs(xvals) < np.abs(yvals)), 2, axis=0).transpose(),
    ]
    x_counters, y_counters = np.piecewise(
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
    return x_counters, y_counters
