r"""PyDRex: Methods to calculate texture diagnostics.

NOTE: Calculations expect orientation matrices $a$ to represent passive
(i.e. alias) rotations, which are defined in terms of the extrinsic ZXZ
euler angles $ϕ, θ, φ$ as

$$
a = \begin{bmatrix}
    \cosφ\cosϕ - \cosθ\sinϕ\sinφ & \cosθ\cosϕ\sinφ + \cosφ\sinϕ & \sinφ\sinθ
   -\sinφ\cosϕ - \cosθ\sinϕ\cosφ & \cosθ\cosϕ\cosφ - \sinφ\sinϕ & \cosφ\sinθ
            \sinθ\sinϕ           &           -\sinθ\cosϕ        &   \cosθ
    \end{bmatrix}
$$

such that a[i, j] gives the direction cosine of the angle between the i-th
grain axis and the j-th external axis (in the global Eulerian frame).

"""
import itertools as it

import numpy as np
import scipy.linalg as la
from numpy import random as rn


def bingham_average(orientations, axis="a"):
    """Compute Bingham averages from olivine orientation matrices.

    Returns the antipodally symmetric average orientation
    of the given crystallographic `axis`, or the a-axis by default.
    Valid axis specifiers are "a" for [100], "b" for [010] and "c" for [001].

    See e.g. [Watson 1966].

    [Watson 1966]: https://doi.org/10.1086%2F627211

    """
    match axis:
        case "a":
            row = 0
        case "b":
            row = 1
        case "c":
            row = 2
        case _:
            raise ValueError(f"axis must be 'a', 'b', or 'c', not {axis}")

    # https://courses.eas.ualberta.ca/eas421/lecturepages/orientation.html
    # Eigenvector corresponding to largest eigenvalue is the mean direction.
    # SciPy returns eigenvalues in ascending order (same order for vectors).
    # SciPy uses lower triangular entries by default, no need for all components.
    mean_vector = la.eigh(_scatter_matrix(orientations, row))[1][:, -1]
    return mean_vector / la.norm(mean_vector)


def symmetry(orientations, axis="a"):
    """Compute texture symmetry eigenvalue diagnostics.

    Compute Point, Girdle and Random symmetry diagnostics
    for ternary texture classification.
    Returns the tuple (P, G, R) where

    $$
    P = (λ_{1} - λ_{2}) / N
    G = 2 (λ_{2} - λ_{3}) / N
    R = 3 λ_{3} / N
    $$

    with N the sum of the eigenvalues $λ_{1} ≥ λ_{2} ≥ λ_{3}$
    of the scatter (inertia) matrix.

    See e.g. [Vollmer 1990].

    [Vollmer 1990]: https://doi.org/10.1130/0016-7606(1990)102%3C0786:AAOEMT%3E2.3.CO;2

    """
    match axis:
        case "a":
            row = 0
        case "b":
            row = 1
        case "c":
            row = 2
        case _:
            raise ValueError(f"axis must be 'a', 'b', or 'c', not {axis}")

    scatter = _scatter_matrix(orientations, row)
    # SciPy uses lower triangular entries by default, no need for all components.
    eigvals_descending = la.eigvalsh(scatter)[::-1]
    sum_eigvals = np.sum(eigvals_descending)
    return (
        (eigvals_descending[0] - eigvals_descending[1]) / sum_eigvals,
        2 * (eigvals_descending[1] - eigvals_descending[2]) / sum_eigvals,
        3 * eigvals_descending[2] / sum_eigvals,
    )


def _scatter_matrix(orientations, row):
    # Lower triangular part of the symmetric scatter (inertia) matrix,
    # see eq. 2.4 in Watson 1966.
    scatter = np.zeros((3, 3))
    scatter[0, 0] = np.sum(orientations[:, row, 0] ** 2)
    scatter[1, 1] = np.sum(orientations[:, row, 1] ** 2)
    scatter[2, 2] = np.sum(orientations[:, row, 2] ** 2)
    scatter[1, 0] = np.sum(orientations[:, row, 0] * orientations[:, row, 1])
    scatter[2, 0] = np.sum(orientations[:, row, 0] * orientations[:, row, 2])
    scatter[2, 1] = np.sum(orientations[:, row, 1] * orientations[:, row, 2])
    return scatter


def resample_orientations(orientations, fractions, n_samples=None):
    """Generate new samples from `orientations` based on volume distribution.

    If the optional number of samples is not specified,
    it will be set to the number of original "grains" (length of `fractions`).

    """
    if n_samples is None:
        n_samples = len(fractions)
    sort_ascending = np.argsort(fractions)
    # Cumulative volume fractions
    fractions_ascending = fractions[sort_ascending]
    cumfrac = fractions_ascending.cumsum()
    # Number of new samples with volume less than each cumulative fraction.
    rng = rn.default_rng()
    count_less = np.searchsorted(cumfrac, rng.random(n_samples))
    return orientations[sort_ascending][count_less], fractions_ascending[count_less]


def misorientation_index(orientations):
    """Calculate M-index for olivine polycrystal orientations.

    See [Skemer et al. 2005].

    [Skemer et al. 2005]: https://doi.org/10.1016/j.tecto.2005.08.023

    """
    misorientations = misorientation_angles(
        np.array(list(it.combinations(orientations, 2)))
    )
    # Number of misorientations within 1° bins.
    # TODO: Make n_bins and θmax args to the function.
    count_misorientations, _ = np.histogram(misorientations, bins=120, range=(0, 120))
    return (1 / 2) * np.sum(
        np.abs(
            [
                misorientations_random(i, i + 1)
                - count_misorientations[i] / len(misorientations)
                for i in range(120)
            ]
        )
    )


def misorientation_angles(combinations):
    """Calculate the misorientation angles for pairs of rotation matrices.

    Calculate the angular distance between the rotations `combinations[:, 0]`
    and `combinations[:, 1]`, which are expected to be 3x3 passive (alias)
    rotation matrices.

    See also <http://boris-belousov.net/2016/12/01/quat-dist/>.

    """
    return np.rad2deg(
        np.arccos(
            np.clip(
                (
                    np.trace(
                        combinations[:, 0]
                        @ np.transpose(combinations[:, 1], axes=[0, 2, 1]),
                        axis1=1,
                        axis2=2,
                    )
                    - 1.0
                )
                / 2,
                -1.0,
                1.0,
            )
        )
    )


def misorientations_random(low, high, symmetry=(2, 4)):
    """Get expected count of misorientation angles for an isotropic aggregate.

    Estimate the expected number of misorientation angles between grains
    that would fall within (`low`, `high`) in degrees for an aggregate
    with randomly oriented grains, where `low` ∈ [0, `high`),
    and `high` is bounded by the maximum theoretical misorientation angle θmax.

    The optional argument `symmetry` accepts a tuple of integers (a, b)
    that specify the crystal symmetry system:

    system  triclinic  monoclinic  orthorhombic  rhombohedral tetragonal hexagonal
    ------------------------------------------------------------------------------
    M       1          2           2             3            4          6
    N       1          2           4             6            8          12
    θmax    180°       180°        120°          120°         90°        90°

    This is identically Table 1 in [Grimmer 1979].
    The orthorhombic system (olivine) is selected by default.

    [Grimmer 1979]: https://doi.org/10.1016/0036-9748(79)90058-9

    """
    match symmetry:
        case (2, 4) | (3, 6):
            maxval = 120
        case (4, 8) | (6, 12):
            maxval = 90
        case (1, 1) | (2, 2):
            maxval = 180
        case _:
            raise ValueError(f"incorrect symmetry values (M, N) = {symmetry}")
    M, N = symmetry
    if not 0 <= low <= high <= maxval:
        raise ValueError(
            f"bounds must obey `low` ∈ [0, `high`) and `high` < {maxval}.\n"
            + f"You've supplied (`low`, `high`) = ({low}, {high})."
        )

    counts_low = 0  # Number of counts at the lower bin edge.
    counts_high = 0  # ... at the higher bin edge.
    counts_both = [counts_low, counts_high]

    # Some constant factors.
    a = np.tan(np.deg2rad(90 / M))
    b = 2 * np.rad2deg(np.arctan(np.sqrt(1 + a**2)))
    c = round(2 * np.rad2deg(np.arctan(np.sqrt(1 + 2 * a**2))))

    for i, edgeval in enumerate([low, high]):
        d = np.deg2rad(edgeval)

        if 0 <= edgeval <= (180 / M):
            counts_both[i] += (N / 180) * (1 - np.cos(d))

        elif (180 / M) <= edgeval <= (180 * M / N):
            counts_both[i] += (N / 180) * a * np.sin(d)

        elif 90 <= edgeval <= b:
            counts_both[i] += (M / 90) * ((M + a) * np.sin(d) - M * (1 - np.cos(d)))

        elif b <= edgeval <= c:
            ν = np.tan(np.deg2rad(edgeval / 2)) ** 2

            counts_both[i] = (M / 90) * (
                (M + a) * np.sin(d)
                - M * (1 - np.cos(d))
                + (M / 180)
                * (
                    (1 - np.cos(d))
                    * (
                        np.rad2deg(
                            np.arccos((1 - ν * np.cos(np.deg2rad(180 / M))) / (ν - 1))
                        )
                        + 2
                        * np.rad2deg(
                            np.arccos(a / (np.sqrt(ν - a**2) * np.sqrt(ν - 1)))
                        )
                    )
                    - 2
                    * np.sin(d)
                    * (
                        2 * np.rad2deg(np.arccos(a / np.sqrt(ν - 1)))
                        + a * np.rad2deg(np.arccos(1 / np.sqrt(ν - a**2)))
                    )
                )
            )
        else:
            assert False  # Should never happen.

    return np.sum(counts_both) / 2


def smallest_angle(vector, axis):
    """Get smallest angle between a unit `vector` and a bidirectional `axis`.

    The axis is specified using either of its two parallel unit vectors.

    """
    angle = np.rad2deg(
        np.arccos(np.dot(vector, axis) / (la.norm(vector) * la.norm(axis)))
    )
    if angle > 90:
        return 180 - angle
    return angle
