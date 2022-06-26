import itertools as it

import numpy as np
from numpy import random as rn
import scipy.linalg as la

import pydrex.tensors as _tensors


def bingham_average(orientations):
    """Compute bingham averages from olivine orientation matrices.

    Returns a tuple of vectors which are the averaged
    a-, b- and c-axes orientations, respectively.

    """
    # https://courses.eas.ualberta.ca/eas421/lecturepages/orientation.html
    # Eigenvector corresponding to largest eigenvalue is the mean direction.
    # SciPy returns eigenvalues in ascending order (same order for vectors).
    a_mean = np.zeros((3, 3))
    a_mean[0, 0] = np.sum(orientations[:, 0, 0] ** 2)
    a_mean[1, 1] = np.sum(orientations[:, 0, 1] ** 2)
    a_mean[2, 2] = np.sum(orientations[:, 0, 2] ** 2)
    a_mean[0, 1] = np.sum(orientations[:, 0, 0] * orientations[:, 0, 1])
    a_mean[0, 2] = np.sum(orientations[:, 0, 0] * orientations[:, 0, 2])
    a_mean[1, 2] = np.sum(orientations[:, 0, 1] * orientations[:, 0, 2])
    _, a_eigenvectors = la.eigh(_tensors.upper_tri_to_symmetric(a_mean))
    a_vector = a_eigenvectors[:, -1]

    b_mean = np.zeros((3, 3))
    b_mean[0, 0] = np.sum(orientations[:, 1, 0] ** 2)
    b_mean[1, 1] = np.sum(orientations[:, 1, 1] ** 2)
    b_mean[1, 2] = np.sum(orientations[:, 1, 2] ** 2)
    b_mean[0, 1] = np.sum(orientations[:, 1, 0] * orientations[:, 1, 1])
    b_mean[0, 2] = np.sum(orientations[:, 1, 0] * orientations[:, 1, 2])
    b_mean[1, 2] = np.sum(orientations[:, 1, 1] * orientations[:, 1, 2])
    _, b_eigenvectors = la.eigh(_tensors.upper_tri_to_symmetric(b_mean))
    b_vector = b_eigenvectors[:, -1]

    c_mean = np.zeros((3, 3))
    c_mean[0, 0] = np.sum(orientations[:, 2, 0] ** 2)
    c_mean[1, 1] = np.sum(orientations[:, 2, 1] ** 2)
    c_mean[2, 2] = np.sum(orientations[:, 2, 2] ** 2)
    c_mean[0, 1] = np.sum(orientations[:, 2, 0] * orientations[:, 2, 1])
    c_mean[0, 2] = np.sum(orientations[:, 2, 0] * orientations[:, 2, 2])
    c_mean[1, 2] = np.sum(orientations[:, 2, 1] * orientations[:, 2, 2])
    _, c_eigenvectors = la.eigh(_tensors.upper_tri_to_symmetric(c_mean))
    c_vector = c_eigenvectors[:, -1]

    return (
        a_vector / la.norm(a_vector),
        b_vector / la.norm(b_vector),
        c_vector / la.norm(c_vector),
    )


def resample_orientations(orientations, fractions, n_samples=None):
    """Generate samples from `orientations` based on volume distribution.

    If the optional number of samples is not specified,
    it will be set to the number of original "grains" (length of `fractions`).

    """
    if n_samples is None:
        n_samples = len(fractions)
    sort_ascending = np.argsort(fractions)[::-1]
    frac_ascending = np.take(fractions, sort_ascending)
    # Cumulative volume fractions.
    cumfrac = np.cumsum(frac_ascending)
    # Number of new samples with volume less than each cumulative fraction.
    count_less = [np.searchsorted(cumfrac, r) for r in rn.random_sample(n_samples)]
    return np.stack([orientations[i] for i in count_less])


def M_index(orientations):
    """Calculate M-index for olivine polycrystal orientations.

    See [Skemer et al. 2005].

    [Skemer et al. 2005]: https://doi.org/10.1016/j.tecto.2005.08.023

    """
    misorientations = np.fromiter(
        (misorientation_angle(A, B) for A, B in it.combinations(orientations, 2)),
        dtype=float,
    )
    # Number of misorientations within 1° bins.
    count_misorientations, _ = np.histogram(misorientations, list(range(121)))
    return (1 / 2 / len(misorientations)) * np.sum(
        np.abs(
            [
                misorientations_random(i, i + 1) * len(misorientations)
                for i in range(120)
            ]
            - count_misorientations
        )
    )


def misorientation_angle(rot_a, rot_b):
    """Calculate the misorientation angle for a pair of rotation matrices.

    Calculate the angle of the difference rotation between `rot_a` and `rot_b`,
    which are expected to be 3x3 rotation matrices.

    """
    diff_rot = rot_a @ rot_b.T
    # Need to clip to [-1, 1] to avoid NaNs from np.arccos due to rounding errs.
    return np.rad2deg(np.arccos(np.clip(np.abs(np.trace(diff_rot) - 1) / 2, -1, 1)))


def misorientations_random(low, high, symmetry=(2, 4)):
    """Get expected count of misorientation angles for an isotropic aggregate.

    Estimate the expected number of misorientation angles between grains
    that would fall within (`low`, `high`) in degrees for an aggregate
    with randomly oriented grains.

    The optional argument `symmetry` accepts a tuple of integers (a, b)
    that specify the crystal symmetry system:

    system  triclinic  monoclinic  orthorhombic  rhombohedral tetragonal hexagonal
    ------------------------------------------------------------------------------
    M       1          2           2             3            4          6
    N       1          2           4             6            8          12

    This is identically Table 1 in [Grimmer 1979].
    The orthorhombic system (olivine) is selected by default.

    [Grimmer 1979]: https://doi.org/10.1016/0036-9748(79)90058-9

    """
    M, N = symmetry
    assert low >= 0
    if (M == 2 and N == 4) or (M == 3 and N == 6):
        maxval = 120
    elif (M == 4 and N == 8) or (M == 6 and N == 12):
        maxval = 90
    else:
        maxval = 180
    assert high <= maxval

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
