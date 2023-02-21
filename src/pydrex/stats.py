"""> PyDRex: Statistical methods."""
import numpy as np
from scipy.spatial.transform import Rotation

_RNG = np.random.default_rng()


def isotropic_orientations(n_grains, symmetry=(2, 4), rng=_RNG):
    """Get orientation matrices for an isotropic polycrystal of minerals.

    The optional argument `symmetry` accepts a tuple of integers (a, b)
    that specifies the crystal symmetry system of the mineral.
    See `misorientations_random` for details.

    The optional argument `rng` can be used to specify an alternative
    pseudorandom generator, see `numpy.random.Generator`.

    NOTE: Experimental, currently only marginally better than
    `scipy.spatial.transform.Rotation.random()` and definitely more expensive.

    """
    # Discrete density of misorientation angles used as weights to the sampler.
    max_θ = _max_misorientation(symmetry)
    weights = np.array(
        [misorientations_random(θ - 1, θ, symmetry) for θ in range(1, max_θ + 1)]
    )
    weights /= weights.sum()
    # Allocate n_grains worth of identity rotations.
    orientations = Rotation.from_quat(np.tile([0, 0, 0, 1], (n_grains, 1)))
    # Set random first orientation.
    orientations[0] = Rotation.random()
    axis_map = {0: "X", 1: "Y", 2: "Z"}
    for n in range(1, n_grains):
        # The location of each new grain is determined by the misorientation angle from
        # the mean of all existing grains.
        # TODO: Improve by composing with the 'most remote' orientation instead of n - 1?
        δθ = rng.choice(max_θ, p=weights)
        orientations[n] = (
            orientations[n - 1]
            * Rotation.mean(orientations[:n])
            * Rotation.from_euler(axis_map[rng.choice(3)], δθ, degrees=True)
        )
    return orientations


def misorientations_random(low, high, symmetry=(2, 4)):
    """Get expected count of misorientation angles for an isotropic aggregate.

    Estimate the expected number of misorientation angles between grains
    that would fall within (`low`, `high`) in degrees for an aggregate
    with randomly oriented grains, where `low` ∈ [0, `high`),
    and `high` is bounded by the maximum theoretical misorientation angle θmax.

    The optional argument `symmetry` accepts a tuple of integers (a, b)
    that specifies the crystal symmetry system:

    system  triclinic  monoclinic  orthorhombic  rhombohedral tetragonal hexagonal
    ------------------------------------------------------------------------------
    M       1          2           2             3            4          6
    N       1          2           4             6            8          12
    θmax    180°       180°        120°          120°         90°        90°

    This is identically Table 1 in [Grimmer 1979].
    The orthorhombic system (olivine) is selected by default.

    [Grimmer 1979]: https://doi.org/10.1016/0036-9748(79)90058-9

    """
    max_θ = _max_misorientation(symmetry)
    M, N = symmetry
    if not 0 <= low <= high <= max_θ:
        raise ValueError(
            f"bounds must obey `low` ∈ [0, `high`) and `high` < {max_θ}.\n"
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


def _max_misorientation(symmetry):
    match symmetry:
        case (2, 4) | (3, 6):
            max_θ = 120
        case (4, 8) | (6, 12):
            max_θ = 90
        case (1, 1) | (2, 2):
            max_θ = 180
        case _:
            raise ValueError(f"incorrect symmetry values (M, N) = {symmetry}")
    return max_θ
