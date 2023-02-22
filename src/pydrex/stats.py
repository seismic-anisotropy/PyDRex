"""> PyDRex: Statistical methods for orientation and elasticity data."""
import numpy as np

from pydrex import tensors as _tensors
from pydrex import stiffness as _stiffness
from pydrex import minerals as _minerals

_RNG = np.random.default_rng()


def average_stiffness(minerals, config):
    """Calculate average elastic tensor from a list of `minerals`.

    The `config` dictionary must contain volume fractions of all occuring mineral phases,
    indexed by keys of the format `"<phase>_fraction"`.

    """
    n_grains = minerals[0].n_grains
    assert np.all(
        [m.n_grains == n_grains for m in minerals[1:]]
    ), "cannot average minerals with varying grain counts"
    elastic_tensors = []
    for phase in _minerals.MineralPhase:
        if phase == _minerals.MineralPhase.olivine:
            elastic_tensors.append(
                _tensors.Voigt_to_elastic_tensor(_stiffness.OLIVINE)
            )
        elif phase == _minerals.MineralPhase.enstatite:
            elastic_tensors.append(
                _tensors.Voigt_to_elastic_tensor(_stiffness.ENSTATITE)
            )

    average_tensor = np.zeros((3, 3, 3, 3))
    for n in n_grains:
        for mineral in minerals:
            if mineral.phase == _minerals.MineralPhase.olivine:
                average_tensor += (
                    _tensors.rotated_tensor(mineral.orientations[n, ...].transpose())
                    * mineral.fractions[n]
                    * config["olivine_fraction"]
                )
            elif mineral.phase == _minerals.MineralPhase.enstatite:
                average_tensor += (
                    _tensors.rotated_tensor(minerals.orientations[n, ...].transpose())
                    * mineral.fractions[n]
                    * config["enstatite_fraction"]
                )
    return _tensors.elastic_tensor_to_Voigt(average_tensor)


def resample_orientations(orientations, fractions, n_samples=None, rng=_RNG):
    """Generate new samples from `orientations` weighed by the volume distribution.

    If the optional number of samples `n_samples` is not specified,
    it will be set to the number of original "grains" (length of `fractions`).
    The argument `rng` can be used to specify a custom random number generator.

    """
    if n_samples is None:
        n_samples = len(fractions)
    sort_ascending = np.argsort(fractions)
    # Cumulative volume fractions
    fractions_ascending = fractions[sort_ascending]
    cumfrac = fractions_ascending.cumsum()
    # Number of new samples with volume less than each cumulative fraction.
    count_less = np.searchsorted(cumfrac, rng.random(n_samples))
    return orientations[sort_ascending][count_less], fractions_ascending[count_less]


def _scatter_matrix(orientations, row):
    # Lower triangular part of the symmetric scatter (inertia) matrix,
    # see eq. 2.4 in Watson 1966 or eq. 9.2.10 in Mardia & Jupp 2009 (with n = 1),
    # it's a summation of the outer product of the [h, k, l] vector with itself,
    # so taking the row assumes that `orientations` are passive rotations of the
    # reference frame [h, k, l] vector.
    scatter = np.zeros((3, 3))
    scatter[0, 0] = np.sum(orientations[:, row, 0] ** 2)
    scatter[1, 1] = np.sum(orientations[:, row, 1] ** 2)
    scatter[2, 2] = np.sum(orientations[:, row, 2] ** 2)
    scatter[1, 0] = np.sum(orientations[:, row, 0] * orientations[:, row, 1])
    scatter[2, 0] = np.sum(orientations[:, row, 0] * orientations[:, row, 2])
    scatter[2, 1] = np.sum(orientations[:, row, 1] * orientations[:, row, 2])
    return scatter


def misorientations_random(low, high, system=(2, 4)):
    """Get expected count of misorientation angles for an isotropic aggregate.

    Estimate the expected number of misorientation angles between grains
    that would fall within $($`low`, `high`$)$ in degrees for an aggregate
    with randomly oriented grains, where `low` $∈ [0, $`high`$)$,
    and `high` is bounded by the maximum theoretical misorientation angle
    for the given symmetry system.

    The optional argument `system` accepts a tuple of integers (a, b)
    that specifies the crystal symmetry system according to:

        system  triclinic  monoclinic  orthorhombic  rhombohedral tetragonal hexagonal
        ------------------------------------------------------------------------------
        a       1          2           2             3            4          6
        b       1          2           4             6            8          12
        θmax    180°       180°        120°          120°         90°        90°

    This is identically Table 1 in [Grimmer 1979](https://doi.org/10.1016/0036-9748(79)90058-9).
    The orthorhombic system (olivine) is selected by default.

    """
    max_θ = _max_misorientation(system)
    M, N = system
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


def _max_misorientation(system):
    # Maximum misorientation angle for two grains of the given symmetry system.
    match system:
        case (2, 4) | (3, 6):
            max_θ = 120
        case (4, 8) | (6, 12):
            max_θ = 90
        case (1, 1) | (2, 2):
            max_θ = 180
        case _:
            raise ValueError(f"incorrect system values (M, N) = {system}")
    return max_θ
