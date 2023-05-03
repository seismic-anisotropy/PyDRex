"""> PyDRex: Statistical methods for orientation and elasticity data."""
import numpy as np

from pydrex import minerals as _minerals
from pydrex import stiffness as _stiffness
from pydrex import tensors as _tensors
from pydrex import geometry as _geo

_RNG = np.random.default_rng(seed=8845)


def average_stiffness(minerals, config):
    """Calculate average elastic tensor from a list of `minerals`.

    The `config` dictionary must contain volume fractions of all occurring mineral phases,
    indexed by keys of the format `"<phase>_fraction"`.

    """
    n_grains = minerals[0].n_grains
    assert np.all(
        [m.n_grains == n_grains for m in minerals[1:]]
    ), "cannot average minerals with varying grain counts"
    elastic_tensors = []
    for phase in _minerals.MineralPhase:
        if phase == _minerals.MineralPhase.olivine:
            elastic_tensors.append(_tensors.Voigt_to_elastic_tensor(_stiffness.OLIVINE))
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


def point_density(
    x_data,
    y_data,
    z_data,
    gridsteps=101,
    weights=1,
    kernel="linear_inverse_kamb",
    axial=True,
    **kwargs,
):
    """Estimate point density of orientation data on the unit sphere.

    Estimates the density of orientations on the unit sphere by counting the input data
    that falls within small areas around a uniform grid of spherical counting locations.
    The input data is expected in cartesian coordinates, and the contouring is performed
    using kernel functions defined in [Vollmer 1995](https://doi.org/10.1016/0098-3004(94)00058-3).
    The following optional parameters control the contouring method:
    - `gridsteps` (int) — the number of steps, i.e. resolution of the spherical counting grid
    - `weights` (array) — auxiliary weights for each data point
    - `kernel` (string) — the name of the kernel function to use, see `SPHERICAL_COUNTING_KERNELS`
    - `axial` (bool) — toggle axial versions of the kernel functions
        (for crystallographic data this should normally be kept as `True`)

    Any other keyword arguments are passed to the kernel function calls.
    Most kernels accept a parameter `σ` to control the degree of smoothing.

    """
    if kernel not in SPHERICAL_COUNTING_KERNELS:
        raise ValueError(f"kernel '{kernel}' is not supported")
    weights = np.asarray(weights, dtype=np.float64)

    # Create a grid of counters on a cylinder.
    ρ_grid, h_grid = np.mgrid[-np.pi : np.pi : gridsteps * 1j, -1 : 1 : gridsteps * 1j]
    # Project onto the sphere using the equal-area projection with centre at (0, 0).
    λ_grid = ρ_grid
    ϕ_grid = np.arcsin(h_grid)
    x_counters, y_counters, z_counters = _geo.to_cartesian(
        np.pi / 2 - λ_grid.ravel(), np.pi / 2 - ϕ_grid.ravel()
    )

    # Basically, we can't model this as a convolution as we're not in Euclidean space,
    # so we have to iterate through and call the kernel function at each "counter".
    data = np.column_stack([x_data, y_data, z_data])
    counters = np.column_stack([x_counters, y_counters, z_counters])
    totals = np.empty(counters.shape[0])
    for i, counter in enumerate(counters):
        products = np.dot(data, counter)
        if axial:
            products = np.abs(products)
        density, scale = SPHERICAL_COUNTING_KERNELS[kernel](
            products, axial=axial, **kwargs
        )
        density *= weights
        totals[i] = (density.sum() - 0.5) / scale

    X_counters, Y_counters = _geo.lambert_equal_area(x_counters, y_counters, z_counters)

    # Normalise to mean, which estimates the density for a "uniform" distribution.
    totals /= totals.mean()
    totals[totals < 0] = 0
    print(totals.min(), totals.mean(), totals.max())
    return (
        np.reshape(X_counters, ρ_grid.shape),
        np.reshape(Y_counters, ρ_grid.shape),
        np.reshape(totals, ρ_grid.shape),
    )


def _kamb_radius(n, σ, axial):
    """Radius of kernel for Kamb-style smoothing."""
    r = σ**2 / (float(n) + σ**2)
    if axial is True:
        return 1 - r
    return 1 - 2 * r


def _kamb_units(n, radius):
    """Normalization function for Kamb-style counting."""
    return np.sqrt(n * radius * (1 - radius))


def exponential_kamb(cos_dist, σ=10, axial=True):
    """Kernel function from Vollmer 1995 for exponential smoothing."""
    n = float(cos_dist.size)
    if axial:
        f = 2 * (1.0 + n / σ**2)
        units = np.sqrt(n * (f / 2.0 - 1) / f**2)
    else:
        f = 1 + n / σ**2
        units = np.sqrt(n * (f - 1) / (4 * f**2))

    count = np.exp(f * (cos_dist - 1))
    return count, units


def linear_inverse_kamb(cos_dist, σ=10, axial=True):
    """Kernel function from Vollmer 1995 for linear smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, σ, axial=axial)
    f = 2 / (1 - radius)
    cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius)
    return count, _kamb_units(n, radius)


def square_inverse_kamb(cos_dist, σ=10, axial=True):
    """Kernel function from Vollmer 1995 for inverse square smoothing."""
    n = float(cos_dist.size)
    radius = _kamb_radius(n, σ, axial=axial)
    f = 3 / (1 - radius) ** 2
    cos_dist = cos_dist[cos_dist >= radius]
    count = f * (cos_dist - radius) ** 2
    return count, _kamb_units(n, radius)


def kamb_count(cos_dist, σ=10, axial=True):
    """Original Kamb 1959 kernel function (raw count within radius)."""
    n = float(cos_dist.size)
    dist = _kamb_radius(n, σ, axial=axial)
    count = (cos_dist >= dist).astype(float)
    return count, _kamb_units(n, dist)


def schmidt_count(cos_dist, axial=None):
    """Schmidt (a.k.a. 1%) counting kernel function."""
    radius = 0.01
    count = ((1 - cos_dist) <= radius).astype(float)
    # To offset the count.sum() - 0.5 required for the kamb methods...
    count = 0.5 / count.size + count
    return count, (cos_dist.size * radius)


SPHERICAL_COUNTING_KERNELS = {
    "kamb_count": kamb_count,
    "schmidt_count": schmidt_count,
    "exponential_kamb": exponential_kamb,
    "linear_inverse_kamb": linear_inverse_kamb,
    "square_inverse_kamb": square_inverse_kamb,
}
"""Kernel functions that return an un-summed distribution and a normalization factor.

Supported kernel functions are based on the discussion in
[Vollmer 1995](https://doi.org/10.1016/0098-3004(94)00058-3).
Kamb methods accept the parameter `σ` (default: 10) to control the degree of smoothing.
Values lower than 3 and higher than 20 are not recommended.

"""
