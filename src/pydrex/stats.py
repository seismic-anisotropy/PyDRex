"""> PyDRex: Statistical methods for orientation and elasticity data."""

import itertools as it

import numpy as np
import scipy.special as sp
from scipy.spatial.transform import Rotation

from pydrex import geometry as _geo
from pydrex import stats as _stats
from pydrex import utils as _utils


def resample_orientations(
    orientations: np.ndarray,
    fractions: np.ndarray,
    n_samples: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return new samples from `orientations` weighted by the volume distribution.

    - `orientations` — NxMx3x3 array of orientations
    - `fractions` — NxM array of grain volume fractions
    - `n_samples` — optional number of samples to return, default is M
    - `seed` — optional seed for the random number generator, which is used to pick
      random grain volume samples from the discrete distribution

    Returns the Nx`n_samples`x3x3 orientations and associated sorted (ascending) grain
    volumes.

    """
    # Allow lists of Rotation.as_matrix() inputs.
    _orientations = np.asarray(orientations)
    _fractions = np.asarray(fractions)
    # Fail early to prevent possibly expensive data processing on incorrect data arrays.
    if (
        len(_orientations.shape) != 4
        or len(_fractions.shape) != 2
        or _orientations.shape[0] != _fractions.shape[0]
        or _orientations.shape[1] != _fractions.shape[1]
        or _orientations.shape[2] != _orientations.shape[3] != 3
    ):
        raise ValueError(
            "invalid shape of input arrays,"
            + f" got orientations of shape {_orientations.shape}"
            + f" and fractions of shape {_fractions.shape}"
        )
    rng = np.random.default_rng(seed=seed)
    if n_samples is None:
        n_samples = _fractions.shape[1]
    out_orientations = np.empty((len(_fractions), n_samples, 3, 3))
    out_fractions = np.empty((len(_fractions), n_samples))
    for i, (frac, orient) in enumerate(zip(_fractions, _orientations, strict=True)):
        sort_ascending = np.argsort(frac)
        # Cumulative volume fractions.
        frac_ascending = frac[sort_ascending]
        cumfrac = frac_ascending.cumsum()
        # Force cumfrac[-1] to be equal to sum(frac_ascending) i.e. 1.
        cumfrac[-1] = 1.0
        # Number of new samples with volume less than each cumulative fraction.
        count_less = np.searchsorted(cumfrac, rng.random(n_samples))
        out_orientations[i, ...] = orient[sort_ascending][count_less]
        out_fractions[i, ...] = frac_ascending[count_less]
    return out_orientations, out_fractions


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


def misorientation_hist(
    orientations: np.ndarray, system: _geo.LatticeSystem, bins: int | None = None
):
    r"""Calculate misorientation histogram for polycrystal orientations.

    The `bins` argument is passed to `numpy.histogram`.
    If left as `None`, 1° bins will be used as recommended by the reference paper.
    The `symmetry` argument specifies the lattice system which determines intrinsic
    symmetry degeneracies and the maximum allowable misorientation angle.
    See `_geo.LatticeSystem` for supported systems.

    .. warning::
        This method must be able to allocate $ \frac{N!}{(N-2)!} × 4M $ floats
        for N the length of `orientations` and M the number of symmetry operations for
        the given `system` (`numpy.float32` values are used to reduce the memory
        requirements)

    See [Skemer et al. (2005)](https://doi.org/10.1016/j.tecto.2005.08.023).

    """
    symmetry_ops = _geo.symmetry_operations(system)
    # Compute and bin misorientation angles from orientation data.
    q1_array = np.empty(
        (sp.comb(len(orientations), 2, exact=True), len(symmetry_ops), 4),
        dtype=np.float32,
    )
    q2_array = np.empty(
        (sp.comb(len(orientations), 2, exact=True), len(symmetry_ops), 4),
        dtype=np.float32,
    )
    for i, e in enumerate(  # Copy is required for proper object referencing in Ray.
        it.combinations(Rotation.from_matrix(orientations.copy()).as_quat(), 2)
    ):
        q1, q2 = list(e)
        for j, qs in enumerate(symmetry_ops):
            if qs.shape == (4, 4):  # Reflection, not a proper rotation.
                q1_array[i, j] = qs @ q1
                q2_array[i, j] = qs @ q2
            else:
                q1_array[i, j] = _utils.quat_product(qs, q1)
                q2_array[i, j] = _utils.quat_product(qs, q2)

    misorientations_data = _geo.misorientation_angles(q1_array, q2_array)
    θmax = _stats._max_misorientation(system)
    return np.histogram(misorientations_data, bins=θmax, range=(0, θmax), density=True)


def misorientations_random(low: float, high: float, system: _geo.LatticeSystem):
    """Get expected count of misorientation angles for an isotropic aggregate.

    Estimate the expected number of misorientation angles between grains
    that would fall within $($`low`, `high`$)$ in degrees for an aggregate
    with randomly oriented grains, where `low` $∈ [0, $`high`$)$,
    and `high` is bounded by the maximum theoretical misorientation angle
    for the given lattice symmetry system.
    See `_geo.LatticeSystem` for supported systems.

    """
    # TODO: Add cubic system: [Handscomb 1958](https://doi.org/10.4153/CJM-1958-010-0)
    max_θ = _max_misorientation(system)
    M, N = system.value
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


def _max_misorientation(system: _geo.LatticeSystem):
    # Maximum misorientation angle for two grains of the given lattice symmetry system.
    s = _geo.LatticeSystem
    match system:
        case s.orthorhombic | s.rhombohedral:
            max_θ = 120
        case s.tetragonal | s.hexagonal:
            max_θ = 90
        case s.triclinic | s.monoclinic:
            max_θ = 180
        case _:
            raise ValueError(f"unsupported lattice system: {system}")
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
    - `gridsteps` (int) — the number of steps, i.e. number of points along a diameter of
        the spherical counting grid
    - `weights` (array) — auxiliary weights for each data point
    - `kernel` (string) — the name of the kernel function to use, see
      `SPHERICAL_COUNTING_KERNELS`
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
    # print(totals.min(), totals.mean(), totals.max())
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
