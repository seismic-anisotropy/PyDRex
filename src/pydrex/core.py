"""PyDRex: Core DRex functions.

Pathlines are calculated from the input velocity and velocity gradient fields.
At each point on the pathline, CPO is calculated by discretizing the virtual particle
into volume bins based on lattice rotation distribution. The number of volume elements
is fixed, but their relative volumes can change. Volume elements do not represent
physical grains, instead providing an efficient alternative volume discretization.

Volume elements do not have a defined spatial extent, and all calculations are
performed at the centre point. For computational efficiency, the DRex model treats
any interactions with other elements as interactions with an averaged effective medium.

"""
import numpy as np
import numba as nb
import numpy.linalg as la
import scipy.integrate as si
from scipy.spatial.transform import Rotation

import pydrex.tensors as _tensors
import pydrex.interpolations as _interp
import pydrex.fabric as _fabric

# import pydrex.deformation_mechanism as _defmech


def solve(config, interpolators, node):
    """Solve the DRex equations for steady state flow.

    Calculate crystalline preferred orientation matrices and their volume distribution.

    Args:
        `config` (dict) — PyDRex configuration dictionary
        `interpolators` (dict) — interpolator callbacks for input fields
        `node` (tuple) — coordinates of the interpolation grid node

    Returns a tuple containing:
    - the finite strain ellipsoid
    - the olivine orientation matrices
    - the enstatite orientation matrices
    - the volume distribution of olivine orientations
    - the volume distribution of enstatite orientations

    """
    # Construct element centre coordinates, halfway between this node and the next.
    point = np.array(
        [
            (coord[i] + coord[i + 1]) / 2
            for coord, i in zip(config["mesh"]["gridcoords"], node)
        ]
    )
    # Calculate pathline (timestamps and a callback).
    path_times, path_eval = get_pathline(
        point, interpolators, config["mesh"]["gridmin"], config["mesh"]["gridmax"]
    )
    # TODO: Handle this in pathline construction somehow?
    # Skip the first timestep if it is outside the numerical domain.
    # Needed because `pathline` will stop _after_ checking _is_inside in _max_strain.
    for t in reversed(path_times):
        if _is_inside(
            path_eval(t), config["mesh"]["gridmin"], config["mesh"]["gridmax"]
        ):
            time = t
            break

    finite_strain_ell = np.identity(3)  # Finite strain ellipsoid.
    init_orientations = Rotation.random(  # Rotation matrices.
        config["number_of_grains"], random_state=1
    ).as_matrix()
    olivine_orientations = init_orientations.copy()
    enstatite_orientations = init_orientations.copy()
    # Volume discretization for each mineral, initially uniform.
    olivine_vol_dist = np.ones(config["number_of_grains"]) / config["number_of_grains"]
    enstatite_vol_dist = olivine_vol_dist.copy()

    while time < path_times[0]:
        point = path_eval(time)
        # Get interpolated field values.
        velocity = _interp.get_velocity(point, interpolators)
        velocity_gradient = _interp.get_velocity_gradient(point, interpolators)
        # deformation_mechanism = _interp.get_deformation_mechanism(point, interpolators)

        # Imposed macroscopic strain rate tensor.
        strain_rate = (velocity_gradient + velocity_gradient.transpose()) / 2
        # Strain rate scale (max. eigenvalue of strain rate).
        strain_rate_max = np.abs(np.eigvalsh(strain_rate)).max()

        grid_steps = _get_steps(config["mesh"]["gridcoords"])
        dt_pathline = min(
            np.min(grid_steps) / 4 / la.norm(velocity, ord=2),
            path_times[0] - time,
        )
        dt = min(dt_pathline, 1e-2 / strain_rate_max)
        n_iter = int(dt_pathline / dt)

        for _ in range(n_iter):
            # TODO: Handle deformation mechanism regimes.
            finite_strain_ell = update_strain(finite_strain_ell, velocity_gradient, dt)
            olivine_orientations, olivine_vol_dist = update_orientations(
                olivine_orientations,
                olivine_vol_dist,
                strain_rate,
                strain_rate_max,
                velocity_gradient,
                _fabric.RRSS_OLIVINE_A,
                config,
                dt,
            )
            enstatite_orientations, enstatite_vol_dist = update_orientations(
                enstatite_orientations,
                enstatite_vol_dist,
                strain_rate,
                strain_rate_max,
                velocity_gradient,
                _fabric.RRSS_ENSTATITE,
                config,
                dt,
            )

        time += n_iter * dt
    return (
        finite_strain_ell,
        olivine_orientations,
        enstatite_orientations,
        olivine_vol_dist,
        enstatite_vol_dist,
    )


def _get_steps(a):
    """Get forward difference of 2D array `a`, with repeated last elements.

    The repeated last elements ensure that output and input arrays have equal shape.

    Examples:

    >>> _get_steps(np.array([1, 2, 3, 4, 5]))
    array([[1, 1, 1, 1, 1]])

    >>> _get_steps(np.array([[1, 2, 3, 4, 5], [1, 3, 6, 9, 10]]))
    array([[1, 1, 1, 1, 1],
           [2, 3, 3, 1, 1]])

    >>> _get_steps(np.array([[1, 2, 3, 4, 5], [1, 3, 6, 9, 10], [1, 0, 0, 0, np.inf]]))
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 2.,  3.,  3.,  1.,  1.],
           [-1.,  0.,  0., inf, nan]])

    """
    a2 = np.atleast_2d(a)
    return np.diff(
        a2, append=np.reshape(a2[:, -1] + (a2[:, -1] - a2[:, -2]), (a2.shape[0], 1))
    )


def get_pathline(point, interpolators, min_coords, max_coords):
    """Determine the pathline for a crystal aggregate in a steady state flow.

    The pathline will intersect the given `point` and follow a curve determined by
    the velocity gradient interpolators.

    Args:
        `point` (NumPy array) — coordinates of the point
        `interpolators` (dict) — dictionary with velocity gradient interpolators
        `min_coords` (iterable) — lower bound coordinate of the interpolation grid
        `max_coords` (iterable) — upper bound coordinate of the interpolation grid

    Returns a tuple containing the time points and a callback that can be used
    to evaulate the pathline position (see `scipy.integrate.OdeSolution`).

    """

    def _max_strain(time, point, interpolators, min_coords, max_coords):
        nonlocal event_time, event_time_prev, event_strain_prev, event_strain, event_strain_prev, event_flag
        # TODO: Refactor, move 10 "max strain" parameter to config?
        if event_flag:
            return (event_strain if time == event_time else event_strain_prev) - 10

        if _is_inside(point, min_coords, max_coords):
            velocity_gradient = _interp.get_velocity_gradient(point, interpolators)
            assert abs(np.trace(velocity_gradient)) < 1e-15
            # Imposed macroscopic strain rate tensor.
            strain_rate = (velocity_gradient + velocity_gradient.transpose()) / 2
            # Strain rate scale (max. eigenvalue of strain rate).
            strain_rate_max = np.abs(np.eigvalsh(strain_rate)).max()
            event_strain_prev = event_strain
            event_strain += abs(time - event_time) * strain_rate_max
            if event_strain >= 10:
                event_flag = True
            event_time_prev = event_time
            event_time = time
            return event_strain - 10

        return 0

    _max_strain.terminal = True
    event_strain = event_time = 0
    event_strain_prev = event_time_prev = None
    event_flag = False

    # Initial condition is the final position of the crystal.
    # Solve backwards in time until outside the domain or max. strain reached.
    # We don't want to stop at a particular time,
    # so integrate time for 100e6 days, in seconds (forever).
    path = si.solve_ivp(
        _ivp_func,
        [0, -100e6 * 365.25 * 8.64e4],
        point,
        method="RK45",
        first_step=1e10,
        max_step=np.inf,
        events=[_max_strain],
        args=(interpolators, min_coords, max_coords),
        dense_output=True,
        jac=_ivp_jac,
        atol=1e-8,
        rtol=1e-5,
    )
    return path.t, path.sol


def _ivp_func(time, point, interpolators, min_coords, max_coords):
    """Internal use only, changing function signature requires changes in `pathline`."""
    if _is_inside(point, min_coords, max_coords):
        return _interp.get_velocity_gradient(point, interpolators)
    return np.zeros(np.asarray(point).shape)


def _ivp_jac(time, point, interpolators, min_coords, max_coords):
    """Internal use only, must have the same signature as `_ivp_func`."""
    if _is_inside(point, min_coords, max_coords):
        return _interp.get_velocity_gradient(point, interpolators)
    return np.zeros([np.asarray(point).size] * 2)


def _is_inside(point, min_coords, max_coords):
    """Check if point lies within the numerical domain."""
    assert len(point) == len(min_coords)
    assert len(point) == len(max_coords)
    if np.any(point < min_coords) or np.any(point > max_coords):
        return False
    return True


def update_strain(finite_strain_ell, velocity_gradient, dt):
    """Update finite strain ellipsoid using the RK4 scheme."""
    fse1 = np.dot(velocity_gradient, finite_strain_ell) * dt
    fsei = finite_strain_ell + 0.5 * fse1
    fse2 = np.dot(velocity_gradient, fsei) * dt
    fsei = finite_strain_ell + 0.5 * fse2
    fse3 = np.dot(velocity_gradient, fsei) * dt
    fsei = finite_strain_ell + fse3
    fse4 = np.dot(velocity_gradient, fsei) * dt
    finite_strain_ell += (fse1 / 2 + fse2 + fse3 + fse4 / 2) / 3
    return finite_strain_ell


@nb.njit(fastmath=True)
def update_orientations(
    orientations,
    vol_dist,
    strain_rate,
    strain_rate_max,
    velocity_gradient,
    rrss,
    config,
    dt,
):
    """Update CPO orientations and their volume distribution using the RK4 scheme."""
    # Dimensionless strain rate and velocity gradient.
    _strain_rate = strain_rate / strain_rate_max
    _velocity_gradient = velocity_gradient / strain_rate_max

    # ========== RK step 1 ==========
    (rotation_rates, slip_rates, slip_indices, slip_rate_softest,) = get_rotation_rates(
        orientations,
        _strain_rate,
        _velocity_gradient,
        rrss,
        config["stress_exponent"],
    )
    strain_residuals = get_strain_energy_residuals(
        vol_dist,
        slip_rates,
        slip_indices,
        slip_rate_softest,
        rrss,
        config["stress_exponent"],
        config["nucleation_efficiency"],
    )
    if rrss == _fabric.RRSS_OLIVINE_A:
        vol_dist_diff = (
            config["olivine_fraction"] * config["gbm_mobility"] * strain_residuals
        )
    elif rrss == _fabric.RRSS_ENSTATITE:
        vol_dist_diff = (
            config["enstatite_fraction"] * config["gbm_mobility"] * strain_residuals
        )

    orientations_1 = rotation_rates * dt * strain_rate_max
    orientations_iter = np.clip(orientations + 0.5 * orientations_1, -1, 1)
    vol_dist_1 = vol_dist_diff * dt * strain_rate_max
    vol_dist_iter = np.clip(vol_dist + 0.5 * vol_dist_1, 0, None)
    vol_dist_iter /= vol_dist_iter.sum()

    # ========== RK step 2 ==========

    (rotation_rates, slip_rates, slip_indices, slip_rate_softest,) = get_rotation_rates(
        orientations_iter,
        _strain_rate,
        _velocity_gradient,
        rrss,
        config["stress_exponent"],
    )
    strain_residuals = get_strain_energy_residuals(
        vol_dist_iter,
        slip_rates,
        slip_indices,
        slip_rate_softest,
        rrss,
        config["stress_exponent"],
        config["nucleation_efficiency"],
    )
    if rrss == _fabric.RRSS_OLIVINE_A:
        vol_dist_diff = (
            config["olivine_fraction"] * config["gbm_mobility"] * strain_residuals
        )
    elif rrss == _fabric.RRSS_ENSTATITE:
        vol_dist_diff = (
            config["enstatite_fraction"] * config["gbm_mobility"] * strain_residuals
        )

    orientations_2 = rotation_rates * dt * strain_rate_max
    orientations_iter = np.clip(orientations_iter + 0.5 * orientations_2, -1, 1)
    vol_dist_2 = vol_dist_diff * dt * strain_rate_max
    vol_dist_iter = np.clip(vol_dist_iter + 0.5 * vol_dist_2, 0, None)
    vol_dist_iter /= vol_dist_iter.sum()

    # ========== rk step 3 ==========

    (rotation_rates, slip_rates, slip_indices, slip_rate_softest,) = get_rotation_rates(
        orientations_iter,
        _strain_rate,
        _velocity_gradient,
        rrss,
        config["stress_exponent"],
    )
    strain_residuals = get_strain_energy_residuals(
        vol_dist_iter,
        slip_rates,
        slip_indices,
        slip_rate_softest,
        rrss,
        config["stress_exponent"],
        config["nucleation_efficiency"],
    )
    if rrss == _fabric.RRSS_OLIVINE_A:
        vol_dist_diff = (
            config["olivine_fraction"] * config["gbm_mobility"] * strain_residuals
        )
    elif rrss == _fabric.RRSS_ENSTATITE:
        vol_dist_diff = (
            config["enstatite_fraction"] * config["gbm_mobility"] * strain_residuals
        )

    orientations_3 = rotation_rates * dt * strain_rate_max
    orientations_iter = np.clip(orientations_iter + orientations_3, -1, 1)
    vol_dist_3 = vol_dist_diff * dt * strain_rate_max
    vol_dist_iter = np.clip(vol_dist_iter + vol_dist_3, 0, None)
    vol_dist_iter /= vol_dist_iter.sum()

    # ========== rk step 4 ==========

    (rotation_rates, slip_rates, slip_indices, slip_rate_softest,) = get_rotation_rates(
        orientations_iter,
        _strain_rate,
        _velocity_gradient,
        rrss,
        config["stress_exponent"],
    )
    strain_residuals = get_strain_energy_residuals(
        vol_dist_iter,
        slip_rates,
        slip_indices,
        slip_rate_softest,
        rrss,
        config["stress_exponent"],
        config["nucleation_efficiency"],
    )
    if rrss == _fabric.RRSS_OLIVINE_A:
        vol_dist_diff = (
            config["olivine_fraction"] * config["gbm_mobility"] * strain_residuals
        )
    elif rrss == _fabric.RRSS_ENSTATITE:
        vol_dist_diff = (
            config["enstatite_fraction"] * config["gbm_mobility"] * strain_residuals
        )

    orientations_4 = rotation_rates * dt * strain_rate_max
    orientations_new = np.clip(
        orientations
        + (orientations_1 / 2 + orientations_2 + orientations_3 + orientations_4 / 2)
        / 3,
        -1,
        1,
    )
    vol_dist_4 = vol_dist_diff * dt * strain_rate_max
    vol_dist_new = (vol_dist_1 / 2 + vol_dist_2 + vol_dist_3 + vol_dist_4 / 2) / 3
    vol_dist_new /= vol_dist_new.sum()

    # Grain boundary sliding for small grains.
    mask = vol_dist_new < config["gbs_threshold"] / config["number_of_grains"]
    orientations_new[mask, :, :] = orientations[mask, :, :]
    vol_dist_new[mask, :, :] = config["gbs_threshold"] / config["number_of_grains"]
    vol_dist_new /= vol_dist_new.sum()

    return orientations_new, vol_dist_new


@nb.njit(fastmath=True)
def _get_deformation_rate(orientation, slip_rates, rrss, velocity_gradient):
    """Calculate deformation rate tensor and dimensionless slip rate.

    The dimensionless slip rate is for the softest slip system.

    """
    # TODO: 2d?
    deformation_rate = np.empty((3, 3))
    for j in range(3):
        for k in range(3):
            if rrss == _fabric.RRSS_OLIVINE_A:
                deformation_rate[j, k] = 2 * (
                    slip_rates[0] * orientation[0, j] * orientation[1, k]
                    + slip_rates[1] * orientation[0, j] * orientation[2, k]
                    + slip_rates[2] * orientation[2, j] * orientation[1, k]
                    + slip_rates[3] * orientation[2, j] * orientation[0, k]
                )
            elif rrss == _fabric.RRSS_ENSTATITE:
                deformation_rate[j, k] = 2 * orientation[2, j] * orientation[0, k]

    # Dimensionless strain rate on the softest slip system, see eq. 4 in Fraters 2021.
    enumerator = 0
    denominator = 0

    for j in range(3):
        # NOTE: Mistake in Kaminski 2001 eq. 7: j + 1, see Fraters & Billen 2021 S1.
        k = (j + 2) % 3
        enumerator -= (velocity_gradient[j, k] - velocity_gradient[k, j]) * (
            deformation_rate[j, k] - deformation_rate[k, j]
        )
        # NOTE: Mistake in Kaminski 2001 eq. 7: kl+1 instead of kk+1
        # See Fraters & Billen 2021 supplementary informaton S1.
        denominator -= (deformation_rate[j, k] - deformation_rate[k, j]) ** 2

        for L in range(3):
            enumerator += 2 * deformation_rate[j, L] * velocity_gradient[j, L]
            denominator += 2 * deformation_rate[j, L] ** 2

        slip_rate_softest = enumerator / denominator

    return deformation_rate, slip_rate_softest


@nb.jit(fastmath=True)
def _get_slip_rates(orientation, strain_rate, rrss, stress_exponent):
    """Calculate relative slip rates of the active slip systems.

    Returns a tuple containing the slip rates and indices that order the
    RRSS array by slip system activity.

    """
    # TODO: 2d?
    if rrss == _fabric.RRSS_OLIVINE_A:
        # Strain rate invariants for the four slip systems.
        invariants = np.zeros(4)
        for j in range(3):
            for k in range(3):
                invariants[0] += (
                    strain_rate[j, k] * orientation[0, j] * orientation[1, k]
                )
                invariants[1] += (
                    strain_rate[j, k] * orientation[0, j] * orientation[2, k]
                )
                invariants[2] += (
                    strain_rate[j, k] * orientation[2, j] * orientation[1, k]
                )
                invariants[3] += (
                    strain_rate[j, k] * orientation[2, j] * orientation[0, k]
                )

        i_inac, i_min, i_int, i_max = np.argsort(np.abs(invariants / rrss))
        # Ratio of slip rates on each slip system to slip rate on softest slip system.
        # Softest slip system has max. slip rate (aka activity).
        # See eq. 5, Kaminski 2001.
        prefactor = rrss[i_max] / invariants[i_max]
        ratio_min = prefactor * invariants[i_min] / rrss[i_min]
        ratio_int = prefactor * invariants[i_int] / rrss[i_int]
        slip_rates = np.empty(4)
        slip_rates[i_inac] = 0  # Hardest system is completely inactive in olivine.
        slip_rates[i_min] = ratio_min * np.abs(ratio_min) ** (stress_exponent - 1)
        slip_rates[i_int] = ratio_int * np.abs(ratio_int) ** (stress_exponent - 1)
        slip_rates[i_max] = 1
        slip_indices = (i_inac, i_min, i_int, i_max)
    elif rrss == _fabric.RRSS_ENSTATITE:
        slip_rates = np.repeat(np.nan, 4)
        slip_indices = np.argsort(rrss)
    return slip_rates, slip_indices


@nb.njit(fastmath=True)
def get_rotation_rates(
    orientations, strain_rate, velocity_gradient, rrss, stress_exponent
):
    """Calculate rotation rates for crystals undergoing dislocation creep.

    Args:
        `orientations` (NxDxD array) — present orientation matrices (direction cosines)
        `strain_rate` (DxD array) — dimensionless macroscopic strain rate tensor
        `velocity_gradient` (DxD array) — dimensionless velocity gradient tensor
        `rrss` (array) — reference resolved shear stress for each slip system
        `stress_exponent` (float) — exponent for the stress dependence of dislocation density

    N is the number of volume elements. D is the number of dimensions.
    Currently only supports 3D calculations.

    Returns a tuple containing:
    - the rotation rates (3D array)
    - the slip rates along each slip system

    See also equations 8, 9 & 14 in Kaminski 2001 [1] and equations 12 & 13 in Kaminski 2004 [2].

    [1]: https://doi.org/10.1016/S0012-821X(01)00356-9
    [2]: https://doi.org/10.1111/j.1365-246X.2004.02308.x

    """
    n_elements = orientations.shape[0]
    rotation_rates = np.empty((n_elements, 3, 3))
    for element_index in range(n_elements):
        # TODO: Make sure that orientations[element_index] is only a pointer, not a copy.
        slip_rates, slip_indices = _get_slip_rates(
            orientations[element_index],
            strain_rate,
            rrss,
            stress_exponent,
        )
        deformation_rate, slip_rate_softest = _get_deformation_rate(
            orientations[element_index],
            slip_rates,
            rrss,
            velocity_gradient,
        )

        # Spin vector for the volume element, see eq. 3 in Fraters 2021.
        spin_vector = np.empty(3)
        for j in range(3):
            r = (j + 1) % 3
            s = (j + 2) % 3
            spin_vector[j] = (
                velocity_gradient[s, r]
                - velocity_gradient[r, s]
                - (deformation_rate[s, r] - deformation_rate[r, s]) * slip_rate_softest
            ) / 2

        # Calculate rotation rate, see eq. 9 Kaminski & Ribe (2001).
        for p in range(3):
            for q in range(3):
                for r in range(3):
                    for s in range(3):
                        rotation_rates[element_index, p, q] += (
                            _tensors.PERMUTATION_SYMBOL[q, r, s]
                            * orientations[element_index, p, s]
                            * spin_vector[r]
                        )

    return rotation_rates, slip_rates, slip_indices, slip_rate_softest


@nb.njit(fastmath=True)
def get_strain_energy_residuals(
    vol_dist,
    slip_rates,
    slip_indices,
    slip_rate_softest,
    rrss,
    stress_exponent,
    nucleation_efficiency,
):
    """Calculate strain energy residual for each volume element.

    Args:
        `vol_dist` (NxD array) — present volume distribution (discretized)
        `slip_rates` (array) — slip rates relative to slip rate on softest slip system
        `slip_indices` (array) — indices that sort the RRSS by increasing slip rate activity
        `slip_rate_softest` (float) — slip rate on the softest (most active) slip system
        `stress_exponent` (float) — exponent for the stress dependence of dislocation density
        `nucleation_efficiency` (float) — parameter controlling grain nucleation

    """
    n_elements = len(vol_dist)
    strain_energies = np.zeros(n_elements)
    for element_index in range(n_elements):
        # Dimensionless dislocation density for each slip system.
        # See eq. 16 Fraters 2021.
        # NOTE: Mistake in eq. 11, Kaminski 2004: spurrious division by strain rate scale.
        if rrss == _fabric.RRSS_OLIVINE_A:
            for i in slip_indices[1:]:
                # TODO: Verify rrss[i] == τ_0 / τ^sv
                dislocation_density = rrss[i] ** (1.5 - stress_exponent) * np.abs(
                    slip_rates[i] * slip_rate_softest
                ) ** (1.5 / stress_exponent)
                # Dimensionless strain energy for this element, see eq. 14, Fraters 2021.
                strain_energies[element_index] += dislocation_density * np.exp(
                    -nucleation_efficiency * dislocation_density ** 2
                )
        elif rrss == _fabric.RRSS_ENSTATITE:
            # NOTE: Mistake in old PyDRex: x / y ** a != (x / y) ** a
            # See original DRex code, line 1437.
            # TODO: What is this? Why is it required?
            weight_factor = slip_rate_softest * rrss[slip_indices[0]] ** (
                -stress_exponent
            )
            # TODO: Verify rrss[i] == τ_0 / τ^sv
            dislocation_density = rrss[i] ** (1.5 - stress_exponent) * np.abs(
                weight_factor
            ) ** (1.5 / stress_exponent)
            # Dimensionless strain energy for this element, see eq. 14, Fraters 2021.
            strain_energies[element_index] = dislocation_density * np.exp(
                -nucleation_efficiency * dislocation_density ** 2
            )

    # Volume average mean strain energy.
    mean_energy = np.sum(vol_dist * strain_energies)
    # Strain energy residual.
    return mean_energy - strain_energies
