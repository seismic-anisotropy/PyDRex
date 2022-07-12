r"""PyDRex: Core DRex functions.

CPO can be calculated along a flow pathline, using interpolated velocity values,
or for a Lagrangian particle experiencing a given velocity gradient.

WARNING: It is recommended to use the `Mineral` class from `pydrex.minerals`
instead of using these routines directly, which do not circumvent all edge cases.
For example, the pathological case with a flow field with zero vorticity will crash.

"""
import numba as nb
import numpy as np

import pydrex.minerals as _minerals
import pydrex.tensors as _tensors


# 12 args is a lot, but this way we can use numba
# (only primitives and numpy containers allowed).
@nb.njit(fastmath=True)
def derivatives(
    phase,
    fabric,
    n_grains,
    orientations,
    fractions,
    strain_rate,
    velocity_gradient,
    stress_exponent,
    dislocation_exponent,
    nucleation_efficiency,
    gbm_mobility,
    volume_fraction,
):
    """Get derivatives of orientation and volume distribution.

    Args:
    - `phase` (int) — ordinal number of the mineral phase
                      see `pydrex.minerals.MineralPhase`
    - `fabric` (int) — ordinal number of the fabric type, see `pydrex.fabric`
    - `n_grains` (int) — number of "grains" i.e. discrete volume segments
    - `orientations` (array) — `n_grains`x3x3 orientations (direction cosines)
    - `fractions` (array) — volume fractions of the "grains" relative to aggregate
                            volume
    - `strain_rate` (array) — 3x3 dimensionless macroscopic strain-rate tensor
    - `velocity_gradient` (array) — 3x3 dimensionless macroscopic velocity gradient
                                    tensor
    - `stress_exponent` (float) — value of `n` for `shear_stress ∝ |strain_rate|^(1/n)`
    - `dislocation_exponent` (float) — value of `p` for
                                       `dislocation_density ∝ shear_stress^p`
    - `nucleation_efficiency` (float) — parameter controlling grain nucleation
    - `gmb_mobility` (float) — grain boundary mobility parameter
    - `volume_fraction` (float) — volume fraction of the mineral phase relative to
                                  other phases

    WARNING: Raises zero-division errors if the vorticity is zero.

    Returns a tuple with the rotation rates and grain volume fraction changes.

    """
    strain_energies = np.empty(n_grains)
    orientations_diff = np.empty((n_grains, 3, 3))
    # TODO: Make sure that orientations[grain_index] is only a pointer, not a copy.
    for grain_index in range(n_grains):
        orientation_change, strain_energy = _get_rotation_and_strain(
            phase,
            fabric,
            orientations[grain_index],
            strain_rate,
            velocity_gradient,
            stress_exponent,
            dislocation_exponent,
            nucleation_efficiency,
        )
        orientations_diff[grain_index] = orientation_change
        strain_energies[grain_index] = strain_energy
    # Volume average mean strain energy.
    mean_energy = np.sum(fractions * strain_energies)
    # Strain energy residual.
    strain_residuals = mean_energy - strain_energies
    fractions_diff = volume_fraction * gbm_mobility * fractions * strain_residuals
    return orientations_diff, fractions_diff


@nb.njit(fastmath=True)
def _get_rotation_and_strain(
    phase,
    fabric,
    orientation,
    strain_rate,
    velocity_gradient,
    stress_exponent,
    dislocation_exponent,
    nucleation_efficiency,
):
    """Get the crystal axes rotation rate and strain energy of individual grain.

    Args:
    - `phase` (int) — ordinal number of the mineral phase
                      see `pydrex.minerals.MineralPhase`
    - `fabric` (int) — ordinal number of the fabric type, see `pydrex.fabric`
    - `orientation` (array) — 3x3 orientation matrix (direction cosines)
    - `strain_rate` (array) — 3x3 dimensionless strain rate matrix
    - `velocity_gradient` (array) — 3x3 dimensionless velocity gradient matrix
    - `nucleation_efficiency (float) — parameter controlling grain nucleation

    Note that "new" grains are assumed to rotate with their parent.

    WARNING: Raises zero-division errors if the vorticity is zero.

    Returns a tuple with the rotation rate of the crystalline axes
    with respect to the principal strain axes and strain energy of the grain.

    """
    rrss = _minerals.get_rrss(phase, fabric)
    if phase == _minerals.MineralPhase.olivine:
        slip_invariants = _get_slip_invariants_olivine(strain_rate, orientation)
        slip_indices = np.argsort(np.abs(slip_invariants / rrss))
        slip_rates = _get_slip_rates_olivine(
            slip_invariants,
            slip_indices,
            rrss,
            stress_exponent,
        )
    elif phase == _minerals.MineralPhase.enstatite:
        slip_indices = np.argsort(1 / rrss)
        slip_rates = np.repeat(np.nan, 4)
    else:
        assert False  # Should never happen.

    deformation_rate = _get_deformation_rate(phase, orientation, slip_rates)
    slip_rate_softest = _get_slip_rate_softest(deformation_rate, velocity_gradient)
    orientation_change = _get_orientation_change(
        orientation,
        velocity_gradient,
        deformation_rate,
        slip_rate_softest,
    )
    if phase == _minerals.MineralPhase.olivine:
        strain_energy = _get_strain_energy_olivine(
            rrss,
            slip_rates,
            slip_indices,
            slip_rate_softest,
            stress_exponent,
            dislocation_exponent,
            nucleation_efficiency,
        )
    elif phase == _minerals.MineralPhase.enstatite:
        strain_energy = _get_strain_energy_enstatite(
            rrss,
            slip_indices,
            slip_rate_softest,
            stress_exponent,
            dislocation_exponent,
            nucleation_efficiency,
        )
    else:
        assert False  # Should never happen.
    return orientation_change, strain_energy


@nb.njit(fastmath=True)
def _get_deformation_rate(phase, orientation, slip_rates):
    """Calculate deformation rate tensor for olivine or enstatite.

    Calculate the deformation rate with respect to the local coordinate frame,
    defined by the principal strain axes (finite strain ellipsoid).

    Args:
    - `phase` (int) — ordinal number of the mineral phase
                      see `pydrex.minerals.MineralPhase`
    - `orientation` (array) — 3x3 orientation matrix (direction cosines)
    - `slip_rates` (array) — slip rates relative to slip rate on softest slip system

    """
    deformation_rate = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            if phase == _minerals.MineralPhase.olivine:
                deformation_rate[i, j] = 2 * (
                    slip_rates[0] * orientation[0, i] * orientation[1, j]
                    + slip_rates[1] * orientation[0, i] * orientation[2, j]
                    + slip_rates[2] * orientation[2, i] * orientation[1, j]
                    + slip_rates[3] * orientation[2, i] * orientation[0, j]
                )
            elif phase == _minerals.MineralPhase.enstatite:
                deformation_rate[i, j] = 2 * orientation[2, i] * orientation[0, j]
            else:
                assert False  # Should never happen.
    return deformation_rate


@nb.njit(fastmath=True)
def _get_slip_rate_softest(deformation_rate, velocity_gradient):
    """Calculate dimensionless strain rate on the softest slip system.

    Args:
    - `deformation_rate` (array) — 3x3 dimensionless deformation rate matrix
    - `velocity_gradient` (array) — 3x3 dimensionless velocity gradient matrix

    """
    # See eq. 4 in Fraters 2021.
    enumerator = 0
    denominator = 0

    for j in range(3):
        # NOTE: Mistake in original DRex code (j + 2), see Fraters & Billen 2021 S1.
        k = (j + 1) % 3
        enumerator -= (velocity_gradient[j, k] - velocity_gradient[k, j]) * (
            deformation_rate[j, k] - deformation_rate[k, j]
        )
        # NOTE: Mistake in Kaminski 2001 eq. 7: kl+1 instead of kk+1
        # See Fraters & Billen 2021 supplementary informaton S1.
        denominator -= (deformation_rate[j, k] - deformation_rate[k, j]) ** 2

        for L in range(3):
            enumerator += 2 * deformation_rate[j, L] * velocity_gradient[j, L]
            denominator += 2 * deformation_rate[j, L] ** 2

    return enumerator / denominator


@nb.njit(fastmath=True)
def _get_slip_rates_olivine(invariants, slip_indices, rrss, stress_exponent):
    """Calculate relative slip rates of the active slip systems for olivine.

    Args:
    - `invariants` (array) — strain rate invariants for the four slip systems
    - `slip_indices` (array) — indices that sort the RRSS by increasing slip-rate
                               activity
    - `rrss` (array) — reference resolved shear stresses (RRSS), see `pydrex.fabric`
    - `stress_exponent` (float) — exponent for the stress dependence of dislocation
                                  density

    """
    i_inac, i_min, i_int, i_max = slip_indices
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
    return slip_rates


@nb.njit(fastmath=True)
def _get_slip_invariants_olivine(strain_rate, orientation):
    r"""Calculate strain rate invariants for the four slip systems of olivine.

    Calculates $I_{ij} = ∑_{ij} l_{i} n_{j} \dot{ε}_{ij}$ for each slip sytem.

    Args:
    - `strain_rate` (array) — 3x3 dimensionless strain rate matrix
    - `orientation` (array) — 3x3 orientation matrix (direction cosines)

    """
    invariants = np.zeros(4)
    for i in range(3):
        for j in range(3):
            invariants[0] += strain_rate[i, j] * orientation[0, i] * orientation[1, j]
            invariants[1] += strain_rate[i, j] * orientation[0, i] * orientation[2, j]
            invariants[2] += strain_rate[i, j] * orientation[2, i] * orientation[1, j]
            invariants[3] += strain_rate[i, j] * orientation[2, i] * orientation[0, j]
    return invariants


@nb.njit(fastmath=True)
def _get_orientation_change(
    orientation, velocity_gradient, deformation_rate, slip_rate_softest
):
    """Calculate the rotation rate for a grain undergoing dislocation creep.

    Args:
    - `orientation` (array) — 3x3 orientation matrix (direction cosines)
    - `velocity_gradient` (array) — 3x3 dimensionless velocity gradient matrix
    - `deformation_rate` (float) — 3x3 dimensionless strain rate matrix
    - `slip_rate_softest` (float) — slip rate on the softest (most active) slip system

    """
    orientation_change = np.zeros((3, 3))
    # Spin vector for the grain, see eq. 3 in Fraters 2021.
    spin_vector = np.empty(3)
    for j in range(3):
        r = (j + 1) % 3
        s = (j + 2) % 3
        spin_vector[j] = (
            (velocity_gradient[s, r] - velocity_gradient[r, s])
            - (deformation_rate[s, r] - deformation_rate[r, s]) * slip_rate_softest
        ) / 2

    # Calculate rotation rate, see eq. 9 Kaminski & Ribe (2001).
    for p in range(3):
        for q in range(3):
            for r in range(3):
                for s in range(3):
                    orientation_change[p, q] += (
                        _tensors.PERMUTATION_SYMBOL[q, r, s]
                        * orientation[p, s]
                        * spin_vector[r]
                    )

    return orientation_change


# TODO: Why is enstatite different and can the logic be merged better?
@nb.njit(fastmath=True)
def _get_strain_energy_olivine(
    rrss,
    slip_rates,
    slip_indices,
    slip_rate_softest,
    stress_exponent,
    dislocation_exponent,
    nucleation_efficiency,
):
    """Calculate strain energy due to dislocations for an olivine grain.

    Args:
    - `rrss` (array) — reference resolved shear stresses (RRSS), see `pydrex.fabric`
    - `slip_rates` (array) — slip rates relative to slip rate on softest slip system
    - `slip_indices` (array) — indices that sort the RRSS by increasing slip-rate
                               activity
    - `slip_rate_softest` (float) — slip rate on the softest (most active) slip system
    - `stress_exponent` (float) — value of `n` for `shear_stress ∝ |strain_rate|^(1/n)`
    - `dislocation_exponent` (float) — value of `p` for
                                       `dislocation_density ∝ shear_stress^p`
    - `nucleation_efficiency` (float) — parameter controlling grain nucleation

    Note that "new" grains are assumed to rotate with their parent.

    """
    strain_energy = 0.0
    # Dimensionless dislocation density for each slip system.
    # See eq. 16 Fraters 2021.
    # NOTE: Mistake in eq. 11, Kaminski 2004: spurrious division by strain rate scale.
    for i in slip_indices[1:]:
        # TODO: Verify rrss[i] == τ_0 / τ^sv
        dislocation_density = rrss[i] ** (
            dislocation_exponent - stress_exponent
        ) * np.abs(slip_rates[i] * slip_rate_softest) ** (
            dislocation_exponent / stress_exponent
        )
        # Dimensionless strain energy for this grain, see eq. 14, Fraters 2021.
        strain_energy += dislocation_density * np.exp(
            -nucleation_efficiency * dislocation_density**2
        )
    return strain_energy


@nb.njit(fastmath=True)
def _get_strain_energy_enstatite(
    rrss,
    slip_indices,
    slip_rate_softest,
    stress_exponent,
    dislocation_exponent,
    nucleation_efficiency,
):
    """Calculate strain energy due to dislocations for an enstatite grain.

    Args:
    - `rrss` (array) — reference resolved shear stresses (RRSS), see `pydrex.fabric`
    - `slip_indices` (array) — indices that sort the RRSS by increasing slip-rate
                               activity
    - `slip_rate_softest` (float) — slip rate on the softest (most active) slip system
    - `stress_exponent` (float) — value of `n` for `shear_stress ∝ |strain_rate|^(1/n)`
    - `dislocation_exponent` (float) — value of `p` for
                                       `dislocation_density ∝ shear_stress^p`
    - `nucleation_efficiency` (float) — parameter controlling grain nucleation

    """
    weight_factor = slip_rate_softest / rrss[slip_indices[-1]] ** stress_exponent
    dislocation_density = rrss[slip_indices[-1]] ** (
        dislocation_exponent - stress_exponent
    ) * np.abs(weight_factor) ** (dislocation_exponent / stress_exponent)
    # Dimensionless strain energy for this grain, see eq. 14, Fraters 2021.
    return dislocation_density * np.exp(
        -nucleation_efficiency * dislocation_density**2
    )
