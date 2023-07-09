r"""> PyDRex: Core D-ReX functions and enums.

The function `derivatives` implements the core DRex solver, which computes the
crystallographic rotation rate and changes in fractional grain volumes.

"""
from enum import IntEnum, unique

import numba as nb
import numpy as np

# NOTE: Do NOT import any pydrex submodules here to avoid cyclical imports.


PERMUTATION_SYMBOL = np.array(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)


@unique
class MineralPhase(IntEnum):
    """Supported mineral phases."""

    olivine = 0
    enstatite = 1


@unique
class DeformationRegime(IntEnum):
    """Deformation mechanism regimes."""

    diffusion = 0
    dislocation = 1
    byerlee = 2
    max_viscosity = 3


@unique
class MineralFabric(IntEnum):
    """Supported mineral fabrics.

    The following fabric types are supported:
    - olivine A-E type fabrics according to e.g.
      [Karato et al., 2008](https://doi.org/10.1146%2Fannurev.earth.36.031207.124120).
    - enstatite AB fabric, see
      [Bernard et al., 2021](https://doi.org/10.1016/j.tecto.2021.228954).

    """

    olivine_A = 0
    olivine_B = 1
    olivine_C = 2
    olivine_D = 3
    olivine_E = 4
    enstatite_AB = 5


@nb.njit
def get_crss(phase, fabric):
    """Get Critical Resolved Shear Stress for the mineral `phase` and `fabric`.

    Returns an array of the normalised threshold stresses required to activate slip on
    each slip system. Olivine slip systems are ordered according to the convention used
    for `OLIVINE_SLIP_SYSTEMS`.

    """
    if phase == MineralPhase.olivine:
        match fabric:
            case MineralFabric.olivine_A:
                return np.array([1, 2, 3, np.inf])
            case MineralFabric.olivine_B:
                return np.array([3, 2, 1, np.inf])
            case MineralFabric.olivine_C:
                return np.array([3, 2, np.inf, 1])
            case MineralFabric.olivine_D:
                return np.array([1, 1, 3, np.inf])
            case MineralFabric.olivine_E:
                return np.array([3, 1, 2, np.inf])
            case _:
                raise ValueError("unsupported olivine fabric")
    elif phase == MineralPhase.enstatite:
        if fabric == MineralFabric.enstatite_AB:
            return np.array([np.inf, np.inf, np.inf, 1])
            raise ValueError("unsupported enstatite fabric")
    raise ValueError("phase must be a valid `MineralPhase`")


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
    deformation_exponent,
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
    - `fractions` (array) — volume fractions of the "grains" relative to aggregate volume
    - `strain_rate` (array) — 3x3 dimensionless macroscopic strain-rate tensor
    - `velocity_gradient` (array) — 3x3 dimensionless macroscopic velocity gradient tensor
    - `stress_exponent` (float) — value of `p` for `dislocation_density ∝ shear_stress^p`
    - `deformation_exponent` (float) — value of `n` for `shear_stress ∝ |deformation_rate|^(1/n)`
    - `nucleation_efficiency` (float) — parameter controlling grain nucleation
    - `gmb_mobility` (float) — grain boundary mobility parameter
    - `volume_fraction` (float) — volume fraction of the mineral phase relative to
                                  other phases
    .. warning:: Raises zero-division errors if the vorticity is zero.

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
            deformation_exponent,
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
            if phase == MineralPhase.olivine:
                deformation_rate[i, j] = 2 * (
                    slip_rates[0] * orientation[0, i] * orientation[1, j]
                    + slip_rates[1] * orientation[0, i] * orientation[2, j]
                    + slip_rates[2] * orientation[2, i] * orientation[1, j]
                    + slip_rates[3] * orientation[2, i] * orientation[0, j]
                )
            elif phase == MineralPhase.enstatite:
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
def _get_slip_rates_olivine(invariants, slip_indices, crss, deformation_exponent):
    """Calculate relative slip rates of the active slip systems for olivine.

    Args:
    - `invariants` (array) — strain rate invariants for the four slip systems
    - `slip_indices` (array) — indices that sort the CRSS by increasing slip-rate
                               activity
    - `crss` (array) — reference resolved shear stresses (CRSS), see `pydrex.fabric`
    - `deformation_exponent` (float) — value of `n` for `shear_stress ∝ |deformation_rate|^(1/n)`

    """
    i_inac, i_min, i_int, i_max = slip_indices
    # Ratio of slip rates on each slip system to slip rate on softest slip system.
    # Softest slip system has max. slip rate (aka activity).
    # See eq. 5, Kaminski 2001.
    prefactor = crss[i_max] / invariants[i_max]
    ratio_min = prefactor * invariants[i_min] / crss[i_min]
    ratio_int = prefactor * invariants[i_int] / crss[i_int]
    slip_rates = np.empty(4)
    slip_rates[i_inac] = 0  # Hardest system is completely inactive in olivine.
    slip_rates[i_min] = ratio_min * np.abs(ratio_min) ** (deformation_exponent - 1)
    slip_rates[i_int] = ratio_int * np.abs(ratio_int) ** (deformation_exponent - 1)
    slip_rates[i_max] = 1
    return slip_rates


@nb.njit(fastmath=True)
def _get_slip_invariants_olivine(strain_rate, orientation):
    r"""Calculate strain rate invariants for the four slip systems of olivine.

    Calculates $I = ∑_{ij} l_{i} n_{j} \dot{ε}_{ij}$ for each slip sytem.

    Args:
    - `strain_rate` (array) — 3x3 dimensionless strain rate matrix
    - `orientation` (array) — 3x3 orientation matrix (direction cosines)

    """
    invariants = np.zeros(4)
    for i in range(3):
        for j in range(3):
            # (010)[100]
            invariants[0] += strain_rate[i, j] * orientation[0, i] * orientation[1, j]
            # (001)[100]
            invariants[1] += strain_rate[i, j] * orientation[0, i] * orientation[2, j]
            # (010)[001]
            invariants[2] += strain_rate[i, j] * orientation[2, i] * orientation[1, j]
            # (100)[001]
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
                        PERMUTATION_SYMBOL[q, r, s] * orientation[p, s] * spin_vector[r]
                    )

    return orientation_change


# TODO: Why is enstatite different and can the logic be merged better?
@nb.njit(fastmath=True)
def _get_strain_energy_olivine(
    crss,
    slip_rates,
    slip_indices,
    slip_rate_softest,
    stress_exponent,
    deformation_exponent,
    nucleation_efficiency,
):
    """Calculate strain energy due to dislocations for an olivine grain.

    Args:
    - `crss` (array) — reference resolved shear stresses (CRSS), see `pydrex.fabric`
    - `slip_rates` (array) — slip rates relative to slip rate on softest slip system
    - `slip_indices` (array) — indices that sort the CRSS by increasing slip-rate
                               activity
    - `slip_rate_softest` (float) — slip rate on the softest (most active) slip system
    - `stress_exponent` (float) — value of `p` for `dislocation_density ∝ shear_stress^p`
    - `deformation_exponent` (float) — value of `n` for `shear_stress ∝ |deformation_rate|^(1/n)`
    - `nucleation_efficiency` (float) — parameter controlling grain nucleation

    Note that "new" grains are assumed to rotate with their parent.

    """
    strain_energy = 0.0
    # Dimensionless dislocation density for each slip system.
    # See eq. 16 Fraters 2021.
    # NOTE: Mistake in eq. 11, Kaminski 2004: spurious division by strain rate scale.
    for i in range(3):
        dislocation_density = (1 / crss[i]) ** (
            deformation_exponent - stress_exponent
        ) * np.abs(slip_rates[i] * slip_rate_softest) ** (
            stress_exponent / deformation_exponent
        )
        # Dimensionless strain energy for this grain, see eq. 14, Fraters 2021.
        strain_energy += dislocation_density * np.exp(
            -nucleation_efficiency * dislocation_density**2
        )
    return strain_energy


@nb.njit(fastmath=True)
def _get_strain_energy_enstatite(
    crss,
    slip_indices,
    slip_rate_softest,
    stress_exponent,
    deformation_exponent,
    nucleation_efficiency,
):
    """Calculate strain energy due to dislocations for an enstatite grain.

    Args:
    - `crss` (array) — reference resolved shear stresses (CRSS), see `pydrex.fabric`
    - `slip_indices` (array) — indices that sort the CRSS by increasing slip-rate activity
    - `slip_rate_softest` (float) — slip rate on the softest (most active) slip system
    - `stress_exponent` (float) — value of `p` for `dislocation_density ∝ shear_stress^p`
    - `deformation_exponent` (float) — value of `n` for `shear_stress ∝ |deformation_rate|^(1/n)`
    - `nucleation_efficiency` (float) — parameter controlling grain nucleation

    """
    weight_factor = slip_rate_softest / crss[slip_indices[-1]] ** stress_exponent
    dislocation_density = (1 / crss[slip_indices[-1]]) ** (
        deformation_exponent - stress_exponent
    ) * np.abs(weight_factor) ** (stress_exponent / deformation_exponent)
    # Dimensionless strain energy for this grain, see eq. 14, Fraters 2021.
    return dislocation_density * np.exp(
        -nucleation_efficiency * dislocation_density**2
    )


# NOTE: Requires explicit Numba signature and must be defined last,
# otherwise Numba 0.57.1 is not happy when CPO for multiple phases is calculated.
# @nb.njit(
#     "Tuple((f8[:,::1], f8))(i8, i8, f8[:,::1], f8[:,::1], f8[:,::1], f8, f8, i8)", fastmath=True
# )
@nb.njit
def _get_rotation_and_strain(
    phase,
    fabric,
    orientation,
    strain_rate,
    velocity_gradient,
    stress_exponent,
    deformation_exponent,
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
    - `stress_exponent` (float) — value of `p` for `dislocation_density ∝ shear_stress^p`
    - `deformation_exponent` (float) — value of `n` for `shear_stress ∝ |deformation_rate|^(1/n)`
    - `nucleation_efficiency (float) — parameter controlling grain nucleation

    Note that "new" grains are assumed to rotate with their parent.

    .. warning:: Raises zero-division errors if the vorticity is zero.

    Returns a tuple with the rotation rate of the crystalline axes
    with respect to the principal strain axes and strain energy of the grain.

    """
    crss = get_crss(phase, fabric)
    if phase == MineralPhase.olivine:
        slip_invariants = _get_slip_invariants_olivine(strain_rate, orientation)
        slip_indices = np.argsort(np.abs(slip_invariants / crss))
        slip_rates = _get_slip_rates_olivine(
            slip_invariants,
            slip_indices,
            crss,
            deformation_exponent,
        )
    elif phase == MineralPhase.enstatite:
        slip_indices = np.argsort(1 / crss)
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
    if phase == MineralPhase.olivine:
        strain_energy = _get_strain_energy_olivine(
            crss,
            slip_rates,
            slip_indices,
            slip_rate_softest,
            stress_exponent,
            deformation_exponent,
            nucleation_efficiency,
        )
    elif phase == MineralPhase.enstatite:
        strain_energy = _get_strain_energy_enstatite(
            crss,
            slip_indices,
            slip_rate_softest,
            stress_exponent,
            deformation_exponent,
            nucleation_efficiency,
        )
    else:
        assert False  # Should never happen.
    return orientation_change, strain_energy
