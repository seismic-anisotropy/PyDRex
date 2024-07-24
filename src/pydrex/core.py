"""> PyDRex: Core D-Rex functions and enums.

The function `derivatives` implements the core D-Rex solver, which computes the
crystallographic rotation rate and changes in fractional grain volumes.

**Acronyms:**
- CRSS = Critical Resolved Shear Stress,
    i.e. threshold stress required to initiate slip on a slip system,
    normalised to the stress required to initiate slip on the softest slip system

"""

from dataclasses import asdict, dataclass
from enum import IntEnum, unique

import numba as nb
import numpy as np

# NOTE: Do NOT import any pydrex submodules here to avoid cyclical imports.


_USE_ORIGINAL_DREX = False  # Switch to use D-Rex 2004 version without corrections.


PERMUTATION_SYMBOL = np.array(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)
"""Sometimes called the Levi-Civita symbol."""


@unique
class MineralPhase(IntEnum):
    """Supported mineral phases.

    Forsterite and fayalite are grouped into “olivine”, because we treat them as
    rheologically equivalent.

    """

    olivine = 0
    """(Mg,Fe)₂SiO₄"""
    enstatite = 1
    """MgSiO₃"""


@unique
class MineralFabric(IntEnum):
    """Supported mineral fabrics.

    The following fabric types are supported:
    - olivine A-E type fabrics according to e.g.
      [Karato et al. (2008)](https://doi.org/10.1146%2Fannurev.earth.36.031207.124120).
    - enstatite AB fabric, see
      [Bernard et al. (2021)](https://doi.org/10.1016/j.tecto.2021.228954).

    """

    olivine_A = 0
    olivine_B = 1
    olivine_C = 2
    olivine_D = 3
    olivine_E = 4
    enstatite_AB = 5


@unique
class DeformationRegime(IntEnum):
    r"""Ordinals to track distinct regimes of dominant deformation mechanisms.

    The mechanism of deformation that dominates accommodation of plastic deformation
    depends in general both on material properties such as grain size and mineral phase
    content as well as on thermodynamic properties such as temperature, pressure and
    water fugacity.

    The activity of diffusive mechanisms depends more strongly on grain size, whereas
    that of dislocation mechanisms depends more strongly on temperature. High
    temperatures enable more frequent recovery of dislocation density equilibrium via
    e.g. dislocation climb. Dislocation mechanisms are often accompanied by dynamic
    recrystallisation, which acts as an additional recovery mechanism.

    Diffusive mechanisms are expected, however, to become dominant at depth, as these
    mechanisms are primarily facilitated by Si diffusion in olivine, which is
    temperature-dependent.

    Rheology in the intra-granular dislocation regime was classically described by
    separate flow laws depending on temperature; a power-law at high temperature
    [$\dot{ε} ∝ σⁿ$] and an exponential-law at low temperature [$\dot{ε} ∝ \exp(σ)$].
    More recent work has suggested unified dislocation creep flow laws,
    e.g. [Gouriet et al. 2019](http://dx.doi.org/10.1016/j.epsl.2018.10.049),
    [Garel et al. 2020](http://dx.doi.org/10.1016/j.epsl.2020.116243) and
    [Demouchy et al. 2023](http://dx.doi.org/10.2138/gselements.19.3.151).

    .. note:: Although a draft texture evolution behaviour is implemented in the
        `frictional_yielding` regime, it is experimental and not yet configurable via
        the parameter interface.

    """

    min_viscosity = 0
    """Arbitrary lower-bound viscosity regime."""
    matrix_diffusion = 1
    """Intra-granular Nabarro-Herring creep, i.e. grains diffuse through the matrix."""
    boundary_diffusion = 2
    """Inter-granular Coble creep, i.e. grains diffuse along grain boundaries."""
    sliding_diffusion = 3
    """Inter-granular diffusion-assisted grain-boundary sliding (diffGBS)."""
    matrix_dislocation = 4
    """Intra-granular dislocation creep (glide + climb) and dynamic recrystallisation."""
    sliding_dislocation = 5
    """Inter-granular dislocation-assisted grain-boundary sliding (disGBS)."""
    frictional_yielding = 6
    """Frictional sliding along micro-fractures (Byerlee's law for yield strength)."""
    max_viscosity = 7
    """Arbitrary upper-bound viscosity regime."""


@dataclass(frozen=True)
class DefaultParams:
    phase_assemblage: tuple = (MineralPhase.olivine,)
    """Mineral phases present in the aggregate."""
    phase_fractions: tuple = (1.0,)
    """Volume fractions of each mineral phase present in the aggregate."""
    stress_exponent: float = 1.5
    """The value for $p$ in $ρ ∝ τᵖ$ where $ρ$ is the dislocation density and $τ$ the shear stress.

    Default value taken from [Kaminski et al. 2004](https://doi.org/10.1111/j.1365-246X.2004.02308.x)
    based on studies referenced therein.

    """
    deformation_exponent: float = 3.5
    """The value for $n$ in $τ ∝ |D|^{1/n}$ where $τ$ is the shear stress and D the deformation rate.

    Default value taken from [Kaminski et al. 2004](https://doi.org/10.1111/j.1365-246X.2004.02308.x)
    based on studies referenced therein.

    """
    gbm_mobility: int = 125
    """Dimensionless grain boundary mobility parameter (M*).

    This controls the rate of all dynamic recrystallisation processes in
    `DeformationRegime.matrix_dislocation`.

    .. note:: This parameter is not easy to constrain. Reasonable values may lie in the
        range [10, 150]. Comparing outputs for multiple M* values is recommended.

    """
    gbs_threshold: float = 0.3
    """Grain boundary sliding threshold.

    In `DeformationRegime.matrix_dislocation` or `DeformationRegime.sliding_dislocation`
    this controls the smallest size of a grain, relative to its original size,
    at which it will still deform via dislocation creep.
    Smaller grains will instead deform via grain boundary sliding,
    therefore not contributing to any further texture evolution.

    .. note:: Values for this parameter do NOT correspond to a physical grain size
        threshold. Grains in PyDRex are a surrogate numerical discretisation. There are
        likely to be feedbacks between this parameter and `number_of_grains`. Values of
        ~0.3 have been used in numerous studies which employed ~3000 grains.

    """
    nucleation_efficiency: float = 5.0
    """Dimensionless nucleation efficiency (λ*).

    This controls the nucleation of subgrains in `DeformationRegime.matrix_dislocation`.

    The default value comes from [Kaminski & Ribe 2001](https://doi.org/10.1016%2Fs0012-821x%2801%2900356-9).

    """
    number_of_grains: int = 3500
    """Number of surrogate grains for numerical discretisation of the aggregate.

    This is a numerical discretisation, so generally more grains is better. However,
    there is a sharp tradeoff in performance, especially for M-index calculation.

    """
    initial_olivine_fabric: MineralFabric = MineralFabric.olivine_A
    """Olivine fabric (CRSS distribution) at the beginning of the simulation.

    In general, the fabric should be allowed to change during the simulation,
    but that is not yet implemented.

    """
    disl_Peierls_stress: float = 2
    """Stress barrier in GPa for activation of dislocation motion at low temperatures.

    - 2GPa suggested by [Demouchy et al. 2023](http://dx.doi.org/10.2138/gselements.19.3.151)

    .. note:: Not relevant if the unified dislocation creep flow law is used.

    """
    # NOTE: For now, we just roll this into the prefactors.
    # water_fugacity: float = 1.5
    # """Approximate constant water fugacity of the aggregate (GPa)."""
    # TODO: Check units of Garel 2020 pre-exp value below.
    # TODO: Find and add references for the other value.
    disl_prefactors: tuple = (1e-16, 1e-17)
    """Prefactors for dislocation creep exponential & power laws (s⁻¹).

    - B = 1.7e-16 s⁻¹ for the exponential law suggested by [Demouchy et al. 2023](http://dx.doi.org/10.2138/gselements.19.3.151)
    - pre-exponential factor of 4.4e-17 Pa⁻ⁿs⁻¹ used for the exponential law by [Garel et al. 2020](http://dx.doi.org/10.1016/j.epsl.2020.116243)ᵃ

    ᵃCheck units, why is the Pa⁻ⁿ in there...

    """
    # TODO: Add references, tweak default values if necessary.
    diff_prefactors: tuple = (1e-10, 1e-10)
    r"""Prefactors for (matrix and boundary) diffusion creep power laws (s⁻¹).

    Dependence on molar volume and physical grain size are suppressed because these
    diagnostics are not readily available. The prefactor is roughly equal to
    $(Vₘ D⁰)/d²$ for $Vₘ$ the molar volume, $D⁰$ the reference diffusion coefficient and
    $d²$ an average grain size assumed to be constant.

    """
    disl_lowtemp_switch: float = 0.7
    """Threshold homologous temperature below which to use the exponential flow law.

    The default value suggested here comes from
    [Demouchy et al. 2023](http://dx.doi.org/10.2138/gselements.19.3.151).

    .. note:: Not relevant if the unified dislocation creep flow law is used instead.

    """
    # TODO: Add more references from experimental studies/reviews.
    disl_activation_energy: float = 460.0
    """Activation energy for dislocation creep power law (kJ/mol).

    - 443 kJ/mol used by [Gouriet et al. 2019](http://dx.doi.org/10.1016/j.epsl.2018.10.049),
        but for an exponential law
    - 540 kJ/mol used by [Garel et al. 2020](http://dx.doi.org/10.1016/j.epsl.2020.116243)

    """
    # TODO: Add more references, tweak default if necessary.
    disl_activation_volume: float = 12.0
    """Activation volume for dislocation creep power law (cm³/mol).

    - 0, 6 and 12 cm³/mol used by [Garel et al. 2020](http://dx.doi.org/10.1016/j.epsl.2020.116243)

    """
    # TODO: Add more references, justify choice of lower value than Garel 2020.
    diff_activation_energies: tuple = (430.0, 330)
    """Activation energies for (matrix and boundary) diffusion creep power laws (kJ/mol).

    - 530 kJ/mol reported for Si self-diffusion in olivine¹
    - 410 kJ/mol used by [Garel et al. 2020](http://dx.doi.org/10.1016/j.epsl.2020.116243)

    ¹[Dohmen et al. 2002](http://dx.doi.org/10.1029/2002GL015480)

    """
    # TODO: Add more references, tweak default values if necessary.
    diff_activation_volumes: tuple = (4.0, 4.0)
    """Activation volumes for (matrix and boundary) diffusion creep power laws (cm³/mol).

    - 4 kJ/mol used by [Garel et al. 2020](http://dx.doi.org/10.1016/j.epsl.2020.116243)

    """
    disl_coefficients: tuple = (
        4.4e8,  # a₀
        -5.26e4,  # b₀
        2.11e-2,  # a₁
        1.74e-4,  # b₁
        -41.8,  # a₂
        4.21e-2,  # b₂
        -1.14e-5,  # c₂
    )
    """Coefficients for polynomials used in the alternative dislocation creep flow law.

    The defaults are for the TANH variant of the unified creep law, proposed in
    [Garel et al. 2020](http://dx.doi.org/10.1016/j.epsl.2020.116243).
    By contrast, [Gouriet et al. 2019](http://dx.doi.org/10.1016/j.epsl.2018.10.049)
    used the ERF variant with: [4.4e8, -2.2e4, 3e-2, 1.3e-4, -42, 4.2e-2, -1.1e-5].

    """

    def as_dict(self):
        """Return mutable copy of default arguments as a dictionary."""
        return asdict(self)


@nb.njit
def get_crss(phase: MineralPhase, fabric: MineralFabric) -> np.ndarray:
    """Get Critical Resolved Shear Stress for the mineral `phase` and `fabric`.

    Returns an array of the normalised threshold stresses required to activate slip on
    each slip system. Olivine slip systems are ordered according to the convention used
    for `pydrex.minerals.OLIVINE_SLIP_SYSTEMS`.

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
                raise ValueError(f"unsupported olivine fabric: {fabric}")
    elif phase == MineralPhase.enstatite:
        if fabric == MineralFabric.enstatite_AB:
            return np.array([np.inf, np.inf, np.inf, 1])
        raise ValueError(f"unsupported enstatite fabric: {fabric}")
    raise ValueError(f"phase must be a valid `MineralPhase`, not {phase}")


# 12+ args is a lot, but this way we can use numba
# (only primitives and numpy containers allowed).
@nb.njit(fastmath=True)
def derivatives(
    regime: DeformationRegime,
    phase: MineralPhase,
    fabric: MineralFabric,
    n_grains: int,
    orientations: np.ndarray,
    fractions: np.ndarray,
    strain_rate: np.ndarray,
    velocity_gradient: np.ndarray,
    deformation_gradient_spin: np.ndarray,
    stress_exponent: float,
    deformation_exponent: float,
    nucleation_efficiency: float,
    gbm_mobility: float,
    volume_fraction: float,
):
    """Get derivatives of orientation and volume distribution.

    - `regime` — ordinal number of the local deformation mechanism
    - `phase` — ordinal number of the mineral phase
    - `fabric` — ordinal number of the fabric type
    - `n_grains` — number of "grains" i.e. discrete volume segments
    - `orientations` — `n_grains`x3x3 orientations (direction cosines)
    - `fractions` — volume fractions of the grains relative to aggregate volume
    - `strain_rate` — 3x3 dimensionless macroscopic strain-rate tensor
    - `velocity_gradient` — 3x3 dimensionless macroscopic velocity gradient
    - `deformation_gradient_spin` — 3x3 spin tensor defining the rate of rotation of the
      finite strain ellipse
    - `stress_exponent` — `p` in `dislocation_density ∝ shear_stress^p`
    - `deformation_exponent` — `n` in `shear_stress ∝ |deformation_rate|^(1/n)`
    - `nucleation_efficiency` — parameter controlling grain nucleation
    - `gmb_mobility` — grain boundary mobility parameter
    - `volume_fraction` — volume fraction of the mineral phase relative to other phases
      (multiphase simulations)

    Returns a tuple with the rotation rates and grain volume fraction changes.

    """
    if regime == DeformationRegime.min_viscosity:
        # Do absolutely nothing, all derivatives are zero.
        return (
            np.repeat(np.eye(3), n_grains).reshape(3, 3, n_grains).transpose(),
            np.zeros(n_grains),
        )
    elif regime == DeformationRegime.matrix_diffusion:
        # Passive rotation based on macroscopic vorticity for diffusion creep?
        # vorticity = 0.5 * (velocity_gradient - velocity_gradient.transpose())
        # Passive rotation based on spin of F for diffusion creep.
        vorticity = deformation_gradient_spin
        # Or just don't change at all?
        # vorticity = np.zeros((3, 3))
        # This 💃 is because numba doesn't let us use np.tile or even np.array([a] * n).
        return (
            np.repeat(vorticity.transpose(), n_grains)
            .reshape(3, 3, n_grains)
            .transpose(),
            np.zeros(n_grains),
        )
    elif regime == DeformationRegime.boundary_diffusion:
        raise ValueError("this deformation mechanism is not yet supported.")
    elif regime == DeformationRegime.sliding_diffusion:
        # Miyazaki et al. 2013 wrote controversial Nature article proposing that CPO can
        # develop in the diffGBS regime. However, Karato 2024 gives some convincing
        # arguments in the Journal of Geodynamics for why their results are likely
        # inapplicable to Earth's upper mantle.
        raise ValueError("this deformation mechanism is not yet supported.")
    elif regime == DeformationRegime.matrix_dislocation:
        # Based on subroutine DERIV in original Fortran.
        strain_energies = np.empty(n_grains)
        orientations_diff = np.empty((n_grains, 3, 3))
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
    elif regime == DeformationRegime.sliding_dislocation:
        raise ValueError("this deformation mechanism is not yet supported.")
    elif regime == DeformationRegime.frictional_yielding:
        # For now, the same as matrix_dislocation, but we smooth the strain energy
        # distribution and the orientation changes, since some energy is lost to
        # micro-fracturing. Also increase the GBS threshold, dislocations will tend to
        # pile up at grain boundaries.
        # TODO: Maybe modify the stress/deformation exponents?
        # TODO: Reduce nucleation efficiency?
        strain_energies = np.empty(n_grains)
        orientations_diff = np.empty((n_grains, 3, 3))
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
            orientations_diff[grain_index] = 0.3 * orientation_change
            strain_energies[grain_index] = strain_energy
        # Volume average mean strain energy.
        mean_energy = np.sum(fractions * strain_energies)
        # Strain energy residuals, minus the energy lost to micro-fracturing.
        strain_residuals = 0.3 * (mean_energy - strain_energies)
        fractions_diff = volume_fraction * gbm_mobility * fractions * strain_residuals
        return orientations_diff, fractions_diff
    elif regime == DeformationRegime.max_viscosity:
        # Do absolutely nothing, all derivatives are zero.
        return (
            np.repeat(np.eye(3), n_grains).reshape(3, 3, n_grains).transpose(),
            np.zeros(n_grains),
        )
    else:
        raise ValueError(f"regime must be a valid `DeformationRegime`, not {regime}")


@nb.njit(fastmath=True)
def _get_deformation_rate(
    phase: MineralPhase, orientation: np.ndarray, slip_rates: np.ndarray
):
    """Calculate deformation rate tensor for olivine or enstatite.

    Calculate the deformation rate with respect to the local coordinate frame,
    defined by the principal strain axes (finite strain ellipsoid).

    - `phase` — ordinal number of the mineral phase
    - `orientation` — 3x3 orientation matrix (direction cosines)
    - `slip_rates` — slip rates relative to slip rate on softest slip system

    """
    # This is called the 'G'/'slip' tensor in the Fortran code, aka the Schmid tensor.
    deformation_rate = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            deformation_rate[i, j] = 2 * (
                slip_rates[0] * orientation[0, i] * orientation[1, j]
                + slip_rates[1] * orientation[0, i] * orientation[2, j]
                + slip_rates[2] * orientation[2, i] * orientation[1, j]
                + slip_rates[3] * orientation[2, i] * orientation[0, j]
            )
    return deformation_rate


@nb.njit(fastmath=True)
def _get_slip_rate_softest(deformation_rate: np.ndarray, velocity_gradient: np.ndarray):
    """Calculate dimensionless strain rate on the softest slip system.

    - `deformation_rate` — 3x3 dimensionless deformation rate matrix
    - `velocity_gradient` — 3x3 dimensionless velocity gradient matrix

    """
    # See eq. 4 in Fraters 2021.
    enumerator = 0
    denominator = 0

    for j in range(3):
        # NOTE: Mistake in original DRex code (j + 2), see Fraters & Billen 2021 S1.
        if _USE_ORIGINAL_DREX:
            k = (j + 2) % 3
        else:
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

    # Avoid zero divisions:
    # Tiny denominator means that relevant deformation_rate entries are zero.
    # No deformation rate means no slip rate.
    if -1e-15 < denominator < 1e-15:
        return 0.0
    return enumerator / denominator


@nb.njit(fastmath=True)
def _get_slip_rates_olivine(
    invariants: np.ndarray,
    slip_indices: np.ndarray,
    crss: np.ndarray,
    deformation_exponent: float,
):
    """Calculate relative slip rates of the active slip systems for olivine.

    - `invariants` — strain rate invariants for the four slip systems
    - `slip_indices` — indices that sort the CRSS by increasing slip-rate activity
    - `crss` — reference resolved shear stresses (CRSS), see `pydrex.fabric`
    - `deformation_exponent` — `n` in `shear_stress ∝ |deformation_rate|^(1/n)`

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
def _get_slip_invariants(strain_rate: np.ndarray, orientation: np.ndarray):
    r"""Calculate strain rate invariants for minerals with four slip systems.

    Calculates $I = ∑_{ij} l_{i} n_{j} \dot{ε}_{ij}$ for each slip sytem of:
    - (010)[100]
    - (001)[100]
    - (010)[001]
    - (100)[001]
    Only the last return value is relevant for enstatite.
    These are not configurable for now.

    - `strain_rate` — 3x3 dimensionless strain rate matrix
    - `orientation` — 3x3 orientation matrix (direction cosines)

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
    orientation: np.ndarray,
    velocity_gradient: np.ndarray,
    deformation_rate: np.ndarray,
    slip_rate_softest: float,
):
    """Calculate the rotation rate for a grain undergoing dislocation creep.

    - `orientation` — 3x3 orientation matrix (direction cosines)
    - `velocity_gradient` — 3x3 dimensionless velocity gradient matrix
    - `deformation_rate` — 3x3 dimensionless deformation rate matrix
    - `slip_rate_softest` — slip rate on the softest (most active) slip system

    """
    orientation_change = np.zeros((3, 3))
    # Spin vector for the grain, see eq. 3 in Fraters 2021.
    # Includes rigid body rotation as well as the plastic deformation contribution.
    # This means that even with no active slip systems, the rotation will be nonzero
    # if there is a rotational (i.e. antisymmetric) component of the velocity_gradient.
    spin_vector = np.empty(3)
    for j in range(3):
        r = (j + 1) % 3
        s = (j + 2) % 3
        spin_vector[j] = (
            (velocity_gradient[s, r] - velocity_gradient[r, s])
            - (deformation_rate[s, r] - deformation_rate[r, s]) * slip_rate_softest
        ) / 2

    # Calculate rotation rate, see eq. 9 Kaminski & Ribe (2001).
    # Equivalent to:
    #   spin_matrix = np.einsum("ikj,k->ij", PERMUTATION_SYMBOL, spin_vector)
    #   orientation_change = spin_matrix.transpose() @ orientation
    for p in range(3):
        for q in range(3):
            for r in range(3):
                for s in range(3):
                    orientation_change[p, q] += (
                        PERMUTATION_SYMBOL[q, r, s] * orientation[p, s] * spin_vector[r]
                    )

    return orientation_change


@nb.njit(fastmath=True)
def _get_strain_energy(
    crss: np.ndarray,
    slip_rates: np.ndarray,
    slip_indices: np.ndarray,
    slip_rate_softest: float,
    stress_exponent: float,
    deformation_exponent: float,
    nucleation_efficiency: float,
):
    """Calculate strain energy due to dislocations for an olivine grain.

    - `crss` — reference resolved shear stresses (CRSS), see `pydrex.fabric`
    - `slip_rates` — slip rates relative to slip rate on softest slip system
    - `slip_indices` — indices that sort the CRSS by increasing slip-rate activity
    - `slip_rate_softest` — slip rate on the softest (most active) slip system
    - `stress_exponent` — `p` in `dislocation_density ∝ shear_stress^p`
    - `deformation_exponent` — `n` in `shear_stress ∝ |deformation_rate|^(1/n)`
    - `nucleation_efficiency` — parameter controlling grain nucleation

    Note that "new" grains are assumed to rotate with their parent.

    """
    strain_energy = 0.0
    # Dimensionless dislocation density for each slip system.
    # See eq. 16 Fraters 2021.
    # NOTE: Mistake in eq. 11, Kaminski 2004: spurious division by strain rate scale.
    # NOTE: Here we call 'p' the 'stress_exponent' and 'n' the 'deformation_exponent',
    # but in the original they use the variable 'stress_exponent' for 'n' (3.5).
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
def _get_rotation_and_strain(
    phase: MineralPhase,
    fabric: MineralFabric,
    orientation: np.ndarray,
    strain_rate: np.ndarray,
    velocity_gradient: np.ndarray,
    stress_exponent: float,
    deformation_exponent: float,
    nucleation_efficiency: float,
):
    """Get the crystal axes rotation rate and strain energy of individual grain.

    - `phase` — ordinal number of the mineral phase
    - `fabric` — ordinal number of the fabric type
    - `orientation` — 3x3 orientation matrix (direction cosines)
    - `strain_rate` — 3x3 dimensionless strain rate matrix
    - `velocity_gradient` — 3x3 dimensionless velocity gradient matrix
    - `stress_exponent` — `p` in `dislocation_density ∝ shear_stress^p`
    - `deformation_exponent` — `n` in `shear_stress ∝ |deformation_rate|^(1/n)`
    - `nucleation_efficiency — parameter controlling grain nucleation

    Note that "new" grains are assumed to rotate with their parent.

    Returns a tuple with the rotation rate of the crystalline axes
    with respect to the principal strain axes and strain energy of the grain.

    """
    crss = get_crss(phase, fabric)
    slip_invariants = _get_slip_invariants(strain_rate, orientation)
    if phase == MineralPhase.olivine:
        slip_indices = np.argsort(np.abs(slip_invariants / crss))
        slip_rates = _get_slip_rates_olivine(
            slip_invariants,
            slip_indices,
            crss,
            deformation_exponent,
        )
    elif phase == MineralPhase.enstatite:
        slip_indices = np.argsort(1 / crss)
        slip_rates = np.zeros(4)
        if _USE_ORIGINAL_DREX:
            slip_rates[-1] = 1  # Original had an arbitrary slip always active (L1410).
        else:
            # Assumes exclusively (100)[001] slip for enstatite.
            if np.abs(slip_invariants[-1]) > 1e-15:
                slip_rates[-1] = 1
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
    strain_energy = _get_strain_energy(
        crss,
        slip_rates,
        slip_indices,
        slip_rate_softest,
        stress_exponent,
        deformation_exponent,
        nucleation_efficiency,
    )
    return orientation_change, strain_energy
