"""> PyDRex: Dynamic viscosity based on various proposed constitutive equations.

The contribution from dislocation creep can be calculated either using an exponential
law and a power-law equation for low and high homologous temperature respectively
(see [Demouchy et al. 2023](https://doi.org/10.2138/gselements.19.3.151)),
or using a unified temperature-insensitive creep law
(see [Garel et al. 2020](https://doi.org/10.1016/j.epsl.2020.116243)).
In the first case, it is necessary to choose a threshold homologous temperature¹ at which
the switch is made using the high-temperature power-law.
The referenced paper recommends a value of 0.7.

¹Homologous temperature is defined as T/Tₘ where T is temperature and Tₘ is the melting
(solidus) temperature. The melting temperature of olivine is inversely proportional to
iron content and water content, and increases with pressure.

"""

import numpy as np
from scipy import constants
from scipy import special as sp

import pydrex.minerals as _minerals
from pydrex.core import DefaultParams


def harmonic_mean(viscosities: tuple[float, ...], bounds: tuple[float, float]):
    """Calculate harmonic mean of `viscosities` and clip result between `bounds`."""
    # Defensive .clip for each component as well.
    return np.clip(
        len(viscosities) / np.sum([1 / np.clip(μ, *bounds) for μ in viscosities]),
        *bounds,
    )


def frictional_yielding(
    strain_rate_invariant_II,
    pressure,
    surface_cohesion,
    friction_coefficient,
    reference_density,
    # depth,
    # temperature,
):
    """Calculate viscosity contribution from semi-brittle yielding."""
    # TODO: Check: enumerator should come out as ~ 2e8.
    # TODO: Use depth and temperature thresholds? Check Bickert 2021 and Taufner 2021.
    τ_y = np.min(friction_coefficient + surface_cohesion * pressure, 1e9)
    return τ_y / (2 * strain_rate_invariant_II)


def matrix_dislocation_unified(
    strain_rate_invariant_II,
    temperature,
    pressure,
    dimension,
    coefficients=DefaultParams().disl_coefficients,
    method="tanh",
):
    """Calculate viscosity contribution from dislocation creep.

    Calculate viscosity contribution from dislocation creep using a unified flow law
    parametrised by 3 polynomials (7 coefficients) as per
    [Garel et al. 2020](https://doi.org/10.1016/j.epsl.2020.116243).

    """
    a0, b0, a1, b1, a2, b2, c2 = coefficients
    A0 = a0 + b0 * temperature
    A1 = a1 + b1 * temperature
    A2 = a2 + b2 * temperature + c2 * temperature**2

    f = None
    match method:
        case "tanh":
            f = np.tanh
        case "erf":
            f = sp.erf
        case "algebraic":

            def f(x):
                return x / np.sqrt(1 + x**2)

    σ_diff = A0 * (1 + f(A1 * (np.log10(strain_rate_invariant_II) - A2)))
    return _adjust_for_geometry(strain_rate_invariant_II, dimension, σ_diff)


def diffusion(
    strain_rate_invariant_II,
    temperature,
    pressure,
    dimension,
    prefactors=DefaultParams().diff_prefactors,
    activation_energies=DefaultParams().diff_activation_energies,
    activation_volumes=DefaultParams().diff_activation_volumes,
    mode: str = "both",
):
    """Calculate viscosity contribution from diffusion creep.

    Calculate viscosity contribution from diffusion in the
    `pydrex.core.DeformationRegime.matrix_diffusion` regime,
    using an exponential flow law.

    - `strain_rate_invariant_II` — second invariant of the strain rate¹
    - `temperature` — temperature in Kelvin
    - `pressure` — pressure in GPa
    - `dimension` — number of coordinates describing a point in the domain
    - `prefactors` — prefactors for the exponential and power laws (s⁻¹)
    - `activation_energies` — see `pydrex.core.DefaultParams.disl_activation_energies`
    - `activation_volumes` — see `pydrex.core.DefaultParams.disl_activation_volumes`
    - `mode` — which kind of diffusive regime is active (`"matrix"`, `"boundary"` or
        `"both"`; `"sliding"` is an alias for `"both"`)

    ¹The strain rate invariants can be calculated from the strain rate tensor using
    `pydrex.tensors.invariants_second_order` if necessary.

    """
    # The pre-multipliers of α₁=14 and α₂=14π are from Kohlstedt & Hansen, 2015:
    # <http://dx.doi.org/10.1016/B978-0-444-53802-4.00042-7> (see eq. 37).
    # Technically, I think those values are only valid for the "both"|"sliding" mode,
    # but I don't think there will be much chance to find values for other modes.
    # In the rare case that only one of the two diffusive mechanisms is active and there
    # is no sliding, I don't think that these 'geometric terms' will change a lot.

    # TODO: Kohlstedt & Hansen 2015 also divide the prefactor by R*T again...
    # This doesn't appear in Garel 2020 (eq. 10) nor in Chris' original models.
    # For now we'll skip it but I have no idea which option is correct.
    # Garel 2020 use R(T + δT) instead where δT is some sort of adiabatic correction,
    # but once again I'll omit this here for now.
    σ_diff_matrix = (
        14
        * prefactors[0] ** -1
        * np.exp(
            (activation_energies[0] + pressure * activation_volumes[0])
            / (constants.gas_constant * temperature)
        )
    )
    σ_diff_boundary = (
        14
        * np.pi
        * prefactors[1] ** -1
        * np.exp(
            (activation_energies[1] + pressure * activation_volumes[1])
            / (constants.gas_constant * temperature)
        )
    )
    match mode:
        case "matrix":
            return _adjust_for_geometry(
                strain_rate_invariant_II, dimension, σ_diff_matrix
            )
        case "boundary":
            return _adjust_for_geometry(
                strain_rate_invariant_II, dimension, σ_diff_boundary
            )
        case "both" | "sliding":
            return _adjust_for_geometry(
                strain_rate_invariant_II, dimension, σ_diff_matrix + σ_diff_boundary
            )
        case _:
            raise ValueError(
                f"diffusion creep mode must be one of 'matrix', 'boundary' or 'both', not '{mode}'"
            )


def matrix_dislocation_split(
    strain_rate_invariant_II,
    temperature,
    pressure,
    dimension,
    prefactors=DefaultParams().disl_prefactors,
    activation_energy=DefaultParams().disl_activation_energy,
    activation_volume=DefaultParams().disl_activation_volume,
    lowtemp_switch=DefaultParams().disl_lowtemp_switch,
    σ_Peierls=DefaultParams().disl_Peierls_stress,
    dry_olivine=True,
):
    """Calculate viscosity contribution from dislocation creep.

    Calculate viscosity contribution from dislocation in the
    `pydrex.core.DeformationRegime.matrix_dislocation` regime,
    using an exponential law below the `lowtemp_switch` homologous temperature and a
    power law above it.

    See e.g. equations 2 and 3 in
    [Demouchy et al. 2023](https://doi.org/10.2138/gselements.19.3.151).

    - `strain_rate_invariant_II` — second invariant of the strain rate¹
    - `temperature` — temperature in Kelvin
    - `pressure` — pressure in GPa
    - `dimension` — number of coordinates describing a point in the domain
    - `prefactors` — prefactors for the exponential and power laws (s⁻¹)
    - `activation_energy` — see `pydrex.core.DefaultParams.disl_activation_energy`
    - `activation_volume` — see `pydrex.core.DefaultParams.disl_activation_volume`
    - `lowtemp_switch` — see `pydrex.core.DefaultParams.disl_lowtemp_switch`
    - `σ_Peierls` — see `pydrex.core.DefaultParams.disl_Peierls_stress`
    - `dry_olivine` — use empyrical parametrisation for dry or hydrous olivine

    ¹The strain rate invariants can be calculated from the strain rate tensor using
    `pydrex.tensors.invariants_second_order` if necessary.

    """
    # Uniaxial strain rate correction.
    strain_rate = 2 * np.sqrt(3) * strain_rate_invariant_II
    homologous_temperature = temperature / _minerals.peridotite_solidus(pressure)

    if homologous_temperature < lowtemp_switch:
        σ_diff = σ_Peierls * np.power(
            1
            - np.power(
                (np.log(strain_rate) * constants.gas_constant * temperature)
                / (-activation_energy * prefactors[0]),
                1.5,  # superscript q (supplement says 1 ≤ q ≤ 2)
            ),
            0.5,  # superscript p (supplement says 0 ≤ p ≤ 1)
        )
    else:
        if dry_olivine:
            n = 3.5
        else:
            n = 3

        σ_diff = np.power(
            strain_rate
            / (
                prefactors[1]
                * np.exp(
                    (-activation_energy + pressure * activation_volume)
                    / (constants.gas_constant * temperature)
                )
            ),
            (1 / n),
        )

    return _adjust_for_geometry(strain_rate_invariant_II, dimension, σ_diff)


# This correction is discussed but not included in the Garel 2020 equations.
# The idea is that we correct for the fact that σ_diff ≠ σ and
# strain_rate_invariant_II ≠ strain_rate, where the deviation depends on the geometry.
# The actual constitutive equations are usually calibrated against uniaxial compression
# experiments or such, which is why they are actually defined in terms of σ_diff and
# strain_rate_invariant_II rather than σ and strain_rate.
def _adjust_for_geometry(strain_rate_invariant_II, dimension, σ_diff):
    strain_rate = 2 * np.sqrt(3) * strain_rate_invariant_II
    if dimension == 2:
        return σ_diff / (2 * np.sqrt(3) * strain_rate)
    elif dimension == 3:
        return σ_diff / (3 * strain_rate)
    else:
        raise ValueError(f"dimension must be 2 or 3, not {dimension}")
