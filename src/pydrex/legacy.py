"""> PyDRex: Legacy solver and deprecated methods."""
import numpy as np
from scipy import linalg as la

from pydrex import core as _core


def update_orientations_Kaminski2001(
    mineral,
    config,
    deformation_gradient,
    get_velocity_gradient,
    pathline,
    gridstep,
    velocity_magnitude,
    n_steps,
    **kwargs,
):
    """Implements the original RK45 approach (subroutine STRAIN).

    Almost a drop-in replacement for `pydrex.Mineral.update_orientations`,
    but requires a few additional arguments:
    - `gridstep`: local 'grid resolution' or an approximation of the spatial
      distance between local simulation grid nodes
    - `velocity_magnitude`: estimate of average local velocity magnitude along the
      pathline
    - `n_steps`: number of steps used for the streamline

    """
    time_start, time_end, get_position = pathline
    timestamps = np.linspace(time_start, time_end, n_steps + 1)
    for time in timestamps:
        position = get_position(time)
        velocity_gradient = get_velocity_gradient(position)
        strain_rate = (velocity_gradient + velocity_gradient.transpose()) / 2
        strain_rate_max = np.abs(la.eigvalsh(strain_rate)).max()
        _step = gridstep / velocity_magnitude
        step = min(_step, 1e-2 / strain_rate_max)
        n_iter = int(_step / step)

        if mineral.phase == _core.MineralPhase.olivine:
            volume_fraction = config["olivine_fraction"]
        elif mineral.phase == _core.MineralPhase.enstatite:
            volume_fraction = config["enstatite_fraction"]
        else:
            assert False

        for n in range(n_iter):
            fsei = deformation_gradient
            odfi = mineral.fractions[-1]
            acsi = mineral.orientations[-1]

            orientations_diff, fractions_diff = _core.derivatives(
                mineral.phase,
                mineral.fabric,
                mineral.n_grains,
                mineral.orientations[-1],
                mineral.fractions[-1],
                strain_rate,
                velocity_gradient,
                config["stress_exponent"],
                config["deformation_exponent"],
                config["nucleation_efficiency"],
                config["gbm_mobility"],
                volume_fraction,
            )

            kfse1 = velocity_gradient @ fsei * step
            kodf1 = fractions_diff * step * strain_rate_max
            kac1 = orientations_diff * step * strain_rate_max

            fsei = deformation_gradient + 0.5 * kfse1
            odfi = mineral.fractions[-1] + 0.5 * kodf1
            acsi = mineral.orientations[-1] + 0.50 * kac1

            for j in range(mineral.n_grains):
                for j1 in range(3):
                    for j2 in range(3):
                        if acsi[j, j1, j2] > 1.0:
                            acsi[j, j1, j2] = 1
                        if acsi[j, j1, j2] < -1.0:
                            acsi[j, j1, j2] = -1.0

            for j in range(mineral.n_grains):
                if odfi[j] < 0:
                    odfi[j] = 0.0

            odfi = odfi / np.sum(odfi)

            orientations_diff, fractions_diff = _core.derivatives(
                mineral.phase,
                mineral.fabric,
                mineral.n_grains,
                mineral.orientations[-1],
                mineral.fractions[-1],
                strain_rate,
                velocity_gradient,
                config["stress_exponent"],
                config["deformation_exponent"],
                config["nucleation_efficiency"],
                config["gbm_mobility"],
                volume_fraction,
            )

            kfse2 = velocity_gradient @ fsei * step
            kodf2 = fractions_diff * step * strain_rate_max
            kac2 = orientations_diff * step * strain_rate_max

            fsei = deformation_gradient + 0.5 * kfse2
            odfi = mineral.fractions[-1] + 0.5 * kodf2
            acsi = mineral.orientations[-1] + 0.50 * kac2

            for j in range(mineral.n_grains):
                for j1 in range(3):
                    for j2 in range(3):
                        if acsi[j, j1, j2] > 1.0:
                            acsi[j, j1, j2] = 1
                        if acsi[j, j1, j2] < -1.0:
                            acsi[j, j1, j2] = -1.0

            for j in range(mineral.n_grains):
                if odfi[j] < 0:
                    odfi[j] = 0.0

            odfi = odfi / np.sum(odfi)

            orientations_diff, fractions_diff = _core.derivatives(
                mineral.phase,
                mineral.fabric,
                mineral.n_grains,
                mineral.orientations[-1],
                mineral.fractions[-1],
                strain_rate,
                velocity_gradient,
                config["stress_exponent"],
                config["deformation_exponent"],
                config["nucleation_efficiency"],
                config["gbm_mobility"],
                volume_fraction,
            )

            kfse3 = velocity_gradient @ fsei * step
            kodf3 = fractions_diff * step * strain_rate_max
            kac3 = orientations_diff * step * strain_rate_max

            fsei = deformation_gradient + kfse3
            odfi = mineral.fractions[-1] + kodf3
            acsi = mineral.orientations[-1] + kac3

            for j in range(mineral.n_grains):
                for j1 in range(3):
                    for j2 in range(3):
                        if acsi[j, j1, j2] > 1.0:
                            acsi[j, j1, j2] = 1
                        if acsi[j, j1, j2] < -1.0:
                            acsi[j, j1, j2] = -1.0

            for j in range(mineral.n_grains):
                if odfi[j] < 0:
                    odfi[j] = 0.0

            odfi = odfi / np.sum(odfi)

            kfse4 = velocity_gradient @ fsei * step
            kodf4 = fractions_diff * step * strain_rate_max
            kac4 = orientations_diff * step * strain_rate_max

            deformation_gradient = (
                deformation_gradient + (kfse1 / 2.0 + kfse2 + kfse3 + kfse4 / 2.0) / 3.0
            )
            mineral.orientations.append(
                mineral.orientations[-1] + (kac1 / 2.0 + kac2 + kac3 + kac4 / 2.0) / 3.0
            )
            mineral.fractions.append(
                mineral.fractions[-1]
                + (kodf1 / 2.0 + kodf2 + kodf3 + kodf4 / 2.0) / 3.0
            )

            for j in range(mineral.n_grains):
                for j1 in range(3):
                    for j2 in range(3):
                        if acsi[j, j1, j2] > 1.0:
                            acsi[j, j1, j2] = 1
                        if acsi[j, j1, j2] < -1.0:
                            acsi[j, j1, j2] = -1.0

            for j in range(mineral.n_grains):
                if odfi[j] < 0:
                    odfi[j] = 0.0

            deformation_gradient = deformation_gradient / np.sum(deformation_gradient)

    return deformation_gradient
