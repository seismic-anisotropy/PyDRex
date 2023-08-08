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


def eigh_Kaminski2001(matrix, n=3, np=3):
    """Calculate eigenvalues and eigenvectors of a real-valued symmetric matrix.

    Implementation according to JACOBI in the Fortran DRexV2b.f90.

    """
    v = np.empty((np, np))  # Eigenvectors.
    d = np.empty(np)  # Eigenvalues.
    b = np.empty(n * (n - 1))
    z = np.empty(n * (n - 1))

    for ip in range(n):
        for iq in range(n):
            v[ip, iq] = 0.0
        v[ip, ip] = 1.0

    for ip in range(n):
        b[ip] = matrix[ip, ip]
        d[ip] = b[ip]
        z[ip] = 0.0

    for i in range(50):
        sm = 0.0
        for ip in range(n - 1):
            for iq in range(ip, n):  # Zero-indexing: ip+1 -> ip.
                sm = sm + abs(matrix[ip, iq])
        if sm == 0.0:
            return
        if i < 3:  # Zero-indexing: 4 -> 3.
            tresh = 0.2 * sm / n**2.0
        else:
            tresh = 0.0

        for ip in range(n - 1):
            for iq in range(ip, n):  # Zero-indexing: ip+1 -> ip.
                g = 100.0 * abs(matrix[ip, iq])
                if (
                    (i > 3)  # Zero-indexing: 4 -> 3.
                    and (abs(d[ip]) + g == abs(d[ip]))
                    and (abs(d[iq]) + g == abs(d[iq]))
                ):
                    matrix[ip, iq] = 0.0
                elif abs(matrix[ip, iq]) > tresh:
                    h = d[iq] - d[ip]
                    if abs(h) + g == abs(h):
                        t = matrix[ip, iq] / h
                    else:
                        theta = 0.5 * h / matrix[ip, iq]
                        t = 1.0 / (abs(theta) + np.sqrt(1.0 + theta**2.0))
                        if theta < 0.0:
                            t = -t
                    c = 1.0 / np.sqrt(1.0 + t**2.0)
                    s = t * c
                    tau = s / (1.0 + c)
                    h = t * matrix[ip, iq]
                    z[ip] = z[ip] - h
                    z[iq] = z[iq] + h
                    d[ip] = d[ip] - h
                    d[iq] = d[iq] + h
                    matrix[ip, iq] = 0.0

                    for j in range(ip - 1):
                        g = matrix[j, ip]
                        h = matrix[j, iq]
                        matrix[j, ip] = g - s * (h + g * tau)
                        matrix[j, iq] = h + s * (g - h * tau)

                    for j in range(ip, iq - 1):  # Zero-indexing: ip+1 -> ip.
                        g = matrix[ip, j]
                        h = matrix[j, iq]
                        matrix[ip, j] = g - s * (h + g * tau)
                        matrix[j, iq] = h + s * (g - h * tau)

                    for j in range(iq, n):  # Zero-indexing: iq+1 -> iq.
                        g = matrix[ip, j]
                        h = matrix[iq, j]
                        matrix[ip, j] = g - s * (h + g * tau)
                        matrix[iq, j] = h + s * (g - h * tau)

                    for j in range(n):
                        g = v[j, ip]
                        h = v[j, iq]
                        v[j, ip] = g - s * (h + g * tau)
                        v[j, iq] = h + s * (g - h * tau)

            for ip in range(n):
                b[ip] = b[ip] + z[ip]
                d[ip] = b[ip]
                z[ip] = 0.0
    return _eigsort(d, v, n, np)


def _eigsort(d, v, n, np):
    for i in range(n - 1):
        k = i
        p = d[i]
        for j in range(i, n):  # Zero-indexing: i+1 -> i.
            if d[j] >= p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            for j in range(n):
                p = v[j, i]
                v[j, i] = v[j, k]
                v[j, k] = p
    return d, v
