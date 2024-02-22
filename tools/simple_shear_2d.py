from time import perf_counter

import numba as nb
import numpy as np

from pydrex import core as _core
from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
from pydrex import minerals as _minerals
from pydrex import velocity as _velocity

shear_direction = np.array([0.0, 1.0, 0.0])
strain_rate = 1e-4
_, get_velocity_gradient = _velocity.simple_shear_2d("Y", "X", strain_rate)
timestamps = np.linspace(0, 1e4, 201)
params = _io.DEFAULT_PARAMS
params["number_of_grains"] = 15**3
params["gbs_threshold"] = 0
seed = 245623452

mineral = _minerals.Mineral(
    phase=_core.MineralPhase.olivine,
    fabric=_core.MineralFabric.olivine_A,
    regime=_core.DeformationRegime.dislocation,
    n_grains=params["number_of_grains"],
    seed=seed,
)
deformation_gradient = np.eye(3)
θ_fse = np.empty_like(timestamps)
θ_fse[0] = 45


@nb.njit
def get_position(t):
    return np.zeros(3)


# Warm up run to make numba compile things.
mineral.update_orientations(
    params,
    deformation_gradient,
    get_velocity_gradient,
    pathline=(timestamps[0], timestamps[1], get_position),
)

_begin = perf_counter()
for t, time in enumerate(timestamps[:-1], start=1):
    deformation_gradient = mineral.update_orientations(
        params,
        deformation_gradient,
        get_velocity_gradient,
        pathline=(time, timestamps[t], get_position),
    )
    _, fse_v = _diagnostics.finite_strain(deformation_gradient)
    θ_fse[t] = _diagnostics.smallest_angle(fse_v, shear_direction)

# Cij_and_friends = _diagnostics.elasticity_components(
#     _minerals.voigt_averages([mineral], params)
# )
# angles = [
#     _diagnostics.smallest_angle(v, shear_direction)
#     for v in Cij_and_friends["hexagonal_axis"]
# ]
# percent_anisotropy = Cij_and_friends["percent_anisotropy"]

print(f"Elapsed time: {perf_counter() - _begin}")
