r"""PyDRex: 2D corner flow tests.

The flow field is defined by:
$$
u = \frac{dr}{dt} ⋅ \hat{r} + r \frac{dθ}{dt} ⋅ \hat{θ}
= \frac{2 U}{π}(θ\sinθ - \cosθ) ⋅ \hat{r} + \frac{2 U}{π}θ\cosθ ⋅ \hat{θ}
$$
where $r = θ = 0$ points vertically downwards along the ridge axis
and $θ = π/2$ points along the surface. $$U$$ is the half spreading velocity.
Streamlines for the flow obey:
$$
ψ = \frac{2 U r}{π}θ\cosθ
$$
and are related to the velocity through:
$$
u = -\frac{1}{r} ⋅ \frac{dψ}{dθ} ⋅ \hat{r} + \frac{dψ}{dr}\hat{θ}
$$
The velocity gradient in the Cartesian (x, z) basis is given by:
$$
L = \frac{4 U}{π} ⋅
\begin{matrix}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    \cosθ\sin^{2}θ & 0 & \cosθ^{2}\sinθ
\end{matrix}
$$


Similar to Fig. 5 in [Kaminski 2002].

[Kaminski 2002](https://doi.org/10.1029/2001GC000222)

"""
import itertools as it

import numba as nb
import numpy as np
from scipy import linalg as la
from scipy.spatial.transform import Rotation

from pydrex import deformation_mechanism as _defmech
from pydrex import diagnostics as _diagnostics
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import pathlines as _pathlines
from pydrex import visualisation as _vis

# from pydrex import vtk_helpers as _vtk


@nb.njit
def get_velocity_polar(θ, plate_velocity):
    """Return velocity in a corner flow (Polar coordinate basis)."""
    return (2 * plate_velocity / np.pi) * np.array(
        [θ * np.sin(θ) - np.cos(θ), θ * np.cos(θ)]
    )


@nb.njit
def get_velocity(θ, plate_velocity):
    """Return velocity in a corner flow (Cartesian coordinate basis)."""
    sinθ = np.sin(θ)
    cosθ = np.cos(θ)
    return (2 * plate_velocity / np.pi) * np.array(
        [θ - sinθ * cosθ, 0.0, cosθ**2],
    )


@nb.njit
def get_velocity_gradient(θ, plate_velocity):
    """Return velocity gradient in a corner flow (Cartesian coordinate basis)."""
    sinθ = np.sin(θ)
    cosθ = np.cos(θ)
    return (4.0 * plate_velocity / np.pi) * np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [cosθ * sinθ**2, 0, cosθ**2 * sinθ],
        ]
    )


class TestOlivineA:
    """Tests for pure A-type olivine polycrystals in 2D corner flows."""

    def test_corner_nopathline_init_random(
        self,
        params_Kaminski2001_fig5_shortdash,
        outdir,
    ):
        """Test CPO evolution during forward advection under prescribed velocity.

        Initial condition: fully random orientations in all 4 `Mineral`s.

        Plate velocity: 2 cm/yr

        """
        # Plate velocity (half spreading rate), convert cm/yr to m/s.
        plate_velocity = 2.0 / (100.0 * 365.0 * 86400.0)
        domain_height = 1.0  # Normalised to olivine-spinel transition.
        domain_width = 5.0
        n_grains = 1000
        orientations_init = Rotation.random(n_grains, random_state=1).as_matrix()

        # Optional plotting/logging setup.
        if outdir is not None:
            _log.logfile_enable(f"{outdir}/corner_olivineA_nopathline.log")
            labels = []
            angles = []
            indices = []
            r_paths = []
            θ_paths = []
            directions = []
            timestamps = []

        # Note: θ values are in radians.
        for x_init in (0.25, 0.5, 1.0, 2.0):
            mineral = _minerals.Mineral(
                _minerals.MineralPhase.olivine,
                _minerals.OlivineFabric.A,
                _defmech.Regime.dislocation,
                n_grains=n_grains,
                fractions_init=np.full(n_grains, 1 / n_grains),
                orientations_init=orientations_init,
            )
            r_init = np.sqrt(domain_height**2 + x_init**2)
            θ_init = np.arctan2(x_init, domain_height)
            r_current = r_init
            θ_current = θ_init
            r_vals = [r_init]
            θ_vals = [θ_init]
            t_vals = [0]
            deformation_gradient = np.eye(3)
            # While the polycrystal is in the domain.
            while r_current * np.sin(θ_current) < domain_width:
                velocity = get_velocity_polar(θ_current, plate_velocity)
                timestep = 0.1 / la.norm(velocity)
                deformation_gradient = mineral.update_orientations(
                    params_Kaminski2001_fig5_shortdash,
                    deformation_gradient,
                    get_velocity_gradient(θ_current, plate_velocity),
                    integration_time=timestep,
                )
                r_current += velocity[0] * timestep
                θ_current += velocity[1] * timestep / r_current
                r_vals.append(r_current)
                θ_vals.append(θ_current)
                t_vals.append(t_vals[-1] + timestep)

            _log.info("final deformation gradient:\n%s", deformation_gradient)
            n_timesteps = len(t_vals)
            assert (
                n_timesteps
                == len(mineral.orientations)
                == len(mineral.fractions)
                == len(r_vals)
                == len(θ_vals)
            ), (
                f"n_timesteps = {n_timesteps}\n"
                + f"len(mineral.orientations) = {len(mineral.orientations)}\n"
                + f"len(mineral.fractions) = {len(mineral.fractions)}\n"
                + f"len(r_vals) = {len(r_vals)}\n"
                + f"len(θ_vals) = {len(θ_vals)}\n"
            )

            misorient_angles = np.zeros(n_timesteps)
            misorient_indices = np.zeros(n_timesteps)
            bingham_vectors = np.zeros((n_timesteps, 3))
            # Loop over first dimension (time steps) of orientations.
            for idx, matrices in enumerate(mineral.orientations):
                orientations_resampled, _ = _diagnostics.resample_orientations(
                    matrices, mineral.fractions[idx]
                )
                direction_mean = _diagnostics.bingham_average(
                    orientations_resampled,
                    axis=_minerals.get_primary_axis(mineral.fabric),
                )
                misorient_angles[idx] = _diagnostics.smallest_angle(
                    direction_mean,
                    [1, 0, 0],
                )
                misorient_indices[idx] = _diagnostics.misorientation_index(
                    orientations_resampled
                )
                bingham_vectors[idx] = direction_mean

            # TODO: Do the asserts here.
            # angles_diff = np.diff(misorient_angles)
            # assert np.max(angles_diff) < ...
            # assert np.min(angles_diff) > ...
            # assert np.sum(angles_diff) < ...

            if outdir is not None:
                mineral.save(
                    f"{outdir}/corner_olivineA_nopathline_{str(x_init).replace('.', 'd')}.npz"
                )
                # TODO: Also save processed data?
                labels.append(rf"$x_{0}$ = {x_init}")
                angles.append(misorient_angles)
                indices.append(misorient_indices)
                r_paths.append(r_vals)
                θ_paths.append(θ_vals)
                # Make timestamps end at 0 for nicer plotting.
                timestamps.append([t - t_vals[-1] for t in t_vals])
                directions.append(bingham_vectors)

        if outdir is not None:
            _vis.corner_flow_nointerp_2d(
                angles,
                indices,
                r_paths,
                θ_paths,
                directions,
                timestamps,
                xlabel=f"x ⇀ ({plate_velocity:.2e} m/s)",
                savefile=f"{outdir}/corner_olivineA_nopathline.png",
                markers=("o", "v", "s", "p"),
                labels=labels,
                xlims=(0, domain_width),
                zlims=(-domain_height, 0),
            )

    def test_corner_pathline_prescribed_init_random(
        self,
        params_Kaminski2001_fig5_shortdash,
        outdir,
    ):
        """Test CPO evolution during forward integration along a pathline.

        Initial condition: fully random orientations in all 4 `Mineral`s.

        Plate velocity: 2 cm/yr

        """
        # Plate velocity (half spreading rate), convert cm/yr to m/s.
        plate_velocity = 2.0 / (100.0 * 365.0 * 86400.0)
        domain_height = 1.0  # Normalised to olivine-spinel transition.
        domain_width = 5.0
        n_grains = 1000
        orientations_init = Rotation.random(n_grains, random_state=1).as_matrix()
        n_timesteps = 20  # Number of places along the pathline to compute CPO.

        # Optional plotting setup.
        if outdir is not None:
            labels = []
            angles = []
            indices = []
            r_paths = []
            θ_paths = []
            directions = []
            timestamps = []

        # Callables used to prescribe the macroscopic fields.
        @nb.njit
        def _get_velocity(point):
            x, _, z = point[0]  # Expects a 2D array for the coords.
            θ = np.arctan2(x, -z)
            # Return with an extra dimension of shape 1, like scipy RBF.
            return np.atleast_2d(get_velocity(θ, plate_velocity))

        @nb.njit
        def _get_velocity_gradient(point):
            x, _, z = point[0]  # Expects a 2D array for the coords.
            θ = np.arctan2(x, -z)
            # Return with an extra dimension of shape 1, like scipy RBF.
            velocity_gradient = get_velocity_gradient(θ, plate_velocity)
            return np.reshape(velocity_gradient, (1, *velocity_gradient.shape))

        # Note: θ values are in radians.
        for z_exit in (-0.1, -0.3, -0.54, -0.78):
            mineral = _minerals.Mineral(
                _minerals.MineralPhase.olivine,
                _minerals.OlivineFabric.A,
                _defmech.Regime.dislocation,
                n_grains=n_grains,
                fractions_init=np.full(n_grains, 1 / n_grains),
                orientations_init=orientations_init,
            )
            r_exit = np.sqrt(z_exit**2 + domain_width**2)
            θ_exit = np.arccos(z_exit / r_exit)
            timestamps_back, get_position = _pathlines.get_pathline(
                np.array([domain_width, 0.0, z_exit]),
                _get_velocity,
                _get_velocity_gradient,
                min_coords=np.array([0.0, 0.0, -domain_height]),
                max_coords=np.array([domain_width, 0.0, 0.0]),
            )
            r_vals = []
            θ_vals = []
            deformation_gradient = np.eye(3)
            times = np.linspace(timestamps_back[-1], timestamps_back[0], n_timesteps)
            for time_start, time_end in it.pairwise(times):
                deformation_gradient = mineral.update_orientations(
                    params_Kaminski2001_fig5_shortdash,
                    deformation_gradient,
                    _get_velocity_gradient,
                    integration_time=time_end - time_start,
                    # FIXME: get_position seems to get stuck, need pathline tests
                    pathline=(time_start, time_end, get_position),
                )
                x_current, _, z_current = get_position(time_end)
                r_current = np.sqrt(x_current**2 + z_current**2)
                θ_current = np.arctan2(x_current, -z_current)
                if outdir is not None:
                    r_vals.append(r_current)
                    θ_vals.append(θ_current)

            r_vals.append(r_exit)
            θ_vals.append(θ_exit)

            _log.info("final deformation gradient:\n%s", deformation_gradient)
            assert (
                n_timesteps
                == len(mineral.orientations)
                == len(mineral.fractions)
                == len(r_vals)
                == len(θ_vals)
            ), (
                f"n_timesteps = {n_timesteps}\n"
                + f"len(mineral.orientations) = {len(mineral.orientations)}\n"
                + f"len(mineral.fractions) = {len(mineral.fractions)}\n"
                + f"len(r_vals) = {len(r_vals)}\n"
                + f"len(θ_vals) = {len(θ_vals)}\n"
            )

            misorient_angles = np.zeros(n_timesteps)
            misorient_indices = np.zeros(n_timesteps)
            bingham_vectors = np.zeros((n_timesteps, 3))
            # Loop over first dimension (time steps) of orientations.
            for idx, matrices in enumerate(mineral.orientations):
                orientations_resampled, _ = _diagnostics.resample_orientations(
                    matrices, mineral.fractions[idx]
                )
                direction_mean = _diagnostics.bingham_average(
                    orientations_resampled,
                    axis=_minerals.get_primary_axis(mineral.fabric),
                )
                misorient_angles[idx] = _diagnostics.smallest_angle(
                    direction_mean,
                    [1, 0, 0],
                )
                misorient_indices[idx] = _diagnostics.misorientation_index(
                    orientations_resampled
                )
                bingham_vectors[idx] = direction_mean

            # TODO: Do the asserts here.
            # angles_diff = np.diff(misorient_angles)
            # assert np.max(angles_diff) < ...
            # assert np.min(angles_diff) > ...
            # assert np.sum(angles_diff) < ...

            if outdir is not None:
                mineral.save(
                    f"{outdir}/corner_olivineA_pathline_prescribed_{str(z_exit).replace('.', 'd')}.npz"
                )
                # TODO: Also save processed data?
                labels.append(rf"$z_{{f}}$ = {z_exit}")
                angles.append(misorient_angles)
                indices.append(misorient_indices)
                r_paths.append(r_vals)
                θ_paths.append(θ_vals)
                timestamps.append(times)
                directions.append(bingham_vectors)

        if outdir is not None:
            _vis.corner_flow_nointerp_2d(
                angles,
                indices,
                r_paths,
                θ_paths,
                directions,
                timestamps,
                xlabel=f"x ⇀ ({plate_velocity:.2e} m/s)",
                savefile=f"{outdir}/cornerXZ_olivineA_pathline_prescribed.png",
                markers=("o", "v", "s", "p"),
                labels=labels,
                xlims=(0, domain_width),
                zlims=(-domain_height, 0),
            )

#     @pytest.mark.wip
#     def test_corner_pathline_prescribed_init_random(
#         self,
#         params_Kaminski2001_fig5_shortdash,
#         outdir,
#     ):
#         """Test CPO evolution during forward integration along a pathline.

#         Initial condition: fully random orientations in all 4 `Mineral`s.

#         Plate velocity: 2 cm/yr

#         """
#         # Plate velocity (half spreading rate), convert cm/yr to m/s.
#         plate_velocity = 2.0 / (100.0 * 365.0 * 86400.0)
#         domain_width = 5.0
#         n_grains = 1000
#         orientations_init = Rotation.random(n_grains, random_state=1).as_matrix()

#         params = params_Kaminski2001_fig5_shortdash
#         params["gbm_mobility"] = 125

#         # Optional plotting setup.
#         if outdir is not None:
#             labels = []
#             angles = []
#             indices = []
#             r_paths = []
#             θ_paths = []
#             directions = []

#         # Callables to prescribe the pathline.
#         @nb.njit
#         def _get_velocity(point_polar):
#             r, θ = point_polar[0]  # Expects a 2D array for the coords.
#             velocity = get_velocity(θ, plate_velocity)
#             # Return with an extra dimension of shape 1, like scipy RBF.
#             return np.reshape(velocity, (1, *velocity.shape))

#         @nb.njit
#         def _get_velocity_gradient(point_polar):
#             # r, θ = point_polar[0]  # Expects a 2D array for the coords.
#             # Return with an extra dimension of shape 1, like scipy RBF.
#             # FIXME: Numba is still complaining about a reflective list here.
#             velocity_gradient = get_velocity_gradient(point_polar[0][1], plate_velocity)
#             return np.reshape(velocity_gradient, (1, *velocity_gradient.shape))

#         # Note: θ values are in radians.
#         for z_exit in (0.1, 0.3, 0.54, 0.78):
#             mineral = _minerals.Mineral(
#                 _minerals.MineralPhase.olivine,
#                 _minerals.OlivineFabric.A,
#                 _defmech.Regime.dislocation,
#                 n_grains=n_grains,
#                 fractions_init=np.full(n_grains, 1 / n_grains),
#                 orientations_init=orientations_init,
#             )
#             # r_init = np.sqrt(domain_height**2 + distance_init**2)
#             # θ_init = np.arcsin(distance_init / r_init)
#             r_exit = np.sqrt(z_exit**2 + domain_width**2)
#             θ_exit = np.arccos(-z_exit / r_exit)
#             timestamps_back, get_position = _pathlines.get_pathline(
#                 np.array([r_exit, θ_exit]),
#                 _get_velocity,
#                 _get_velocity_gradient,
#                 min_coords=np.array([0., 0.]),
#                 max_coords=np.array([5., np.pi / 2]),
#             )
#             r_vals = []
#             θ_vals = []
#             deformation_gradient = np.eye(3)
#             for time_start, time_end in it.pairwise(reversed(timestamps_back)):
#                 # log.debug("Δt: %g", time_end - time_start)
#                 deformation_gradient = mineral.update_orientations(
#                     # params_Kaminski2001_fig5_shortdash,
#                     params,
#                     deformation_gradient,
#                     _get_velocity_gradient,
#                     # integration_time=np.inf,
#                     integration_time=time_end - time_start,
#                     pathline=(time_start, time_end, get_position),
#                 )
#                 r_current, θ_current = get_position(time_end)
#                 if outdir is not None:
#                     r_vals.append(r_current)
#                     θ_vals.append(θ_current)

#             r_vals.append(r_exit)
#             θ_vals.append(θ_exit)

#             n_timesteps = len(timestamps_back)
#             assert (
#                 n_timesteps
#                 == len(mineral.orientations)
#                 == len(mineral.fractions)
#                 == len(r_vals)
#                 == len(θ_vals)
#             ), (
#                 f"n_timesteps = {n_timesteps}\n"
#                 + f"len(mineral.orientations) = {len(mineral.orientations)}\n"
#                 + f"len(mineral.fractions) = {len(mineral.fractions)}\n"
#                 + f"len(r_vals) = {len(r_vals)}\n"
#                 + f"len(θ_vals) = {len(θ_vals)}\n"
#             )

#             misorient_angles = np.zeros(n_timesteps)
#             misorient_indices = np.zeros(n_timesteps)
#             bingham_vectors = np.zeros((n_timesteps, 3))
#             # Loop over first dimension (time steps) of orientations.
#             for idx, matrices in enumerate(mineral.orientations):
#                 orientations_resampled, _ = _diagnostics.resample_orientations(
#                     matrices, mineral.fractions[idx]
#                 )
#                 direction_mean = _diagnostics.bingham_average(
#                     orientations_resampled,
#                     axis=_minerals.get_primary_axis(mineral.fabric),
#                 )
#                 misorient_angles[idx] = _diagnostics.smallest_angle(
#                     direction_mean,
#                     [1, 0, 0],
#                 )
#                 misorient_indices[idx] = _diagnostics.misorientation_index(
#                     orientations_resampled
#                 )
#                 bingham_vectors[idx] = direction_mean

#             # TODO: Do the asserts here.
#             # angles_diff = np.diff(misorient_angles)
#             # assert np.max(angles_diff) < ...
#             # assert np.min(angles_diff) > ...
#             # assert np.sum(angles_diff) < ...

#             if outdir is not None:
#                 mineral.save(
#                     f"{outdir}/corner_olivineA_pathline_prescribed_{str(z_exit).replace('.', 'd')}.npz"
#                 )
#                 # TODO: Also save processed data?
#                 labels.append(rf"$z_{{f}}$ = {z_exit}")
#                 angles.append(misorient_angles)
#                 indices.append(misorient_indices)
#                 r_paths.append(r_vals)
#                 θ_paths.append(θ_vals)
#                 directions.append(bingham_vectors)

#         if outdir is not None:
#             _vis.corner_flow_nointerp_2d(
#                 angles,
#                 indices,
#                 r_vals,
#                 θ_vals,
#                 directions,
#                 timestamps=timestamps_back[::-1],
#                 savefile=f"{outdir}/cornerXZ_olivineA_pathline_prescribed.png",
#                 markers=("o", "v", "s", "p"),
#                 labels=labels,
#             )
