r"""> PyDRex: 2D corner flow tests

The flow field is defined by:
$$
\bm{u} = \frac{dr}{dt} \bm{\hat{r}} + r \frac{dθ}{dt} \bm{\hat{θ}}
= \frac{2 U}{π}(θ\sinθ - \cosθ) ⋅ \bm{\hat{r}} + \frac{2 U}{π}θ\cosθ ⋅ \bm{\hat{θ}}
$$
where $θ = 0$ points vertically downwards along the ridge axis
and $θ = π/2$ points along the surface. $U$ is the half spreading velocity.
Streamlines for the flow obey:
$$
ψ = \frac{2 U r}{π}θ\cosθ
$$
and are related to the velocity through:
$$
\bm{u} = -\frac{1}{r} ⋅ \frac{dψ}{dθ} ⋅ \bm{\hat{r}} + \frac{dψ}{dr}\bm{\hat{θ}}
$$
Conversion to Cartesian ($x,y,z$) coordinates yields:
$$
\bm{u} = \frac{2U}{π} \left[
\tan^{-1}\left(\frac{x}{-z}\right) + \frac{xz}{x^{2} + z^{2}} \right] \bm{\hat{x}} +
\frac{2U}{π} \frac{z^{2}}{x^{2} + z^{2}} \bm{\hat{z}}
$$
where
\begin{align\*}
x &= r \sinθ \cr
z &= -r \cosθ
\end{align\*}
and the velocity gradient is:
$$
L = \frac{4 U}{π{(x^{2}+z^{2})}^{2}} ⋅
\begin{bmatrix}
    -x^{2}z & 0 & x^{3} \cr
    0 & 0 & 0 \cr
    -xz^{2} & 0 & x^{2}z
\end{bmatrix}
$$

See also Fig. 5 in [Kaminski & Ribe, 2002](https://doi.org/10.1029/2001GC000222).

"""
import contextlib as cl
import itertools as it

import numba as nb
import numpy as np
from scipy.spatial.transform import Rotation

from pydrex import deformation_mechanism as _defmech
from pydrex import diagnostics as _diagnostics
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import pathlines as _pathlines
from pydrex import stats as _stats
from pydrex import visualisation as _vis


@nb.njit
def get_velocity(x, z, plate_velocity):
    """Return velocity in a corner flow (Cartesian coordinate basis)."""
    return (2 * plate_velocity / np.pi) * np.array(
        [np.arctan2(x, -z) + x * z / (x**2 + z**2), 0.0, z**2 / (x**2 + z**2)]
    )


@nb.njit
def get_velocity_gradient(x, z, plate_velocity):
    """Return velocity gradient in a corner flow (Cartesian coordinate basis)."""
    return (4.0 * plate_velocity / (np.pi * (x**2 + z**2) ** 2)) * np.array(
        [[-(x**2) * z, 0.0, x**3], [0.0, 0.0, 0.0], [-x * z**2, 0.0, x**2 * z]]
    )


class TestOlivineA:
    """Tests for pure A-type olivine polycrystals in 2D corner flows."""

    def test_corner_pathline_prescribed_init_isotropic(
        self,
        params_Kaminski2001_fig5_shortdash,
        rng,
        outdir,
    ):
        """Test CPO evolution during integration along a pathline.

        Initial condition: random orientations in all 4 `Mineral`s.

        Plate velocity: 2 cm/yr

        """
        # Plate velocity (half spreading rate), convert cm/yr to m/s.
        plate_velocity = 2.0 / (100.0 * 365.0 * 86400.0)
        domain_height = 1.0  # Normalised to olivine-spinel transition.
        domain_width = 5.0
        n_grains = 1000  # NOTE: Changes affect texture strength and tolerances.
        orientations_init = Rotation.random(n_grains, random_state=rng).as_matrix()
        n_timesteps = 20  # Number of places along the pathline to compute CPO.

        # Optional plotting and logging setup.
        optional_logging = cl.nullcontext()
        if outdir is not None:
            optional_logging = _log.logfile_enable(
                f"{outdir}/corner_olivineA_pathline_prescribed.log"
            )
            labels = []
            angles = []
            indices = []
            x_paths = []
            z_paths = []
            directions = []
            timestamps = []

        # Callables used to prescribe the macroscopic fields.
        @nb.njit
        def _get_velocity(point):
            x, _, z = point[0]  # Expects a 2D array for the coords.
            # Return with an extra dimension of shape 1, like scipy RBF.
            return np.atleast_2d(get_velocity(x, z, plate_velocity))

        @nb.njit
        def _get_velocity_gradient(point):
            x, _, z = point[0]  # Expects a 2D array for the coords.
            # Return with an extra dimension of shape 1, like scipy RBF.
            velocity_gradient = get_velocity_gradient(x, z, plate_velocity)
            return np.reshape(velocity_gradient, (1, *velocity_gradient.shape))

        with optional_logging:
            for z_exit in (-0.1, -0.3, -0.54, -0.78):
                mineral = _minerals.Mineral(
                    _minerals.MineralPhase.olivine,
                    _minerals.OlivineFabric.A,
                    _defmech.Regime.dislocation,
                    n_grains=n_grains,
                    fractions_init=np.full(n_grains, 1 / n_grains),
                    orientations_init=orientations_init,
                )
                timestamps_back, get_position = _pathlines.get_pathline(
                    np.array([domain_width, 0.0, z_exit]),
                    _get_velocity,
                    _get_velocity_gradient,
                    min_coords=np.array([0.0, 0.0, -domain_height]),
                    max_coords=np.array([domain_width, 0.0, 0.0]),
                )
                x_vals = []
                z_vals = []
                deformation_gradient = np.eye(3)
                times = np.linspace(
                    timestamps_back[-1], timestamps_back[0], n_timesteps
                )
                for time_start, time_end in it.pairwise(times):
                    deformation_gradient = mineral.update_orientations(
                        params_Kaminski2001_fig5_shortdash,
                        deformation_gradient,
                        _get_velocity_gradient,
                        integration_time=time_end - time_start,
                        pathline=(time_start, time_end, get_position),
                    )
                    x_current, _, z_current = get_position(time_end)
                    x_vals.append(x_current)
                    z_vals.append(z_current)

                x_vals.append(domain_width)
                z_vals.append(z_exit)

                _log.info("final deformation gradient:\n%s", deformation_gradient)
                assert (
                    n_timesteps
                    == len(mineral.orientations)
                    == len(mineral.fractions)
                    == len(x_vals)
                    == len(z_vals)
                ), (
                    f"n_timesteps = {n_timesteps}\n"
                    + f"len(mineral.orientations) = {len(mineral.orientations)}\n"
                    + f"len(mineral.fractions) = {len(mineral.fractions)}\n"
                    + f"len(x_vals) = {len(x_vals)}\n"
                    + f"len(z_vals) = {len(z_vals)}\n"
                )

                misorient_angles = np.zeros(n_timesteps)
                misorient_indices = np.zeros(n_timesteps)
                bingham_vectors = np.zeros((n_timesteps, 3))
                # Loop over first dimension (time steps) of orientations.
                for idx, matrices in enumerate(mineral.orientations):
                    orientations_resampled, _ = _stats.resample_orientations(
                        matrices, mineral.fractions[idx], rng=rng
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

                if outdir is not None:
                    mineral.save(
                        f"{outdir}/corner_olivineA_pathline_prescribed.npz",
                        str(z_exit).replace(".", "d"),
                    )
                    labels.append(rf"$z_{{f}}$ = {z_exit}")
                    angles.append(misorient_angles)
                    indices.append(misorient_indices)
                    x_paths.append(x_vals)
                    z_paths.append(z_vals)
                    timestamps.append(times)
                    directions.append(bingham_vectors)

                # Check that we start below the default cpo visualisation threshold.
                # assert misorient_indices[0] < 0.33
                # halfway = round(n_timesteps / 2)
                # TODO: Use np.isclose
                # match z_exit:
                #     case -0.1:
                #         assert misorient_indices[halfway] > 0.75
                #         assert misorient_angles[halfway] > 17.5
                #         assert misorient_indices[-1] > 0.8
                #         assert misorient_angles[-1] < 17.5
                #     case -0.3:
                #         assert misorient_indices[halfway] > 0.65
                #         assert misorient_angles[halfway] > 20
                #         assert misorient_indices[-1] > 0.75
                #         assert misorient_angles[-1] < 20
                #     case -0.54:
                #         assert misorient_indices[halfway] > 0.575
                #         assert misorient_angles[halfway] > 22
                #         assert misorient_indices[-1] > 0.7
                #         assert misorient_angles[-1] < 22
                #     case -0.78:
                #         # TODO: Results from this pathline are a bit weird.
                #         # No tests for now...
                #         pass

        if outdir is not None:
            _vis.corner_flow_2d(
                x_paths,
                z_paths,
                angles,
                indices,
                directions,
                timestamps,
                xlabel=f"x ⇀ ({plate_velocity:.2e} m/s)",
                savefile=f"{outdir}/corner_olivineA_pathline_prescribed.png",
                markers=("o", "v", "s", "p"),
                labels=labels,
                xlims=(0, domain_width + 0.25),
                zlims=(-domain_height, 0),
            )
