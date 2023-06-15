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
import pathlib as pl
import time

import numba as nb
import numpy as np
from numpy import testing as nt
from scipy.spatial.transform import Rotation

from pydrex import deformation_mechanism as _defmech
from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
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

    def test_corner_prescribed_init_isotropic(
        self,
        params_Kaminski2001_fig5_shortdash,
        rng,
        stringify,
        outdir,
    ):
        """Test CPO evolution in prescribed 2D corner flow.

        Initial condition: random orientations and uniform volumes in all `Mineral`s.

        Plate velocity: 2 cm/yr

        """
        # Plate velocity (half spreading rate), convert cm/yr to m/s.
        plate_velocity = 2.0 / (100.0 * 365.0 * 86400.0)
        domain_height = 2.0e5  # Represents the depth of olivine-spinel transition.
        domain_width = 1.0e6
        n_grains = 2000
        orientations_init = Rotation.random(n_grains, random_state=rng).as_matrix()
        n_timesteps = 20  # Number of places along the pathline to compute CPO.
        # Z-values at the end of each pathline.
        z_ends = list(map(lambda x: x * domain_height, (-0.1, -0.3, -0.54, -0.78)))
        test_id = "corner_olivineA_prescribed"  # Unique string ID of this test.

        # Optional plotting and logging setup.
        optional_logging = cl.nullcontext()
        if outdir is not None:
            optional_logging = _log.logfile_enable(f"{outdir}/{test_id}.log")
            npzpath = pl.Path(f"{outdir}/{test_id}.npz")
            # Clean up existing output file to prevent appending to it.
            npzpath.unlink(missing_ok=True)  # missing_ok: Python 3.8
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
            _begin = time.perf_counter()
            for z_exit in z_ends:
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
                        pathline=(time_start, time_end, get_position),
                    )
                    x_current, _, z_current = get_position(time_start)
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
                        axis=_minerals.OLIVINE_PRIMARY_AXIS[mineral.fabric],
                    )
                    misorient_angles[idx] = _diagnostics.smallest_angle(
                        direction_mean,
                        [1, 0, 0],
                    )
                    misorient_indices[idx] = _diagnostics.misorientation_index(
                        orientations_resampled
                    )
                    bingham_vectors[idx] = direction_mean

                _log.debug(
                    "Total walltime: %s minutes", (time.perf_counter() - _begin) / 60
                )

                if outdir is not None:
                    mineral.save(npzpath, postfix=stringify(z_exit))
                    labels.append(rf"$z_{{f}}$ = {z_exit/1e3:.1f} km")
                    angles.append(misorient_angles)
                    indices.append(misorient_indices)
                    x_paths.append(x_vals)
                    z_paths.append(z_vals)
                    timestamps.append(times)
                    directions.append(bingham_vectors)

                # fmt: off
                match z_exit:
                    case -0.1:
                        nt.assert_allclose(
                            misorient_indices,
                            np.array(
                                [
                                    0.29354073, 0.29139814, 0.28965082, 0.26923181,
                                    0.43728815, 0.55902656, 0.67040709, 0.71021054,
                                    0.66081722, 0.65040127, 0.66930637, 0.67661135,
                                    0.63910248, 0.6707902, 0.66679258, 0.67516914,
                                    0.69877907, 0.72979705, 0.73528611, 0.71979801,
                                ]
                            ),
                            atol=1e-6,
                            rtol=1e-99,
                        )
                        nt.assert_allclose(
                            misorient_angles,
                            np.array(
                                [
                                    22.24261747, 24.99605342, 29.71796518, 40.0912132,
                                    58.41158265, 36.86306583, 31.87868878, 30.11087659,
                                    26.51558709, 23.34835358, 23.41384721, 21.93323976,
                                    21.48935828, 20.08908965, 19.57542213, 18.6647496,
                                    17.5105292, 15.91363738, 14.33753086, 13.41946783,
                                ]
                            ),
                            atol=1e-6,
                            rtol=1e-99,
                        )
                    case -0.3:
                        nt.assert_allclose(
                            misorient_indices,
                            np.array(
                                [
                                    0.2940009, 0.28771646, 0.27639005, 0.25886719,
                                    0.40972653, 0.50731756, 0.60316864, 0.64811264,
                                    0.69256278, 0.71008648, 0.73889148, 0.74481225,
                                    0.75434712, 0.75209596, 0.75393394, 0.7675225,
                                    0.76246927, 0.77881215, 0.77141318, 0.77484375,
                                ]
                            ),
                            atol=1e-6,
                            rtol=1e-99,
                        )
                        nt.assert_allclose(
                            misorient_angles,
                            np.array(
                                [
                                    5.91728301, 17.07815867, 23.25332998, 34.17169797,
                                    41.66395009, 36.79018269, 32.55988675, 30.14552646,
                                    27.30582272, 25.49009754, 24.4217203, 22.90840585,
                                    21.66657909, 20.68103436, 20.27783999, 20.06226786,
                                    20.20396384, 19.95005673, 20.09769095, 20.10364311,
                                ]
                            ),
                            atol=1e-6,
                            rtol=1e-99,
                        )
                    case -0.54:
                        nt.assert_allclose(
                            misorient_indices,
                            np.array(
                                [
                                    0.2944119, 0.28656134, 0.27582298, 0.25884225,
                                    0.2615378, 0.32354765, 0.44345791, 0.51995699,
                                    0.59311994, 0.63452573, 0.6788792, 0.6994935,
                                    0.71107442, 0.73233345, 0.75584854, 0.76953765,
                                    0.77943593, 0.78909139, 0.80136383, 0.7946032,
                                ]
                            ),
                            atol=1e-6,
                            rtol=1e-99,
                        )
                        nt.assert_allclose(
                            misorient_angles,
                            np.array(
                                [
                                    40.7125026, 5.52574182, 2.06510475, 9.51367731,
                                    16.48842428, 22.11392716, 24.85899564, 24.54769036,
                                    24.14955996, 23.90791376, 23.3676604, 23.07129318,
                                    22.65626005, 22.61909731, 22.63638591, 23.00285289,
                                    22.74623449, 22.79083959, 22.05867133, 22.29152601,
                                ]
                            ),
                            atol=1e-6,
                            rtol=1e-99,
                        )
                    case -0.78:
                        nt.assert_allclose(
                            misorient_indices,
                            np.array(
                                [
                                    0.29382031, 0.29141999, 0.29000587, 0.28289251,
                                    0.27760402, 0.27496249, 0.26485081, 0.26716991,
                                    0.26558728, 0.27393002, 0.29054229, 0.3169605,
                                    0.34624343, 0.38476239, 0.4217034, 0.46967351,
                                    0.48761024, 0.51957981, 0.53310085, 0.57009067,
                                ]
                            ),
                            atol=1e-6,
                            rtol=1e-99,
                        )
                        nt.assert_allclose(
                            misorient_angles,
                            np.array(
                                [
                                    74.10661435, 23.96729185, 20.59992663, 17.02088753,
                                    9.09000678, 6.04395152, 4.58153096, 2.96577108,
                                    2.9621749, 2.68839051, 7.07400922, 8.1715697,
                                    9.02297977, 11.96472443, 11.46670069, 12.97670256,
                                    13.70356205, 13.50136437, 14.73625771, 14.49643156,
                                ]
                            ),
                            atol=1e-6,
                            rtol=1e-99,
                        )
                # fmt: on

        if outdir is not None:
            # np.savez(f"{outdir}/{test_id}_diagnostics.npz", *angles, *indices)
            _vis.corner_flow_2d(
                x_paths,
                z_paths,
                angles,
                indices,
                directions,
                timestamps,
                xlabel=f"x ⇀ ({plate_velocity:.2e} m/s)",
                savefile=f"{outdir}/{test_id}.png",
                markers=("o", "v", "s", "p"),
                labels=labels,
                xlims=(0, domain_width + 0.25 * domain_height),
                zlims=(-domain_height, 0),
            )

            scsv_schema = {
                "delimiter": ",",
                "missing": "-",
                "fields": [
                    {
                        "name": f"t_{stringify(e)}",
                        "type": "float",
                        "fill": "NaN",
                        "unit": "seconds",
                    }
                    for e in z_ends
                ]
                + [
                    {"name": f"x_{stringify(e)}", "type": "float", "fill": "NaN"}
                    for e in z_ends
                ]
                + [
                    {"name": f"z_{stringify(e)}", "type": "float", "fill": "NaN"}
                    for e in z_ends
                ],
            }
            scsv_data = list(it.chain.from_iterable(zip(timestamps, x_paths, z_paths)))
            _io.save_scsv(f"{outdir}/{test_id}.scsv", scsv_schema, scsv_data)
