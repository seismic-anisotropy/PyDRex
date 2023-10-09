"""> PyDRex: 2D corner flow tests."""
import contextlib as cl
import pathlib as pl
import functools as ft
from multiprocessing import Pool
from time import perf_counter

import numpy as np
import pytest

from pydrex import core as _core
from pydrex import diagnostics as _diagnostics
from pydrex import geometry as _geo
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import pathlines as _path
from pydrex import stats as _stats
from pydrex import visualisation as _vis
from pydrex import velocity as _velocity
from pydrex import utils as _utils

# Subdirectory of `outdir` used to store outputs from these tests.
SUBDIR = "2d_cornerflow"


class TestCornerOlivineA:
    """Tests for pure A-type olivine polycrystals in 2D corner flows."""

    class_id = "corner_olivineA"

    @classmethod
    def run(
        cls,
        params,
        seed,
        get_velocity,
        get_velocity_gradient,
        min_coords,
        max_coords,
        max_strain,
        n_timesteps,
        final_location,
    ):
        """Run 2D corner flow A-type olivine simulation."""
        mineral = _minerals.Mineral(
            _core.MineralPhase.olivine,
            _core.MineralFabric.olivine_A,
            _core.DeformationRegime.dislocation,
            n_grains=params["number_of_grains"],
        )
        deformation_gradient = np.eye(3)
        timestamps_back, get_position = _path.get_pathline(
            final_location,
            get_velocity,
            get_velocity_gradient,
            min_coords,
            max_coords,
            max_strain,
        )
        timestamps = np.linspace(timestamps_back[-1], timestamps_back[0], n_timesteps)
        positions = [get_position(t) for t in timestamps]
        velocity_gradients = [get_velocity_gradient(np.asarray(x)) for x in positions]
        strains = np.empty_like(timestamps)
        strains[0] = 0
        for t, time in enumerate(timestamps[:-1], start=1):
            strains[t] = strains[t - 1] + (
                _utils.strain_increment(timestamps[t] - time, velocity_gradients[t])
            )
            _log.info("step %d/%d (ε = %.2f)", t, len(timestamps) - 1, strains[t])

            deformation_gradient = mineral.update_orientations(
                params,
                deformation_gradient,
                get_velocity_gradient,
                pathline=(time, timestamps[t], get_position),
            )
        return timestamps, positions, strains, mineral, deformation_gradient

    @pytest.mark.slow
    def test_prescribed(self, outdir, params_Kaminski2001_fig5_shortdash, seed, ncpus):
        """Test CPO evolution in prescribed 2D corner flow.

        Initial condition: random orientations and uniform volumes in all `Mineral`s.

        Plate velocity: 2 cm/yr

        .. note::
            This example takes about 11 CPU hours to run and uses around 60GB of RAM.

        """
        # Plate speed (half spreading rate), convert cm/yr to m/s.
        plate_speed = 2.0 / (100.0 * 365.0 * 86400.0)
        domain_height = 2.0e5  # Represents the depth of olivine-spinel transition.
        domain_width = 1.0e6
        params = params_Kaminski2001_fig5_shortdash
        n_timesteps = 50  # Number of places along the pathline to compute CPO.
        get_velocity, get_velocity_gradient = _velocity.corner_2d("X", "Z", plate_speed)
        # Z-values at the end of each pathline.
        z_ends = list(map(lambda x: x * domain_height, (-0.1, -0.3, -0.54, -0.78)))

        # Optional plotting and logging setup.
        optional_logging = cl.nullcontext()
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_prescribed"
            optional_logging = _log.logfile_enable(f"{out_basepath}.log")
            npzpath = pl.Path(f"{out_basepath}.npz")
            # Clean up existing output file to prevent appending to it.
            # npzpath.unlink(missing_ok=True)  # missing_ok: Python 3.8
            labels = []
            angles = []
            indices = []
            x_paths = []
            z_paths = []
            directions = []
            timestamps = []

        _run = ft.partial(
            self.run,
            params,
            seed,
            get_velocity,
            get_velocity_gradient,
            np.array([0.0, 0.0, -domain_height]),
            np.array([domain_width, 0.0, 0.0]),
            10,
            n_timesteps,
        )

        final_locations = [np.array([domain_width, 0.0, z_exit]) for z_exit in z_ends]
        with optional_logging:
            _begin = perf_counter()
            with Pool(processes=ncpus) as pool:
                for i, out in enumerate(pool.map(_run, final_locations)):
                    times, positions, strains, mineral, deformation_gradient = out
                    _log.info("final deformation gradient:\n%s", deformation_gradient)
                    z_exit = z_ends[i]
                    if outdir is not None:
                        mineral.save(npzpath, _io.stringify(z_exit))

                    misorient_angles = np.zeros(n_timesteps)
                    bingham_vectors = np.zeros((n_timesteps, 3))
                    orientations_resampled, _ = _stats.resample_orientations(
                        mineral.orientations, mineral.fractions, seed=seed
                    )
                    misorient_indices = _diagnostics.misorientation_indices(
                        orientations_resampled,
                        _geo.LatticeSystem.orthorhombic,
                        ncpus=ncpus,
                    )
                    for idx, matrices in enumerate(orientations_resampled):
                        direction_mean = _diagnostics.bingham_average(
                            matrices,
                            axis=_minerals.OLIVINE_PRIMARY_AXIS[mineral.fabric],
                        )
                        misorient_angles[idx] = _diagnostics.smallest_angle(
                            direction_mean,
                            [1, 0, 0],
                        )
                        bingham_vectors[idx] = direction_mean

                    _log.debug(
                        "Total walltime: %s",
                        _utils.readable_timestamp(perf_counter() - _begin),
                    )

                    if outdir is not None:
                        mineral.save(npzpath, postfix=_io.stringify(z_exit))
                        labels.append(rf"$z_{{f}}$ = {z_exit/1e3:.1f} km")
                        angles.append(misorient_angles)
                        indices.append(misorient_indices)
                        x_paths.append([p[0] for p in positions])
                        z_paths.append([p[2] for p in positions])
                        timestamps.append(times)
                        directions.append(bingham_vectors)

        if outdir is not None:
            _vis.corner_flow_2d(
                x_paths,
                z_paths,
                angles,
                indices,
                directions,
                timestamps,
                xlabel=f"x ⇀ ({plate_speed:.2e} m/s)",
                savefile=f"{out_basepath}.png",
                markers=("o", "v", "s", "p"),
                labels=labels,
                xlims=(0, domain_width + 0.25 * domain_height),
                zlims=(-domain_height, 0),
            )
