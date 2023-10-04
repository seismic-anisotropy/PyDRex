"""> PyDRex: tests for CPO stability in 2D vortex and Stokes cell flows."""
import contextlib as cl
import functools as ft
from multiprocessing import Pool
from time import process_time

import numpy as np
from scipy import linalg as la
import pytest
from numpy import testing as nt

from pydrex import core as _core
from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import utils as _utils
from pydrex import velocity as _velocity
from pydrex import visualisation as _vis
from pydrex import pathlines as _path

# Subdirectory of `outdir` used to store outputs from these tests.
SUBDIR = "2d_vortex"


class TestCellOlivineA:
    """Tests for A-type olivine polycrystals in a 2D Stokes cell."""

    class_id = "cell_olivineA"

    @classmethod
    def run(
        cls,
        params,
        final_location,
        get_velocity,
        get_velocity_gradient,
        min_coords,
        max_coords,
        max_strain,
        seed=None,
    ):
        """Run 2D Stokes cell A-type olivine simulation."""
        mineral = _minerals.Mineral(
            phase=_core.MineralPhase.olivine,
            fabric=_core.MineralFabric.olivine_A,
            n_grains=params["number_of_grains"],
            seed=seed,
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
        timestamps = timestamps_back[::-1] - timestamps_back[-1]
        for t, time in enumerate(timestamps[:-1], start=1):
            _log.info("step %d/%d (t = %.2f)", t, len(timestamps) - 1, time)

            deformation_gradient = mineral.update_orientations(
                params,
                deformation_gradient,
                get_velocity_gradient,
                pathline=(time, timestamps[t], get_position),
            )
        positions = [get_position(t) for t in timestamps]
        return timestamps, positions, mineral, deformation_gradient

    def test_xz(self, outdir, seed):
        if outdir is not None:
            out_basepath = f"{outdir}/{SUBDIR}/{self.class_id}_xz"

        params = _io.DEFAULT_PARAMS
        get_velocity, get_velocity_gradient = _velocity.cell_2d(
            "X",
            "Z",
            1,
        )

        timestamps, positions, mineral, deformation_gradient = self.run(
            params,
            np.asarray([0.5, 0.0, -0.75]),
            get_velocity,
            get_velocity_gradient,
            np.asarray([-1, 0, -1]),
            np.asarray([1, 0, 1]),
            25,
            seed=seed,
        )
        angles = [
            _diagnostics.smallest_angle(
                _diagnostics.bingham_average(a, axis="a"), get_velocity(x)
            )
            for a, x in zip(mineral.orientations, positions)
        ]
        # velocity_gradients = [get_velocity_gradient(np.asarray(x)) for x in positions]
        if outdir is not None:
            fig, ax, colors = _vis.alignment(
                None,
                # [
                #     t * la.eigh((grad_v + grad_v.transpose()) / 2)[0][-1]
                #     for t, grad_v in zip(timestamps, velocity_gradients)
                # ],
                timestamps,
                [angles],
                (".",),
                (None,),
            )
            fig.savefig(_io.resolve_path(f"{out_basepath}.png"))

            figq, axq, q = _vis.pathline_box2d(
                None,
                get_velocity,
                "XZ",
                timestamps,
                positions,
                ".",
                [-1, -1],
                [1, 1],
                [10, 10],
                scale=1,
            )
            figq.savefig(_io.resolve_path(f"{out_basepath}_domain.png"))
