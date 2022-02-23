"""PyDRex: Dynamic CPO calculations for olivine-enstatite polycrystal aggregates,
<https://github.com/Patol75/PyDRex>.

Parallel execution is achieved through multiprocessing. For distributed-memory
(multi-node) multiprocessing enable either Ray or Charm4Py. Charm4py requires an MPI
library (e.g. openmpi), and the program should be executed using `mpirun`.
Shared-memory (single-node) multiprocessing can be achieved using either Ray or the
Python standard library multiprocessing module. For shared-memory multiprocessing,
the number of CPUs should be explicitly specified.

Interpolation using Matplotlib's CubicTriInterpolator instead of Scipy's
CloughTocher2DInterpolator is supported by `--mpl-interp`. Only applies to
2D simulations for which a mesh triangulation already exists.

"""
import os
import argparse
import warnings
import itertools as it
import functools as ft
import time
import logging

from logging.handlers import WatchedFileHandler
from multiprocessing import Pool

import numpy as np

try:  # <https://pypi.org/project/colored-traceback/>
    import colored_traceback.always  # pylint: disable=unused-import
except ImportError:
    pass

HAS_CHARM = False
try:
    from charm4py import charm  # TODO: Suppress charm4py complaints when not in use.

    HAS_CHARM = True
except ImportError:
    pass

HAS_RAY = False
try:
    import ray

    HAS_RAY = True
except ImportError:
    pass

import pydrex.vtk_helpers as _vtk
import pydrex.io as _io
import pydrex.interpolations as _interp
import pydrex.exceptions as _err
import pydrex.core as _core
import pydrex.deformation_mechanism as _defmech
#import pydrex.pole_figures as _fig


def main():
    # TODO: Profiling with cProfile hooks <https://github.com/mgedmin/profilehooks>,
    #   which uses Python's cProfile <https://docs.python.org/3/library/profile.html>.
    begin = time.perf_counter()

    args = _get_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        handlers=[PIDFileHandler("pydrex.log")],
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
    logging.captureWarnings(True)

    if args.charm and args.ncpus:
        warnings.warn("ignoring --cpus for Charm4Py multiprocessing")
    if args.ray and args.ncpus:
        warnings.warn("ignoring --cpus for Ray distributed multiprocessing")

    if args.ray:
        if HAS_RAY:
            assert False, "ray multiprocessing not implemented yet..."
        else:
            raise ImportError("no module named 'ray'")
    elif args.charm:
        if HAS_CHARM:
            assert False, "charm4py multiprocessing not implemented yet..."
            # charm.start(_run_with_charm)
        else:
            raise ImportError("no module named 'charm4py'")
    elif args.ncpus:
        config, interpolators, diagnostics = _setup(args, begin)
        solve = ft.partial(_core.solve, config, interpolators)
        nodes2do = np.asarray(diagnostics["grid_mask_completed"] == 0).nonzero()
        with Pool(processes=args.ncpus) as pool:
            for results in pool.imap_unordered(solve, zip(*nodes2do)):
                _update_diagnostics(diagnostics, config, *results)
    else:
        warnings.warn("PyDRex was started without multiprocessing")
        config, interpolators, diagnostics = _setup(args, begin)
        n_total_nodes = np.prod(diagnostics["grid_mask_completed"].shape)
        logging.info("Starting main loop with %s nodes", n_total_nodes)
        for node in zip(*np.asarray(diagnostics["grid_mask_completed"] == 0).nonzero()):
            _update_diagnostics(
                diagnostics, config, *_core.solve(config, interpolators, node)
            )

    np.savez(
        f"PyDRex_{config['mesh']['dimension']}D_{config['simulation_name']}",
        **diagnostics,
    )
    #    _fig.plot_contours(
    #        f"PyDRex_{config['mesh']['dimension']}D_{config['simulation_name']}.png",
    #        diagnostics["olivine_orientations"],
    #    )

    end = time.perf_counter()
    hours = int((end - begin) / 3600)
    minutes = int((end - begin - hours * 3600) / 60)
    seconds = end - begin - hours * 3600 - minutes * 60
    logging.info("Total runtime = %s:%s:%s", hours, minutes, seconds)

    if args.charm:
        exit()


def _get_args() -> argparse.Namespace:
    description, epilog = __doc__.split(os.linesep + os.linesep, 1)
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "input",
        help="input file (.vtu or .pvtu)",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="path to configuration file with simulation parameters (default: ./pydrex.ini)",
        default="pydrex.ini",
    )
    parser.add_argument(
        "-n",
        "--ncpus",
        type=int,
        help="number of CPUs for shared-memory multiprocessing",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="enable debugging mode (verbose logging)",
    )
    mp_choices = parser.add_mutually_exclusive_group()
    mp_choices.add_argument(
        "--charm",
        action="store_true",
        help="enable Charm4Py multiprocessing",
    )
    mp_choices.add_argument(
        "--ray",
        action="store_true",
        help="enable Ray multiprocessing",
    )
    parser.add_argument(
        "--redis-pass",
        help="Redis password; required when using Ray for distributed multiprocessing",
    )
    # TODO: Consider moving this to the .ini config file.
    parser.add_argument(
        "--mpl-interp",
        choices=["geom", "min_E"],
        help="interpolate velocity and velocity gradients using Matplotlib, see below",
        metavar="{geom, min_E}",
    )
    #    parser.add_argument(
    #        "--restart",
    #        help="restart from the given NumPy checkpoint file (.npz)",
    #        metavar="checkpoint",
    #    )
    return parser.parse_args()


def _setup(args, begin):
    config = _io.parse_params(args.config)
    vtk_output = _vtk.get_output(args.input)
    coords = _vtk.read_coord_array(vtk_output)
    interpolators = _interp.default_interpolators(
        config, coords, vtk_output, args.mpl_interp
    )
    # diagnostics = _init_diagnostics(config, args.restart)
    diagnostics = _init_diagnostics(config, None)
    diagnostics["_begin"] = begin

    return config, interpolators, diagnostics


def _init_diagnostics(config, restart):
    gridshape = [x - 1 for x in config["mesh"]["gridnodes"]]
    n_volume_elements = config["number_of_grains"]
    dim = config["mesh"]["dimension"]
    # Supported output options and their required sizes.
    output_opts = {
        "olivine_volume_distribution": (*gridshape, n_volume_elements),
        "olivine_orientations": (*gridshape, n_volume_elements, dim, dim),
        "enstatite_volume_distribution": (*gridshape, n_volume_elements),
        "enstatite_orientations": (*gridshape, n_volume_elements, dim, dim),
    }
    # Error on unknown options.
    enstatite_opts = config["enstatite_output"]
    enstatite_unknown = [opt for opt in enstatite_opts if opt not in set(output_opts)]
    if enstatite_unknown:
        raise ValueError(f"unknown enstatite output options: {enstatite_unknown}")
    olivine_opts = config["olivine_output"]
    olivine_unknown = [opt for opt in olivine_opts if opt not in set(output_opts)]
    if olivine_unknown:
        raise ValueError(f"unknown olivine output options: {olivine_unknown}")

    # Construct diagnostics dict.
    diagnostics = {
        k: np.zeros(output_opts[k]) for k in it.chain(olivine_opts, enstatite_opts)
    }
    # TODO: Store a more compact representation of the strain?
    diagnostics["finite_strain_ell"] = np.empty((*gridshape, dim, dim))
    diagnostics["grid_mask_completed"] = np.zeros(gridshape)

    if restart:
        assert False, "restarting from checkpoints is not implemented yet..."
        # TODO: Read existing diagnostic values.

    return diagnostics


def _update_diagnostics(
    diagnostics,
    config,
    node,
    finite_strain_ell,
    olivine_orientations,
    enstatite_orientations,
    olivine_vol_dist,
    enstatite_vol_dist,
):
    diagnostics["finite_strain_ell"][node] = finite_strain_ell
    diagnostics["olivine_orientations"][node, :, :, :] = olivine_orientations
    diagnostics["enstatite_orientations"][node, :, :, :] = enstatite_orientations
    diagnostics["olivine_volume_distribution"][node, :] = olivine_vol_dist
    diagnostics["enstatite_volume_distribution"][node, :] = enstatite_vol_dist
    diagnostics["grid_mask_completed"][node] = 1
    n_completed_nodes = np.sum(diagnostics["grid_mask_completed"] == 1)

    begin = diagnostics["_begin"]
    now = time.perf_counter()
    hours = int((now - begin) / 3600)
    minutes = int((now - begin - hours * 3600) / 60)
    seconds = now - begin - hours * 3600 - minutes * 60

    n_total_nodes = np.prod(diagnostics["grid_mask_completed"].shape)
    logging.info(
        "Completed node %s/%s at %s:%s:%s",
        n_completed_nodes,
        n_total_nodes,
        hours,
        minutes,
        seconds,
    )


#    # TODO: Checkpointing
#    if (
#        config["checkpoint_interval"] > 0
#        and n_completed_nodes % config["checkpoint_interval"] == 0
#    ) or n_completed_nodes == np.product(config["mesh"]["gridnodes"]):
#        # Note: diagnostics.keys need to be unique, and the dictionary must be flat.
#        np.savez(
#            f"PyDRex_{config['mesh']['dimension']}D_{config['simulation_name']}_{n_completed_nodes}",
#            **diagnostics,
#        )
#        _fig.plot_contours(diagnostics["olivine_orientations"])


# def _run_with_charm(args):
#    config, interpolators, diagnostics = _setup(_get_args(), begin)
#
#    # TODO: Refactor and remove magic numbers.
#    n_batch = np.ceil(np.sum(diagnostics["grid_mask_completed"] == 0) / 6e4).astype(int)
#    for _ in range(n_batch):
#        nodes2do = np.asarray(diagnostics["grid_mask_completed"] == 0).nonzero()
#        futures = charm.pool.map_async(
#            ft.partial(_core.solve, config=config, interpolators=interpolators),
#            list(zip(*[nodes[:60000] for nodes in nodes2do])),
#            multi_future=True,
#        )
#
#        for future in charm.iwait(futures):
#            _update_diagnostics(diagnostics, config, *future.get())


# Appending to the same file from multiple processes is not safe.
class PIDFileHandler(WatchedFileHandler):
    def __init__(self, filename, mode="a", encoding=None, delay=0):
        filename = self._append_pid_to_filename(filename)
        super(PIDFileHandler, self).__init__(filename, mode, encoding, delay)

    def _append_pid_to_filename(self, filename):
        pid = os.getpid()
        path, extension = os.path.splitext(filename)
        return f"{path}-{pid}{extension}"
