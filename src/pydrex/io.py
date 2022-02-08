"""PyDRex: Input/Output functions and helpers."""
import configparser
import pathlib
import runpy

import numpy as np

import pydrex.exceptions as _err


def parse_params(file):
    """Parse an INI file containing PyDRex parameters."""
    config = configparser.ConfigParser()
    configpath = _resolve_path(file)
    config.read(configpath)

    geometry = config["Geometry"]
    output = config["Output"]
    params = config["Parameters"]

    mesh = read_mesh(_resolve_path(geometry.get("meshfile"), configpath.parent))
    olivine_fraction = params.getfloat("olivine_fraction")
    enstatite_fraction = params.getfloat("enstatite_fraction", 1.0 - olivine_fraction)

    # TODO: Allow user-given order for Euler angle output (like Fraters 2021)?
    olivine_output = tuple(
        output.get("olivine", fallback="volume_fraction, rotation_matrix")
        .replace(" ", "")
        .split(",")
    )
    enstatite_output = tuple(
        output.get("enstatite", fallback="volume_fraction, rotation_matrix")
        .replace(" ", "")
        .split(",")
    )

    return dict(
        zip(
            [
                "mesh",
                "simulation_name",
                "checkpoint_interval",
                "olivine_output",
                "enstatite_output",
                "olivine_fraction",
                "enstatite_fraction",
                "stress_exponent",
                "gbm_mobility",
                "gbs_threshold",
                "nucleation_efficiency",
                "olivine_fabric",
                "number_of_grains",
            ],
            [
                mesh,
                output.get("simulation_name", fallback="simulation_name"),
                output.getint(
                    "checkpoint_interval", fallback=0
                ),  # 0 disables checkpointing
                ["olivine_" + s for s in olivine_output],  # Unique names for np.savez()
                [
                    "enstatite_" + s for s in enstatite_output
                ],  # Unique names for np.savez()
                olivine_fraction,
                enstatite_fraction,
                params.getfloat("stress_exponent"),
                params.getfloat("gbm_mobility"),
                params.getfloat("gbs_threshold"),
                params.getfloat("nucleation_efficiency"),
                params.get("olivine_fabric"),
                params.getint("number_of_grains"),
            ],
        )
    )


def _resolve_path(path, refdir=None):
    cwd = pathlib.Path.cwd()
    if refdir is None:
        _path = cwd / path
    else:
        _path = cwd / refdir / path
    if _path.is_file():
        return _path.resolve()
    raise IOError(f"file '{_path}' does not exist")


def read_mesh(meshfile):
    """Read mesh from `meshfile` into numpy arrays.

    Supported formats:
    - Python script that assigns grid coordinates (2D Numpy array, see `create_mesh`)

    """
    if meshfile.suffix == ".py":
        return create_mesh(runpy.run_path(meshfile)["gridcoords"])

    # TODO: Other mesh formats
    # - MFEM <https://mfem.org/mesh-format-v1.0/#mfem-mesh-v10>
    # - GMSH <https://gitlab.onelab.info/gmsh/gmsh>
    # Use meshio? <https://github.com/nschloe/meshio>

    raise _err.MeshError(
        f"unable to read '{meshfile}'."
        + " The header is corrupted, or the format is not supported."
    )


def create_mesh(gridcoords):
    """Create mesh for input/output or visualisation.

    Only supports rectangular grids for now.

    Args:
        `gridcoords` (array) — 2D NumPy array of the X, [Y,] Z coordinates

    """
    dimension = len(gridcoords)
    if dimension not in (2, 3):
        raise _err.MeshError(
            "mesh dimension must be 2 or 3."
            + f" You've supplied a dimension of {dimension}."
        )
    return dict(
        zip(
            [
                "dimension",
                "gridnodes",
                "gridcoords",
                "gridmin",
                "gridmax",
            ],
            [
                dimension,
                [arr.size for arr in gridcoords],
                gridcoords,
                [np.amin(arr) for arr in gridcoords],
                [np.amax(arr) for arr in gridcoords],
            ],
        )
    )
