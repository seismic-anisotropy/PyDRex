"""> PyDRex: Input/Output functions and helpers."""
import os
import collections as c
import configparser
import csv
import functools as ft
import pathlib
import runpy

import frontmatter as fm
import numpy as np

import pydrex.exceptions as _err

SCSV_TYPEMAP = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "complex": complex,
}
"""Mapping of supported SCSV field types to corresponding Python types."""


def _validate_scsv_schema(schema):
    format_ok = (
        "delimiter" in schema
        and "missing" in schema
        and "fields" in schema
        and len(schema["fields"]) > 0
    )
    if not format_ok:
        return False
    for field in schema["fields"]:
        if not field["name"].isidentifier():
            return False
        if not field["type"] in SCSV_TYPEMAP.keys():
            return False
    return True


def _parse_scsv_cell(func, data, missingstr=None, fillval=None):
    if data.strip() == missingstr:
        if fillval == "NaN":
            return np.nan
        return func(fillval)
    return func(data.strip())


def read_scsv(file):
    """Read data from an SCSV file.

    SCSV files are our custom CSV files with a YAML header.
    The header is used for data attribution and metadata,
    as well as a column type spec.
    There is no official spec for SCSV files at the moment
    but they should follow the format of existing  SCSV files in the `data/` folder
    of the source repository.
    For supported cell types, see `SCSV_TYPEMAP`.

    """
    with open(file) as fileref:
        metadata, content = fm.parse(fileref.read())
        schema = metadata["schema"]
        if not _validate_scsv_schema(schema):
            raise _err.SCSVError(f"unable to parse SCSV schema from '{file}'")
        reader = csv.reader(content.splitlines(), delimiter=schema["delimiter"])

        schema_colnames = [d["name"] for d in schema["fields"]]
        header_colnames = [s.strip() for s in next(reader)]
        if not schema_colnames == header_colnames:
            raise _err.SCSVError(
                f"field names specified in schema must match CSV column headers in '{file}'."
                + f" You've supplied schema fields:\n{schema_colnames}\nCSV header:\n{header_colnames}"
            )

        Columns = c.namedtuple("Columns", schema_colnames)
        coltypes = [SCSV_TYPEMAP[d["type"]] for d in schema["fields"]]
        missingstr = schema["missing"]
        fillvals = [d.get("fill", "") for d in schema["fields"]]
        return Columns._make(
            [
                tuple(
                    map(
                        ft.partial(
                            _parse_scsv_cell, f, missingstr=missingstr, fillval=fill
                        ),
                        x,
                    )
                )
                for f, fill, x in zip(coltypes, fillvals, zip(*list(reader)))
            ]
        )


def write_scsv_header(stream, schema, comments=None):
    """Write YAML header to an SCSV stream.

    SCSV files are our custom CSV files with a YAML header.
    The header is used for data attribution and metadata,
    as well as a column type spec.
    There is no official spec for SCSV files at the moment
    but they should follow the format of existing  SCSV files in the `data/` folder
    of the source repository.
    For supported cell types, see `SCSV_TYPEMAP`.

    Args:
    - `stream` — open output stream (e.g. file handle) where data should be written
    - `schema` — SCSV schema dictionary, with 'delimiter', 'missing' and 'fields' keys
    - `comments` (optional) — array of comments to be written above the schema, each on
      a new line with an '#' prefix

    """
    if not _validate_scsv_schema(schema):
        raise _err.SCSVError("refusing to write invalid schema to stream")

    stream.write("---" + os.linesep)
    if comments is not None:
        for comment in comments:
            stream.write("# " + comment + os.linesep)
    stream.write("schema:" + os.linesep)
    delimiter = schema["delimiter"]
    missing = schema["missing"]
    stream.write(f"  delimiter: '{delimiter}'{os.linesep}")
    stream.write(f"  missing: '{missing}'{os.linesep}")
    stream.write("  fields:" + os.linesep)

    for field in schema["fields"]:
        name = field["name"]
        kind = field["type"]
        stream.write(f"    - name: {name}{os.linesep}")
        stream.write(f"      type: {kind}{os.linesep}")
        if "unit" in field:
            unit = field["unit"]
            stream.write(f"      unit: {unit}{os.linesep}")
        if "fill" in field:
            fill = field["fill"]
            stream.write(f"      fill: {fill}{os.linesep}")
    stream.write("---" + os.linesep)


def save_scsv(file, schema, data, **kwargs):
    """Save data to SCSV file.

    SCSV files are our custom CSV files with a YAML header.
    The header is used for data attribution and metadata,
    as well as a column type spec.
    There is no official spec for SCSV files at the moment
    but they should follow the format of existing  SCSV files in the `data/` folder
    of the source repository.
    For supported cell types, see `SCSV_TYPEMAP`.

    Args:
    - `file` — path to the file where the data should be written
    - `schema` — SCSV schema dictionary, with 'delimiter', 'missing' and 'fields' keys
    - `data` — data arrays (columns) of equal length

    Optional keyword arguments are passed to `write_scsv_header`.

    """
    with open(file, mode="w") as stream:
        write_scsv_header(stream, schema, **kwargs)
        delim = schema["delimiter"]
        stream.write(
            delim.join([field["name"] for field in schema["fields"]]) + os.linesep
        )
        for col in zip(*data):
            stream.write(delim.join([str(d) for d in col]) + os.linesep)


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
        output.get("olivine", fallback="volume_distribution, orientations")
        .replace(" ", "")
        .split(",")
    )
    enstatite_output = tuple(
        output.get("enstatite", fallback="volume_distribution, orientations")
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
                # 0 disables checkpointing
                output.getint("checkpoint_interval", fallback=0),
                # Unique names for np.savez()
                ["olivine_" + s for s in olivine_output],
                ["enstatite_" + s for s in enstatite_output],
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
    raise OSError(f"file '{_path}' does not exist")


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
                "gridsteps",
                "gridmin",
                "gridmax",
            ],
            [
                dimension,
                [arr.size for arr in gridcoords],
                gridcoords,
                _get_steps(gridcoords),
                [arr.min() for arr in gridcoords],
                [arr.max() for arr in gridcoords],
            ],
        )
    )


def _get_steps(a):
    """Get forward difference of 2D array `a`, with repeated last elements.

    The repeated last elements ensure that output and input arrays have equal shape.

    Examples:

    >>> _get_steps(np.array([1, 2, 3, 4, 5]))
    array([[1, 1, 1, 1, 1]])

    >>> _get_steps(np.array([[1, 2, 3, 4, 5], [1, 3, 6, 9, 10]]))
    array([[1, 1, 1, 1, 1],
           [2, 3, 3, 1, 1]])

    >>> _get_steps(np.array([[1, 2, 3, 4, 5], [1, 3, 6, 9, 10], [1, 0, 0, 0, np.inf]]))
    array([[ 1.,  1.,  1.,  1.,  1.],
           [ 2.,  3.,  3.,  1.,  1.],
           [-1.,  0.,  0., inf, nan]])

    """
    a2 = np.atleast_2d(a)
    return np.diff(
        a2, append=np.reshape(a2[:, -1] + (a2[:, -1] - a2[:, -2]), (a2.shape[0], 1))
    )
