"""> PyDRex: Mesh, configuration and supporting data Input/Output functions.

PyDRex can read/write three kinds of plain text files:
- PyDRex configuration files, which specify simulation parameters and initial conditions
- 'SCSV' files, CSV files with YAML frontmatter for (small) scientific datasets
- Mesh files via `meshio`, to set up final mineral positions in steady flows.

SCSV files are our custom CSV files with a YAML header. The header is used for data
attribution and metadata, as well as a column type spec. There is no official spec for
SCSV files at the moment but they should follow the format of existing  SCSV files in
the `data/` folder of the source repository. For supported cell types, see
`SCSV_TYPEMAP`.

"""
import collections as c
import configparser
import csv
import functools as ft
import os
import pathlib
from importlib.resources import files

import frontmatter as fm
import meshio
import numpy as np

from pydrex import exceptions as _err
from pydrex import logger as _log

SCSV_TYPEMAP = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "complex": complex,
}
"""Mapping of supported SCSV field types to corresponding Python types."""

_SCSV_DEFAULT_TYPE = "string"
_SCSV_DEFAULT_FILL = ""


def read_scsv(file):
    """Read data from an SCSV file.

    See also `save_scsv`, `read_scsv_header`.

    """
    with open(resolve_path(file)) as fileref:
        metadata, content = fm.parse(fileref.read())
        schema = metadata["schema"]
        if not _validate_scsv_schema(schema):
            raise _err.SCSVError(
                f"unable to parse SCSV schema from '{file}'."
                + " Check logging output for details."
            )
        reader = csv.reader(
            content.splitlines(), delimiter=schema["delimiter"], skipinitialspace=True
        )

        schema_colnames = [d["name"] for d in schema["fields"]]
        header_colnames = [s.strip() for s in next(reader)]
        if not schema_colnames == header_colnames:
            raise _err.SCSVError(
                f"field names specified in schema must match CSV column headers in '{file}'."
                + f" You've supplied schema fields:\n{schema_colnames}\nCSV header:\n{header_colnames}"
            )

        Columns = c.namedtuple("Columns", schema_colnames)
        coltypes = [
            SCSV_TYPEMAP[d.get("type", _SCSV_DEFAULT_TYPE)] for d in schema["fields"]
        ]
        missingstr = schema["missing"]
        fillvals = [d.get("fill", _SCSV_DEFAULT_FILL) for d in schema["fields"]]
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

    Args:
    - `stream` — open output stream (e.g. file handle) where data should be written
    - `schema` — SCSV schema dictionary, with 'delimiter', 'missing' and 'fields' keys
    - `comments` (optional) — array of comments to be written above the schema, each on
      a new line with an '#' prefix

    See also `read_scsv`, `save_scsv`.

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
        kind = field.get("type", _SCSV_DEFAULT_TYPE)
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

    Args:
    - `file` — path to the file where the data should be written
    - `schema` — SCSV schema dictionary, with 'delimiter', 'missing' and 'fields' keys
    - `data` — data arrays (columns) of equal length

    Optional keyword arguments are passed to `write_scsv_header`. See also `read_scsv`.

    """
    with open(resolve_path(file), mode="w") as stream:
        write_scsv_header(stream, schema, **kwargs)
        writer = csv.writer(
            stream, delimiter=schema["delimiter"], lineterminator=os.linesep
        )
        writer.writerow([field["name"] for field in schema["fields"]])
        fills = [field.get("fill", _SCSV_DEFAULT_FILL) for field in schema["fields"]]
        types = [
            SCSV_TYPEMAP[field.get("type", _SCSV_DEFAULT_TYPE)]
            for field in schema["fields"]
        ]
        for col in zip(*data):
            row = []
            for d, t, f in zip(col, types, fills):
                if t == bool:
                    row.append(d)
                elif t in (float, complex):
                    if np.isnan(d) and np.isnan(t(f)):
                        row.append(schema["missing"])
                    elif d == t(f):
                        row.append(schema["missing"])
                    else:
                        row.append(d)
                elif t in (int, str) and d == t(f):
                    row.append(schema["missing"])
                else:
                    row.append(d)
            writer.writerow(row)


def parse_params(file):
    """Parse an INI file containing PyDRex parameters."""
    config = configparser.ConfigParser()
    configpath = resolve_path(file)
    config.read(configpath)

    geometry = config["Geometry"]
    output = config["Output"]
    params = config["Parameters"]

    mesh = read_mesh(resolve_path(geometry.get("meshfile"), configpath.parent))
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


def read_mesh(meshfile, *args, **kwargs):
    """Wrapper of `meshio.read`, see <https://github.com/nschloe/meshio>."""
    return meshio.read(meshfile, *args, **kwargs)


def resolve_path(path, refdir=None):
    """Resolve relative paths and create parent directories if necessary.

    Relative paths are interpreted with respect to the current working directory,
    i.e. the directory from whith the current Python process was executed,
    unless a specific reference directory is provided with `refdir`.

    """
    cwd = pathlib.Path.cwd()
    if refdir is None:
        _path = cwd / path
    else:
        _path = refdir / path
    _path.parent.mkdir(parents=True, exist_ok=True)
    return _path.resolve()


def _validate_scsv_schema(schema):
    format_ok = (
        "delimiter" in schema
        and "missing" in schema
        and "fields" in schema
        and len(schema["fields"]) > 0
        and schema["delimiter"] != schema["missing"]
        and schema["delimiter"] not in schema["missing"]
    )
    if not format_ok:
        _log.error(
            "invalid format for SCSV schema: %s"
            + "\nMust contain: 'delimiter', 'missing', 'fields'"
            + "\nMust contain at least one field."
            + "\nMust contain compatible 'missing' and 'delimiter' values.",
            schema,
        )
        return False
    for field in schema["fields"]:
        if not field["name"].isidentifier():
            _log.error(
                "SCSV field name '%s' is not a valid Python identifier", field["name"]
            )
            return False
        if not field.get("type", _SCSV_DEFAULT_TYPE) in SCSV_TYPEMAP.keys():
            _log.error("unsupported SCSV field type: '%s'", field["type"])
            return False
        if (
            field.get("type", _SCSV_DEFAULT_TYPE) not in (_SCSV_DEFAULT_TYPE, "boolean")
            and "fill" not in field
        ):
            _log.error("SCSV field of type '%s' requires a fill value", field["type"])
            return False
    return True


def _parse_scsv_bool(x):
    """Parse boolean from string, for SCSV files."""
    return str(x).lower() in ("yes", "true", "t", "1")


def _parse_scsv_cell(func, data, missingstr=None, fillval=None):
    if data.strip() == missingstr:
        if fillval == "NaN":
            return func(np.nan)
        return func(fillval)
    elif func == bool:
        return _parse_scsv_bool(data)
    return func(data.strip())


def stringify(s):
    """Return a cleaned version of a string for use in filenames, etc."""
    return "".join(filter(lambda c: str.isidentifier(c) or str.isdecimal(c), str(s)))


def data(folder):
    """Get resolved path to a pydrex data folder."""
    return resolve_path(files("pydrex.data") / folder)
