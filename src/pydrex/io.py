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
import contextlib as cl
import csv
import functools as ft
import io
import itertools as it
import logging
import os
import pathlib
import re
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from importlib.resources import files

import h5py
import meshio
import numpy as np
import yaml
from tqdm import tqdm

from pydrex import core as _core
from pydrex import exceptions as _err
from pydrex import logger as _log
from pydrex import utils as _utils
from pydrex import velocity as _velocity

SCSV_TYPEMAP = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "complex": complex,
}
"""Mapping of supported SCSV field types to corresponding Python types."""

SCSV_TERSEMAP = {
    "s": "string",
    "i": "integer",
    "f": "float",
    "b": "boolean",
    "c": "complex",
}
"""Mapping of supported terse format SCSV field types to their standard names."""

_SCSV_DEFAULT_TYPE = "string"
_SCSV_DEFAULT_FILL = ""


def extract_h5part(
    file, phase: _core.MineralPhase, fabric: _core.MineralFabric, n_grains: int, output
):
    """Extract CPO data from Fluidity h5part file and save to canonical formats."""
    from pydrex.minerals import Mineral

    with h5py.File(file, "r") as f:
        for particle_id in f["Step#0/id"][:]:
            # Fluidity writes empty arrays to the particle data after they are deleted.
            # We need only the timesteps before deletion of this particle.
            steps = []
            for k in sorted(list(f.keys()), key=lambda s: int(s.lstrip("Step#"))):
                if f[f"{k}/x"].shape[0] >= particle_id:
                    steps.append(k)

            # Temporary data arrays.
            n_timesteps = len(steps)
            x = np.zeros(n_timesteps)
            y = np.zeros(n_timesteps)
            z = np.zeros(n_timesteps)
            orientations = np.empty((n_timesteps, n_grains, 3, 3))
            fractions = np.empty((n_timesteps, n_grains))

            strains = np.zeros(n_timesteps)
            for t, k in enumerate(
                tqdm(steps, desc=f"Extracting particle {particle_id}")
            ):
                # Extract particle position.
                x[t] = f[f"{k}/x"][particle_id - 1]
                y[t] = f[f"{k}/y"][particle_id - 1]
                z[t] = f[f"{k}/z"][particle_id - 1]

                # Extract CPO data.
                strains[t] = f[f"{k}/CPO_{n_grains * 10 + 1}"][particle_id - 1]
                vals = np.empty(n_grains * 10)
                for n in range(len(vals)):
                    vals[n] = f[f"{k}/CPO_{n+1}"][particle_id - 1]

                orientations[t] = np.array(
                    [
                        np.reshape(vals[n : n + 9], (3, 3))
                        for n in range(0, 9 * n_grains, 9)
                    ]
                )
                fractions[t] = vals[9 * n_grains :]

            _postfix = str(particle_id)
            _fractions = list(fractions)
            _orientations = list(orientations)
            mineral = Mineral(
                phase=phase,
                fabric=fabric,
                n_grains=n_grains,
                fractions_init=_fractions[0],
                orientations_init=_orientations[0],
            )
            mineral.fractions = _fractions
            mineral.orientations = _orientations
            mineral.save(output, postfix=_postfix)
            save_scsv(
                output[:-4] + f"_{_postfix}" + ".scsv",
                {
                    "delimiter": ",",
                    "missing": "-",
                    "fields": [
                        {
                            "name": "strain",
                            "type": "float",
                            "unit": "percent",
                            "fill": np.nan,
                        },
                        {
                            "name": "x",
                            "type": "float",
                            "unit": "m",
                            "fill": np.nan,
                        },
                        {
                            "name": "y",
                            "type": "float",
                            "unit": "m",
                            "fill": np.nan,
                        },
                        {
                            "name": "z",
                            "type": "float",
                            "unit": "m",
                            "fill": np.nan,
                        },
                    ],
                },
                [strains * 200, x, y, z],
            )


@_utils.defined_if(sys.version_info >= (3, 12))
def parse_scsv_schema(terse_schema: str) -> dict:
    """Parse terse scsv schema representation and return the expanded schema.

    The terse schema is useful for command line tools and can be specified in a single
    line of text. However, there are some limitations compared to using a Python
    dictionary, all of which are edge cases and not recommended usage:
    - the delimiter cannot be the character `d` or the character `m`
    - the missing data encoding cannot be the character `m`
    - fill values are not able to contain the colon (`:`) character
    - the arbitrary unit/comment for any field is not able to contain parentheses

    The delimiter is specified after the letter `d` and the missing data encoding after
    `m`. These are succeeded by the column specs which are a sequence of column names
    (which must be valid Python identifiers) and their (optional) data type, missing
    data fill value, and unit/comment.

    .. note:: This function is only defined if the version of your Python interpreter is
        greater than 3.11.x.

    >>> #                delimiter
    >>> #                | missing data encoding    column specifications
    >>> #                | |  ______________________|______________________________
    >>> #                v v /                                                     `
    >>> schemastring = "d,m-:colA(s)colB(s:N/A:...)colC()colD(i:999999)colE(f:NaN:%)"
    >>> schema = parse_scsv_schema(schemastring)
    >>> schema["delimiter"]
    ','
    >>> schema["missing"]
    '-'
    >>> schema["fields"][0]
    {'name': 'colA', 'type': 'string', 'fill': ''}
    >>> schema["fields"][1]
    {'name': 'colB', 'type': 'string', 'fill': 'N/A', 'unit': '...'}
    >>> schema["fields"][2]
    {'name': 'colC', 'type': 'string', 'fill': ''}
    >>> schema["fields"][3]
    {'name': 'colD', 'type': 'integer', 'fill': '999999'}
    >>> schema["fields"][4]
    {'name': 'colE', 'type': 'float', 'fill': 'NaN', 'unit': '%'}

    """
    if not terse_schema.startswith("d"):
        raise _err.SCSVError(
            "terse schema must start with delimiter specification (format: d<delimiter>)"
        )
    i_cols = terse_schema.find(":")
    if i_cols < 4:
        raise _err.SCSVError(
            "could not parse missing data encoding from terse SCSV schema"
        )
    i_missing = terse_schema.find("m", 0, i_cols)
    if i_missing < 2:
        raise _err.SCSVError(
            "could not parse missing data encoding from terse SCSV schema"
        )

    delimiter = terse_schema[1:i_missing]
    missing = terse_schema[i_missing + 1 : i_cols]

    raw_colspecs = re.split(r"\(|\)", terse_schema[i_cols + 1 :])
    raw_colspecs.pop()  # Get rid of additional last empty string element.
    if len(raw_colspecs) < 2:
        raise _err.SCSVError("failed to parse any fields from terse SCSV schema")
    if len(raw_colspecs) % 2 != 0:
        raise _err.SCSVError("invalid field specifications in terse SCSV schema")

    fields = []
    for name, spec in it.batched(raw_colspecs, 2):
        _spec = spec.split(":")
        _type = _SCSV_DEFAULT_TYPE
        if _spec[0] != "":
            try:
                _type = SCSV_TERSEMAP[_spec[0]]
            except KeyError:
                raise _err.SCSVError(
                    f"invalid field type {_spec[0]} in terse SCSV schema"
                ) from None
        field = {
            "name": name,
            "type": _type,
            "fill": _spec[1] if len(_spec) > 1 else _SCSV_DEFAULT_FILL,
        }
        if len(_spec) == 3:
            field["unit"] = _spec[2]
        fields.append(field)
    return {"delimiter": delimiter, "missing": missing, "fields": fields}


def read_scsv(file):
    """Read data from an SCSV file.

    Prints the YAML header section to output and returns a NamedTuple with columns of
    the csv data. See also `save_scsv`.

    """
    with open(resolve_path(file)) as fileref:
        yaml_lines = []
        csv_lines = []

        is_yaml = False
        for line in fileref:
            if line == "\n":  # Empty lines are skipped.
                continue
            if line == "---\n":
                if is_yaml:
                    is_yaml = False  # Second --- ends YAML section.
                    continue
                else:
                    is_yaml = True  # First --- begins YAML section.
                    continue

            if is_yaml:
                yaml_lines.append(line)
            else:
                csv_lines.append(line)

        metadata = yaml.safe_load(io.StringIO("".join(yaml_lines)))
        schema = metadata["schema"]
        if not _validate_scsv_schema(schema):
            raise _err.SCSVError(
                f"unable to parse SCSV schema from '{file}'."
                + " Check logging output for details."
            )
        reader = csv.reader(
            csv_lines, delimiter=schema["delimiter"], skipinitialspace=True
        )

        schema_colnames = [d["name"] for d in schema["fields"]]
        header_colnames = [s.strip() for s in next(reader)]
        if not schema_colnames == header_colnames:
            raise _err.SCSVError(
                f"schema field names must match column headers in '{file}'."
                + f" You've supplied schema fields\n{schema_colnames}"
                + f"\n with column headers\n{header_colnames}"
            )

        _log.info("reading SCSV file: %s", resolve_path(file))
        Columns = c.namedtuple("Columns", schema_colnames)
        # __dict__() and __slots__() of NamedTuples is empty :(
        # Set up some pretty printing instead to give a quick view of column names.
        Columns.__str__ = lambda self: f"Columns: {self._fields}"
        Columns._repr_pretty_ = lambda self, p, _: p.text(f"Columns: {self._fields}")
        # Also add some extra attributes to inspect the schema and yaml header.
        Columns._schema = schema
        Columns._metadata = (
            "".join(yaml_lines)
            .replace("# ", "")
            .replace("-\n", "")
            .replace("\n", " ")
            .rsplit("schema:", maxsplit=1)[0]  # Assumes comments are above the schema.
        )
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
                for f, fill, x in zip(
                    coltypes, fillvals, zip(*list(reader), strict=True), strict=True
                )
            ]
        )


def write_scsv_header(stream, schema, comments=None):
    """Write YAML header to an SCSV stream.

    - `stream` — open output stream (e.g. file handle) where data should be written
    - `schema` — SCSV schema dictionary, with 'delimiter', 'missing' and 'fields' keys
    - `comments` (optional) — array of comments to be written above the schema, each on
      a new line with an '#' prefix

    See also `read_scsv`, `save_scsv`.

    """
    if not _validate_scsv_schema(schema):
        raise _err.SCSVError(
            "refusing to write invalid schema to stream."
            + " Check logging output for details."
        )

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

    - `file` — path to the file where the data should be written
    - `schema` — SCSV schema dictionary, with 'delimiter', 'missing' and 'fields' keys
    - `data` — data arrays (columns) of equal length

    Optional keyword arguments are passed to `write_scsv_header`. See also `read_scsv`.

    """
    path = resolve_path(file)
    n_rows = len(data[0])
    for col in data[1:]:
        if len(col) != n_rows:
            raise _err.SCSVError(
                "refusing to write data columns of unequal length to SCSV file"
            )

    _log.info("writing to SCSV file: %s", file)
    try:  # Check that the output is valid by attempting to parse.
        with open(path, mode="w") as stream:
            write_scsv_header(stream, schema, **kwargs)
            fills = [
                field.get("fill", _SCSV_DEFAULT_FILL) for field in schema["fields"]
            ]
            types = [
                SCSV_TYPEMAP[field.get("type", _SCSV_DEFAULT_TYPE)]
                for field in schema["fields"]
            ]
            names = [field["name"] for field in schema["fields"]]
            writer = csv.writer(
                stream, delimiter=schema["delimiter"], lineterminator=os.linesep
            )
            writer.writerow(names)

            # No need for strict=True here since column lengths were already checked.
            for col in zip(*data):
                row = []
                for i, (d, t, f) in enumerate(zip(col, types, fills, strict=True)):
                    try:
                        _parse_scsv_cell(
                            t, str(d), missingstr=schema["missing"], fillval=f
                        )
                    except ValueError:
                        raise _err.SCSVError(
                            f"invalid data for column '{names[i]}'."
                            + f" Cannot parse {d} as type '{t.__qualname__}'."
                        ) from None
                    if isinstance(t, bool):
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
    except ValueError:
        path.unlink(missing_ok=True)
        raise _err.SCSVError(
            "number of fields declared in schema does not match number of data columns."
            + f" Declared schema fields were {names}; got {len(data)} data columns"
        ) from None


def parse_config(path):
    """Parse a TOML file containing PyDRex configuration."""
    path = resolve_path(path)
    _log.info("parsing configuration file: %s", path)
    with open(path, "rb") as file:
        toml = tomllib.load(file)

    # Use provided name or set randomized default.
    toml["name"] = toml.get(
        "name", f"pydrex.{np.random.default_rng().integers(1,1e10)}"
    )

    toml["parameters"] = _parse_config_params(toml)
    _params = toml["parameters"]
    toml["input"] = _parse_config_input_common(toml, path)
    _input = toml["input"]

    if "mesh" in _input:
        # Input option 1: velocity gradient mesh + final particle locations.
        _input = _parse_config_input_steadymesh(_input, path)
    elif "velocity_gradient" in _input:
        # Input option 2: velocity gradient callable + initial locations.
        _input = _parse_config_input_calcpaths(_input, path)
    elif "paths" in _input:
        # Input option 3: NPZ or SCSV files with pre-computed input pathlines.
        _input = _parse_config_input_postpaths(_input, path)
    else:
        _input["paths"] = None

    # Output fields are optional, default: most data output, least logging output.
    _output = toml.get("output", {})
    if "directory" in _output:
        _output["directory"] = resolve_path(_output["directory"], path.parent)
    else:
        _output["directory"] = resolve_path(pathlib.Path.cwd())

    # Raw output means rotation matrices and grain volumes.
    _parse_output_options(_output, "raw_output", _params["phase_assemblage"])
    # Diagnostic output means texture diagnostics (strength, symmetry, mean angle).
    _parse_output_options(_output, "diagnostics", _params["phase_assemblage"])
    # Anisotropy output means hexagonal symmetry axis and ΔVp (%).
    _output["anisotropy"] = _output.get(
        "anisotropy", ["Voigt", "hexaxis", "moduli", "%decomp"]
    )

    # Optional SCSV or NPZ pathline outputs, not sensible if there are pathline inputs.
    if "paths" in _input and "paths" in _output:
        _log.warning(
            "input pathlines and output pathline filenames are mutually exclusive;"
            + " ignoring output pathline filenames"
        )
        _output["paths"] = None
    _output["paths"] = _output.get("paths", None)

    # Default logging level for all log files.
    _output["log_level"] = _output.get("log_level", "WARNING")

    return toml


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


def _parse_config_params(toml):
    """Parse DRex and other rheology parameters."""
    _params = toml.get("parameters", {})
    for key, default in _core.DefaultParams().as_dict().items():
        _params[key] = _params.get(key, default)

    # Make sure volume fractions sum to 1.
    if np.abs(np.sum(_params["phase_fractions"]) - 1.0) > 1e-16:
        raise _err.ConfigError(
            "Volume fractions of mineral phases must sum to 1."
            + f" You've provided phase_fractions = {_params['phase_fractions']}."
        )

    # Make sure all mineral phases are accounted for and valid.
    if len(_params["phase_assemblage"]) != len(_params["phase_fractions"]):
        raise _err.ConfigError(
            "All mineral phases must have an associated volume fraction."
            + f" You've provided phase_assemblage = {_params['phase_assemblage']} and"
            + f" phase_fractions = {_params['phase_fractions']}."
        )
    try:
        _params["phase_assemblage"] = tuple(
            _parse_phase(ϕ) for ϕ in _params["phase_assemblage"]
        )
    except AttributeError:
        raise _err.ConfigError(
            f"invalid phase assemblage: {_params['phase_assemblage']}"
        ) from None

    # Make sure initial olivine fabric is valid.
    try:
        _params["initial_olivine_fabric"] = getattr(
            _core.MineralFabric, "olivine_" + _params["initial_olivine_fabric"]
        )
    except AttributeError:
        raise _err.ConfigError(
            f"invalid initial olivine fabric: {_params['initial_olivine_fabric']}"
        ) from None

    # Make sure we have enough unified dislocation creep law coefficients.
    n_provided = len(_params["disl_coefficients"])
    n_required = len(_core.DefaultParams().disl_coefficients)
    if n_provided != n_required:
        raise _err.ConfigError(
            "not enough unified dislocation creep law coefficients."
            + f"You've provided {n_provided}/{n_required} coefficients."
        )
    _params["disl_coefficients"] = tuple(_params["disl_coefficients"])

    return _params


def _parse_config_input_common(toml, path):
    try:
        _input = toml["input"]
    except KeyError:
        raise _err.ConfigError(f"missing [input] section in '{path}'") from None
    if "timestep" not in _input and "paths" not in _input:
        raise _err.ConfigError(f"unspecified input timestep in '{path}'")

    _input["timestep"] = _input.get("timestep", np.nan)
    if not isinstance(_input["timestep"], float | int):
        raise _err.ConfigError(
            f"timestep must be float or int, not {type(input['timestep'])}"
        )

    _input["strain_final"] = _input.get("strain_final", np.inf)
    if not isinstance(_input["strain_final"], float | int):
        raise _err.ConfigError(
            f"final strain must be float or int, not {type(input['strain_final'])}"
        )

    return _input


def _parse_config_input_steadymesh(input, path):
    input["mesh"] = meshio.read(resolve_path(input["mesh"], path.parent))
    input["locations_final"] = read_scsv(
        resolve_path(input["locations_final"], path.parent)
    )
    if "velocity_gradient" in input:
        _log.warning(
            "input mesh and velocity gradient callable are mutually exclusive;"
            + " ignoring velocity gradient callable"
        )
    if "locations_initial" in input:
        _log.warning(
            "initial particle locations are not used for pathline interpolation"
            + " and will be ignored"
        )
    if "paths" in input:
        _log.warning(
            "input mesh and input pathlines are mutually exclusive;"
            + " ignoring input pathlines"
        )
    input["velocity_gradient"] = None
    input["locations_initial"] = None
    input["paths"] = None
    return input


def _parse_config_input_calcpaths(input, path):
    _velocity_gradient_func = getattr(_velocity, input["velocity_gradient"][0])
    input["velocity_gradient"] = _velocity_gradient_func(
        *input["velocity_gradient"][1:]
    )
    input["locations_initial"] = read_scsv(
        resolve_path(input["locations_initial"], path.parent)
    )
    if "locations_final" in input:
        _log.warning(
            "final particle locations are not used for forward advection"
            + " and will be ignored"
        )
    if "paths" in input:
        _log.warning(
            "velocity gradient callable and input pathlines are mutually exclusive;"
            + " ignoring input pathlines"
        )
    input["locations_final"] = None
    input["paths"] = None
    input["mesh"] = None
    return input


def _parse_config_input_postpaths(input, path):
    input["paths"] = [np.load(resolve_path(p, path.parent)) for p in input["paths"]]
    if "locations_initial" in input:
        _log.warning(
            "input pathlines and initial particle locations are mutually exclusive;"
            + " ignoring initial particle locations"
        )
    if "locations_final" in input:
        _log.warning(
            "input pathlines and final particle locations are mutually exclusive;"
            + " ignoring final particle locations"
        )
    input["locations_initial"] = None
    input["locations_final"] = None
    input["mesh"] = None
    return input


def _parse_output_options(output_opts, level, phase_assemblage):
    try:
        output_opts[level] = [
            getattr(_core.MineralPhase, ϕ) for ϕ in output_opts[level]
        ]
    except AttributeError:
        raise _err.ConfigError(
            f"unsupported mineral phase in '{level}' output option.\n"
            + f" You supplied the value: {output_opts[level]}.\n"
            + " Check pydrex.core.MineralPhase for supported phases."
        ) from None
    for phase in output_opts[level]:
        if phase not in phase_assemblage:
            raise _err.ConfigError(
                f"cannot output '{level}' for phase that is not being simulated"
            )


def _parse_phase(ϕ: str | _core.MineralPhase | int) -> _core.MineralPhase:
    if isinstance(ϕ, str):
        try:
            return getattr(_core.MineralPhase, ϕ)
        except AttributeError:
            raise _err.ConfigError(f"invalid phase in phase assemblage: {ϕ}") from None
    elif isinstance(ϕ, _core.MineralPhase):
        return ϕ
    elif isinstance(ϕ, int):
        try:
            return _core.MineralPhase(ϕ)
        except IndexError:
            raise _err.ConfigError(f"invalid phase in phase assemblage: {ϕ}") from None
    raise _err.ConfigError(f"invalid phase in phase assemblage: {ϕ}") from None


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
        if field.get("type", _SCSV_DEFAULT_TYPE) not in SCSV_TYPEMAP.keys():
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
    elif func.__qualname__ == "bool":
        return _parse_scsv_bool(data)
    return func(data.strip())


def stringify(s):
    """Return a cleaned version of a string for use in filenames, etc."""
    return "".join(filter(lambda c: str.isidentifier(c) or str.isdecimal(c), str(s)))


def data(directory):
    """Get resolved path to a pydrex data directory."""
    resources = files("pydrex.data")
    if (resources / directory).is_dir():
        return resolve_path(resources / directory)
    else:
        raise NotADirectoryError(f"{resources / directory} is not a directory")


@cl.contextmanager
def logfile_enable(path, level: str | int = logging.DEBUG, mode="w"):
    """Enable logging to a file at `path` with given `level`.

    See the `pydrex.logger` documentation for examples.

    Logging levels are documented here:
    - <https://docs.python.org/3/library/logging.html#logging-levels>

    """
    formatter = logging.Formatter(
        "%(levelname)s [%(asctime)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Path can be an io.TextIOWrapper or io.StringIO, for testing purposes.
    logger_file: logging.StreamHandler | logging.FileHandler
    if isinstance(path, (io.StringIO, io.TextIOWrapper)):
        _log.debug("enabling logging at %s level to IO stream")
        logger_file = logging.StreamHandler(path)
        logger_file.setFormatter(formatter)
        logger_file.setLevel(level)
        _log.LOGGER.addHandler(logger_file)
    else:
        _log.debug("enabling logging at %s level to %s", level, path)
        logger_file = logging.FileHandler(resolve_path(path), mode=mode)
        logger_file.setFormatter(formatter)
        logger_file.setLevel(level)
        _log.LOGGER.addHandler(logger_file)
    yield
    if not isinstance(path, (io.StringIO, io.TextIOWrapper)):
        logger_file.close()
    _log.LOGGER.removeHandler(logger_file)


@cl.contextmanager
def log_cli_level(level: str | int, handler: logging.Handler = _log.CONSOLE_LOGGER):
    """Set console logging handler level for current context.

    See the `pydrex.logger` documentation for examples.

    Logging levels are documented here:
    - <https://docs.python.org/3/library/logging.html#logging-levels>

    """
    default_level = handler.level
    handler.setLevel(level)
    yield
    handler.setLevel(default_level)
