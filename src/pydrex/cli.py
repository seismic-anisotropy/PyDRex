"""> PyDRex: Entry points and argument handling for command line tools.

All CLI handlers should be registered in the `CLI_HANDLERS` namedtuple,
which ensures that they will be installed as executable scripts alongside the package.

"""

import argparse
import os
from collections import namedtuple
from zipfile import ZipFile

from pydrex import exceptions as _err
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import stats as _stats
from pydrex import visualisation as _vis


class CliTool:
    """Base class for CLI tools defining the required interface."""

    def __call__(self):
        return NotImplementedError

    def _get_args(self) -> argparse.Namespace | type[NotImplementedError]:
        return NotImplementedError


class MeshGenerator(CliTool):
    """PyDRex script to generate various simple meshes.

    Only rectangular (2D) meshes are currently supported. The RESOLUTION must be a comma
    delimited set of directives of the form `<LOC>:<RES>` where `<LOC>` is a location
    specifier, i.e. either "G" (global) or a compas direction like "N", "S", "NE", etc.,
    and `<RES>` is a floating point value to be set as the resolution at that location.

    """

    def __call__(self):
        try:  # This one is dangerous, especially in CI.
            from pydrex import mesh as _mesh
        except ImportError:
            raise _err.MissingDependencyError(
                "missing optional meshing dependencies."
                + " Have you installed the package with 'pip install pydrex[mesh]'?"
            )

        args = self._get_args()

        if args.center is None:
            center = (0, 0)
        else:
            center = [float(s) for s in args.center.split(",")]
            assert len(center) == 2

        custom_points = None
        if args.custom_points is not None:
            _custom_points = [
                [*map(float, point.split(":"))]
                for point in args.custom_points.split(",")
            ]
            # Extract the insertion indices and parse into tuple.
            custom_indices = [int(point[0]) for point in _custom_points]
            custom_points = (custom_indices, [point[1:] for point in _custom_points])

        if args.kind == "rectangle":
            width, height = map(float, args.size.split(","))
            _loc_map = {
                "G": "global",
                "N": "north",
                "S": "south",
                "E": "east",
                "W": "west",
                "NE": "north-east",
                "NW": "north-west",
                "SE": "south-east",
                "SW": "south-west",
            }
            try:
                resolution = {
                    _loc_map[k]: float(v)
                    for k, v in map(lambda s: s.split(":"), args.resolution.split(","))
                }
            except KeyError:
                raise KeyError(
                    "invalid or unsupported location specified in resolution directive"
                ) from None
            except ValueError:
                raise ValueError(
                    "invalid resolution value. The format should be '<LOC1>:<RES1>,<LOC2>:<RES2>,...'"
                ) from None
            _mesh.rectangle(
                args.output[:-4],
                (args.ref_axes[0], args.ref_axes[1]),
                center,
                width,
                height,
                resolution,
                custom_constraints=custom_points,
            )

    def _get_args(self) -> argparse.Namespace:
        assert self.__doc__ is not None, f"missing docstring for {self}"
        description, epilog = self.__doc__.split(os.linesep + os.linesep, 1)
        parser = argparse.ArgumentParser(description=description, epilog=epilog)
        parser.add_argument("size", help="width,height[,depth] of the mesh")
        parser.add_argument(
            "-r",
            "--resolution",
            help="resolution for the mesh (edge length hint(s) for gmsh)",
            required=True,
        )
        parser.add_argument("output", help="output file (.msh)")
        parser.add_argument(
            "-c",
            "--center",
            help="center of the mesh as 2 or 3 comma-separated coordinates. default: (0, 0[, 0])",
            default=None,
        )
        parser.add_argument(
            "-a",
            "--ref-axes",
            help=(
                "two letters from {'x', 'y', 'z'} that specify"
                + " the horizontal and vertical axes of the mesh"
            ),
            default="xz",
        )
        parser.add_argument(
            "-k", "--kind", help="kind of mesh, e.g. 'rectangle'", default="rectangle"
        )
        parser.add_argument(
            "-p",
            "--custom-points",
            help="comma-separated custom point constraints (in the format index:x1:x2[:x3]:resolution)",
            default=None,
        )
        return parser.parse_args()


class H5partExtractor(CliTool):
    """PyDRex script to extract raw CPO data from Fluidity .h5part files.

    Fluidity saves data stored on model `particles` to an `.h5part` file.
    This script converts that file to canonical serialisation formats:
    - a `.npz` file containing the raw CPO orientations and (surrogate) grain sizes
    - an `.scsv` file containing the pathline positions and accumulated strain

    It is assumed that CPO data is stored in keys called 'CPO_<N>' in the .h5part
    data, where `<N>` is an integer in the range 1—`n_grains`. The accumulated strain is
    read from the attribute `CPO_<S>` where S=`ngrains`+1. Particle positions are read
    from the attributes `x`, `y`, and `z`.

    At the moment, dynamic changes in fabric or phase are not supported.

    """

    def __call__(self):
        args = self._get_args()
        _io.extract_h5part(
            args.input, args.phase, args.fabric, args.ngrains, args.output
        )

    def _get_args(self) -> argparse.Namespace:
        assert self.__doc__ is not None, f"missing docstring for {self}"
        description, epilog = self.__doc__.split(os.linesep + os.linesep, 1)
        parser = argparse.ArgumentParser(description=description, epilog=epilog)
        parser.add_argument("input", help="input file (.h5part)")
        parser.add_argument(
            "-p",
            "--phase",
            help="type of `pydrex.MineralPhase` (as an ordinal number); 0 by default",
            default=0,
        )
        parser.add_argument(
            "-f",
            "--fabric",
            type=int,
            help="type of `pydrex.MineralFabric` (as an ordinal number); 0 by default",
            default=0,
        )
        parser.add_argument(
            "-n",
            "--ngrains",
            help="number of grains used in the Fluidity simulation",
            type=int,
            required=True,
        )
        parser.add_argument(
            "-o",
            "--output",
            help="filename for the output NPZ file (stem also used for the .scsv)",
            required=True,
        )
        return parser.parse_args()


class NPZFileInspector(CliTool):
    """PyDRex script to show information about serialized CPO data.

    Lists the keys that should be used for the `postfix` in `pydrex.Mineral.load` and
    `pydrex.Mineral.from_file`.

    """

    def __call__(self):
        args = self._get_args()
        with ZipFile(args.input) as npz:
            names = npz.namelist()
            print("NPZ file with keys:")
            for name in names:
                if not (
                    name.startswith("meta")
                    or name.startswith("fractions")
                    or name.startswith("orientations")
                ):
                    _log.warning(f"found unknown NPZ key '{name}' in '{args.input}'")
                print(f" - {name}")

    def _get_args(self) -> argparse.Namespace:
        assert self.__doc__ is not None, f"missing docstring for {self}"
        description, epilog = self.__doc__.split(os.linesep + os.linesep, 1)
        parser = argparse.ArgumentParser(description=description, epilog=epilog)
        parser.add_argument("input", help="input file (.npz)")
        return parser.parse_args()


class PoleFigureVisualiser(CliTool):
    """PyDRex script to plot pole figures of serialized CPO data.

    Produces [100], [010] and [001] pole figures for serialized `pydrex.Mineral`s.
    If the range of indices is not specified,
    a maximum of 25 of each pole figure will be produced by default.

    """

    def __call__(self):
        try:
            args = self._get_args()
            if args.range is None:
                i_range = None
            else:
                start, stop_ex, step = (int(s) for s in args.range.split(":"))
                # Make command line start:stop:step stop-inclusive, it's more intuitive.
                i_range = range(start, stop_ex + step, step)

            density_kwargs = {"kernel": args.kernel}
            if args.smoothing is not None:
                density_kwargs["σ"] = args.smoothing

            mineral = _minerals.Mineral.from_file(args.input, postfix=args.postfix)
            if i_range is None:
                i_range = range(0, len(mineral.orientations))
                if len(i_range) > 25:
                    _log.warning(
                        "truncating to 25 timesteps (out of %s total)", len(i_range)
                    )
                    i_range = range(0, 25)

            orientations_resampled, _ = _stats.resample_orientations(
                mineral.orientations[i_range.start : i_range.stop : i_range.step],
                mineral.fractions[i_range.start : i_range.stop : i_range.step],
            )
            if args.scsv is None:
                strains = None
            else:
                strains = _io.read_scsv(args.scsv).strain[
                    i_range.start : i_range.stop : i_range.step
                ]
            _vis.polefigures(
                orientations_resampled,
                ref_axes=args.ref_axes,
                i_range=i_range,
                density=args.density,
                savefile=args.out,
                strains=strains,
                **density_kwargs,
            )
        except (argparse.ArgumentError, ValueError, _err.Error) as e:
            _log.error(str(e))

    def _get_args(self) -> argparse.Namespace:
        assert self.__doc__ is not None, f"missing docstring for {self}"
        description, epilog = self.__doc__.split(os.linesep + os.linesep, 1)
        parser = argparse.ArgumentParser(description=description, epilog=epilog)
        parser.add_argument("input", help="input file (.npz)")
        parser.add_argument(
            "-r",
            "--range",
            help="range of strain indices to be plotted, in the format start:stop:step",
            default=None,
        )
        parser.add_argument(
            "-f",
            "--scsv",
            help=(
                "path to SCSV file with a column named 'strain'"
                + " that lists shear strain percentages for each strain index"
            ),
            default=None,
        )
        parser.add_argument(
            "-p",
            "--postfix",
            help=(
                "postfix of the mineral to load,"
                + " required if the input file contains data for multiple minerals"
            ),
            default=None,
        )
        parser.add_argument(
            "-d",
            "--density",
            help="toggle contouring of pole figures using point density estimation",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "-k",
            "--kernel",
            help=(
                "kernel function for point density estimation, one of:"
                + f" {list(_stats.SPHERICAL_COUNTING_KERNELS.keys())}"
            ),
            default="linear_inverse_kamb",
        )
        parser.add_argument(
            "-s",
            "--smoothing",
            help="smoothing parameter for Kamb type density estimation kernels",
            default=None,
            type=float,
            metavar="σ",
        )
        parser.add_argument(
            "-a",
            "--ref-axes",
            help=(
                "two letters from {'x', 'y', 'z'} that specify"
                + " the horizontal and vertical axes of the pole figures"
            ),
            default="xz",
        )
        parser.add_argument(
            "-o",
            "--out",
            help="name of the output file, with either .png or .pdf extension",
            default="polefigures.png",
        )
        return parser.parse_args()


# These are not the final names of the executables (those are set in pyproject.toml).
_CLI_HANDLERS = namedtuple(
    "_CLI_HANDLERS",
    (
        "pole_figure_visualiser",
        "npz_file_inspector",
        "mesh_generator",
        "h5part_extractor",
    ),
)
CLI_HANDLERS = _CLI_HANDLERS(
    pole_figure_visualiser=PoleFigureVisualiser(),
    npz_file_inspector=NPZFileInspector(),
    mesh_generator=MeshGenerator(),
    h5part_extractor=H5partExtractor(),
)
