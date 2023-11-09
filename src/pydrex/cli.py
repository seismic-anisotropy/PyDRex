"""> PyDRex: Entry points for command line tools."""
import argparse
import os
from collections import namedtuple
from dataclasses import dataclass
from zipfile import ZipFile

from pydrex import exceptions as _err
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import minerals as _minerals
from pydrex import stats as _stats
from pydrex import visualisation as _vis

# NOTE: Register all cli handlers in the namedtuple at the end of the file.


@dataclass
class NPZFileInspector:
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
        description, epilog = self.__doc__.split(os.linesep + os.linesep, 1)
        parser = argparse.ArgumentParser(description=description, epilog=epilog)
        parser.add_argument("input", help="input file (.npz)")
        return parser.parse_args()


@dataclass
class PoleFigureVisualiser:
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
                mineral.orientations[i_range.start:i_range.stop:i_range.step],
                mineral.fractions[i_range.start:i_range.stop:i_range.step],
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


_CLI_HANDLERS = namedtuple(
    "CLI_HANDLERS",
    {
        "pole_figure_visualiser",
        "npz_file_inspector",
    },
)
CLI_HANDLERS = _CLI_HANDLERS(
    pole_figure_visualiser=PoleFigureVisualiser(), npz_file_inspector=NPZFileInspector()
)
