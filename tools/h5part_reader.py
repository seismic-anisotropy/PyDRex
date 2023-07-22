#!/bin/env python
"""Script to read CPO data from fluidity h5part files.

CPO data is saved to a NPZ file which can be read by `pydrex.mineral.Mineral.from_file`.
The data is extracted from particles with a `scalar_attribute_array` named 'CPO',
and saved to the NPZ file under fields named with the postfix set to the particle ID.
For example, the fields 'meta_1', 'fractions_1' and 'orientations_1' store data for the
particle with id 1. The particle positions are stored in the additional fields `x`, `y`
and `z`.

"""
import argparse
import io
import os
from zipfile import ZipFile

import h5py
import numpy as np

from pydrex import MineralFabric, MineralPhase, Mineral


def _get_args() -> argparse.Namespace:
    description, epilog = __doc__.split(os.linesep + os.linesep, 1)
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "input",
        help="input file (.h5part)",
    )
    parser.add_argument(
        "-p",
        "--phase",
        help="mineral phase, default: olivine",
        default="olivine",
    )
    parser.add_argument(
        "-f",
        "--fabric",
        help="olivine fabric type, default: A, use 'N' for enstatite",
        default="A",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="filename for the output NPZ file",
    )
    parser.add_argument(
        "-n",
        "--ngrains",
        type=int,
        help="number of grains in the polycrystal",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()

    if not args.input.endswith(".h5part"):
        raise ValueError(
            f"expected input file with .h5part extension, not {args.input}"
        )
    outfile = args.input[:-6] + "npz"  # Replace `.h5part` with `.npz`.
    if type(args.output) == str and args.output != "":
        if args.output.endswith(".npz"):
            outfile = args.output
        else:
            raise ValueError(f"can only save to NPZ format, not {args.output}")

    with h5py.File(args.input) as file:
        # Don't use the last timestep,
        # fluidity deletes the detector and just writes empty data there.
        n_timesteps = len(file.keys()) - 1

        for particle_id in file["Step#0/id"][:]:
            option_map = {
                "olivine": MineralPhase.olivine,
                "enstatite": MineralPhase.enstatite,
                "A": MineralFabric.olivine_A,
                "B": MineralFabric.olivine_B,
                "C": MineralFabric.olivine_C,
                "D": MineralFabric.olivine_D,
                "E": MineralFabric.olivine_E,
                "N": MineralFabric.enstatite_AB,
            }

            # Temporary data arrays.
            x = np.zeros(n_timesteps)
            y = np.zeros(n_timesteps)
            z = np.zeros(n_timesteps)
            orientations = np.empty((n_timesteps, args.ngrains, 3, 3))
            fractions = np.empty((n_timesteps, args.ngrains))

            for t in range(n_timesteps):
                # Extract particle position.
                x[t] = file[f"Step#{t}/x"][particle_id - 1]
                y[t] = file[f"Step#{t}/y"][particle_id - 1]
                z[t] = file[f"Step#{t}/z"][particle_id - 1]

                # Extract CPO data.
                vals = np.empty(args.ngrains * 10)
                for n in range(len(vals)):
                    vals[n] = file[f"Step#{t}/CPO{n+1}"][particle_id - 1]

                orientations[t] = np.array(
                    [
                        np.reshape(vals[n : n + 9], (3, 3))
                        for n in range(0, 9 * args.ngrains, 9)
                    ]
                )
                fractions[t] = vals[9 * args.ngrains :]

            _postfix = str(particle_id)
            _fractions = list(fractions)
            _orientations = list(orientations)
            mineral = Mineral(
                phase=option_map[args.fabric],
                fabric=option_map[args.phase],
                n_grains=args.ngrains,
                fractions_init=_fractions[0],
                orientations_init=_orientations[0],
            )
            mineral.fractions = _fractions
            mineral.orientations = _orientations
            mineral.save(outfile, postfix=_postfix)
            archive = ZipFile(outfile, mode="a", allowZip64=True)
            for key, data in zip(("x", "y", "z"), (x, y, z)):
                with archive.open(f"{key}_{_postfix}", "w", force_zip64=True) as file:
                    buffer = io.BytesIO()
                    np.save(buffer, data)
                    file.write(buffer.getvalue())
                    buffer.close()
