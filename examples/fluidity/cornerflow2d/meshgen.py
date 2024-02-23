"""Generate 2D corner flow mesh using the gmsh Python API.

The mesh filename and physical surface/line tags
must match the expected values in the fluidity input file.

See also the gmsh API tutorial:
<https://gitlab.onelab.info/gmsh/gmsh/-/tree/master/tutorials/python>.

The API is defined in this file:
<https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/api/gmsh.py>.

"""
import argparse
import pathlib
import os

import gmsh as gm


def _get_args() -> argparse.Namespace:
    description, epilog = __doc__.split(os.linesep + os.linesep, 1)
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "-o",
        "--out",
        help="output path",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--width",
        help="width of the domain (+X)",
        required=True,
        type=float,
    )
    parser.add_argument(
        "-d",
        "--depth",
        help="depth of the domain (-Y)",
        required=True,
        type=float,
    )
    parser.add_argument(
        "-u",
        "--upper-res",
        help="target upper limit of the edge lengths, used near the origin (0, 0)",
        required=True,
        type=float,
    )
    parser.add_argument(
        "-l",
        "--lower-res",
        help="target lower limit of the edge lengths, used at all corners except the origin (0, 0)",
        required=True,
        type=float,
    )
    return parser.parse_args()


def write_cornerflow_2d(outfile, mesh_width, mesh_depth, mesh_res_hi, mesh_res_lo):
    """Write a 2D corner-flow model mesh to `outfile`."""
    outpath = pathlib.Path(outfile)
    outpath.resolve().parent.mkdir(parents=True, exist_ok=True)
    gm.initialize()
    gm.model.add(outpath.stem)

    # Points are specified as (x, y, z, target_edge_length_for_neighbouring_elements)
    # TODO: Use X, -Z instead of X, -Y for the 2D domain?
    point_coords = [
        (0, -mesh_depth, 0, mesh_res_lo),
        (mesh_width, -mesh_depth, 0, mesh_res_lo),
        (mesh_width, 0, 0, mesh_res_lo),
        (0, 0, 0, mesh_res_hi),
    ]

    point_tags = [gm.model.geo.addPoint(*point) for point in point_coords]

    line_tags = []
    for i, point in enumerate(point_tags):
        line_tags.append(
            gm.model.geo.addLine(point, point_tags[(i + 1) % len(point_tags)])
        )

    loop_tag = gm.model.geo.addCurveLoop(line_tags)
    surface_tag = gm.model.geo.addPlaneSurface([loop_tag])

    gm.model.geo.synchronize()

    # First arg to addPhysicalGroup() is the component dimension: 0D, 1D, 2D or 3D
    phys_line_tags = [gm.model.addPhysicalGroup(1, [line]) for line in line_tags]
    phys_group_tag = gm.model.addPhysicalGroup(2, [surface_tag])

    gm.model.mesh.generate(2)  # Generate 2D mesh
    # Using pathlib allows relative paths and it's a nice API.
    gm.write(f"{outpath}")

    print(f"Line tags (anticlockwise from bottom left): {phys_line_tags}")
    print(f"Surface tag: {phys_group_tag}")

    gm.finalize()


if __name__ == "__main__":
    ARGS = _get_args()
    write_cornerflow_2d(
        ARGS.out, ARGS.width, ARGS.depth, ARGS.upper_res, ARGS.lower_res
    )
