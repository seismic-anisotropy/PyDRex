"""Generate shearbox cube meshes using the gmsh API

See also the gmsh API tutorial:
<https://gitlab.onelab.info/gmsh/gmsh/-/tree/master/demos/api>.

The API is defined in this file:
<https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/api/gmsh.py>.

"""
import argparse
import os
import pathlib

import gmsh as gm  # type: ignore


def _get_args() -> argparse.Namespace:
    description, epilog = __doc__.split(os.linesep + os.linesep, 1)
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "-o",
        "--outdir",
        help="path to the output directory, which must already exist",
        required=True,
    )
    return parser.parse_args()


def write_shearbox(outfile, edgelength, resolution):
    """Write a shearbox cube mesh to `outfile`.

    Specify the total length of one edge of the cube with `edgelength`.
    Specify the initial resolution of the mesh with `resolution`.
    It is used to set the initial edge length for the elements.

    """
    outpath = pathlib.Path(outfile)
    print(f"Creating mesh '{outfile}'...")
    gm.initialize()
    gm.model.add(outpath.stem)

    gm.option.setNumber("Mesh.MshFileVersion", 2.2)
    # gm.option.setNumber("Mesh.PartitionOldStyleMsh2", 1)
    # gm.option.setNumber("Mesh.PartitionCreateGhostCells", 1)

    # Set up coordinates for the front face: positive x-axis forward, right-handed axes
    # Points are specified as (x, y, z, target_edge_length_for_neighbouring_elements)
    front_vertices = [
        (edgelength / 2, -edgelength / 2, -edgelength / 2, resolution),
        (edgelength / 2, edgelength / 2, -edgelength / 2, resolution),
        (edgelength / 2, edgelength / 2, edgelength / 2, resolution),
        (edgelength / 2, -edgelength / 2, edgelength / 2, resolution),
    ]
    # Create points from coodinates.
    front_vertex_tags = [gm.model.geo.addPoint(*point) for point in front_vertices]
    # Create lines joining pairs of points.
    front_line_tags = [
        gm.model.geo.addLine(point, front_vertex_tags[(i + 1) % len(front_vertex_tags)])
        for i, point in enumerate(front_vertex_tags)
    ]
    # Create front face.
    front_face_tag = gm.model.geo.addPlaneSurface(
        [gm.model.geo.addCurveLoop(front_line_tags)]
    )
    # Create remaining tags using extrusion.
    # The returned sequence contains tuples of the form (dimension, tag) for:
    # - the last extruded component entity at index 0 (read it again),
    # - the extruded entity itself at index 1, and
    # - the remaining components at successive indices.
    # The ordering of the remaining components is not documented... ¯\_(ツ)_/¯
    # Seems to be anticlockwise from the bottom side for a cube
    # i.e. bottom, right, top, left
    # See also <https://gmsh.info/doc/texinfo/gmsh.html#Extrusions>
    (
        (_, back_face_tag),
        (_, volume_tag),
        (_, bot_face_tag),
        (_, right_face_tag),
        (_, top_face_tag),
        (_, left_face_tag),
    ) = gm.model.geo.extrude(
        [(2, front_face_tag)], -edgelength, 0, 0, [round(edgelength / resolution)]
    )

    gm.model.geo.synchronize()

    # First arg to addPhysicalGroup() is the component dimension: 0D, 1D, 2D or 3D
    # front_phys_tag = gm.model.addPhysicalGroup(2, [front_face_tag])
    # gm.model.setPhysicalName(2, front_phys_tag, "Front")
    # back_phys_tag = gm.model.addPhysicalGroup(2, [back_face_tag])
    # gm.model.setPhysicalName(2, back_phys_tag, "Back")
    # bot_phys_tag = gm.model.addPhysicalGroup(2, [bot_face_tag])
    # gm.model.setPhysicalName(2, bot_phys_tag, "Bottom")
    # right_phys_tag = gm.model.addPhysicalGroup(2, [right_face_tag])
    # gm.model.setPhysicalName(2, right_phys_tag, "Right")
    # top_phys_tag = gm.model.addPhysicalGroup(2, [top_face_tag])
    # gm.model.setPhysicalName(2, top_phys_tag, "Top")
    # left_phys_tag = gm.model.addPhysicalGroup(2, [left_face_tag])
    # gm.model.setPhysicalName(2, left_phys_tag, "Left")

    gm.model.mesh.generate(3)  # Generate 3D mesh
    # Using pathlib allows relative paths and it's a nice API.
    gm.write(f"{outpath.resolve()}")

    # print(f"Front surface tag: {front_phys_tag}")
    # print(f"Back surface tag: {back_phys_tag}")
    # print(f"Bottom surface tag: {bot_phys_tag}")
    # print(f"Right surface tag: {right_phys_tag}")
    # print(f"Top surface tag: {top_phys_tag}")
    # print(f"Left surface tag: {left_phys_tag}")

    gm.finalize()


if __name__ == "__main__":
    ARGS = _get_args()
    write_shearbox(f"{ARGS.outdir}/shearbox_1m.msh", 1, 0.1)
    write_shearbox(f"{ARGS.outdir}/shearbox_100km.msh", 1e5, 1e4)
