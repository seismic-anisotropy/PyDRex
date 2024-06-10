"""> PyDRex: Builtin mesh generation helpers using meshio and pygmsh."""

from dataclasses import dataclass, field

import gmsh as gm
import numpy as np

from pydrex import exceptions as _err
from pydrex import geometry as _geo
from pydrex import logger as _log
from pydrex import utils as _utils


@dataclass
class Model:
    """A context manager for using the gmsh model API.

    Examples:

    >>> with Model("example_model", 2, _write_file=False) as model:
    ...     model.point_constraints = [
    ...         (0, -10, 0, 0.1),  # x, y, z, nearby_edge_length
    ...         (10, -10, 0, 0.1),
    ...         (10, 0, 0, 1e-2),
    ...         (0, 0, 0, 1e-2),
    ...     ]
    ...     model.add_tags()
    ...     model.add_physical_groups()
    >>> model.surface_tags
    [1]
    >>> model.physical_line_tags
    [6, 7, 8, 9]
    >>> model.physical_group_tags
    [10]

    """

    # https://gitlab.onelab.info/gmsh/gmsh/-/raw/master/api/gmsh.py

    name: str
    dim: int
    optimize_args: dict = field(default_factory=dict)
    point_constraints: list = field(default_factory=list)
    point_tags: list = field(default_factory=list)
    line_tags: list = field(default_factory=list)
    loop_tags: list = field(default_factory=list)
    surface_tags: list = field(default_factory=list)
    physical_line_tags: list = field(default_factory=list)
    physical_group_tags: list = field(default_factory=list)
    mesh_info: dict = field(default_factory=dict)
    _write_file: bool = True
    _was_entered: bool = False

    # TODO: Possible attributes worth adding, no particular order:
    # - boundary/boundaries, see gm.model.getBoundary()
    # - bounding_box, see gm.model.getBoundingBox()
    # - wrapper method for gm.model.getClosestPoint() ? maybe we need entities for that

    def __enter__(self):
        # See: <https://gitlab.onelab.info/gmsh/gmsh/-/issues/1142>
        gm.initialize(["-noenv"])  # Don't let gmsh mess with $PYTHONPATH and $PATH.
        gm.model.add(self.name)
        self._was_entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gm.model.geo.synchronize()
        self.add_physical_groups()
        gm.model.mesh.generate(self.dim)
        # Populate some mesh info for later reference.
        node_tags, node_coords, _ = gm.model.mesh.getNodes()
        self.mesh_info["node_tags"] = node_tags
        self.mesh_info["node_coords"] = node_coords.reshape((node_tags.size, 3))
        element_types, element_tags, _ = gm.model.mesh.getElements()
        self.mesh_info["element_types"] = element_types
        self.mesh_info["element_tags"] = element_tags
        edge_tags, edge_orientations = gm.model.mesh.getAllEdges()
        self.mesh_info["edge_tags"] = edge_tags
        self.mesh_info["edge_orientations"] = edge_orientations
        tri_face_tags, tri_face_nodes = gm.model.mesh.getAllFaces(3)
        self.mesh_info["tri_face_tags"] = tri_face_tags
        self.mesh_info["tri_face_nodes"] = tri_face_nodes
        quad_face_tags, quad_face_nodes = gm.model.mesh.getAllFaces(4)
        self.mesh_info["quad_face_tags"] = quad_face_tags
        self.mesh_info["quad_face_nodes"] = quad_face_nodes
        if len(self.optimize_args) > 0:
            gm.model.mesh.optimize(**self.optimize_args)
        _log.info(
            f"Line tags (anticlockwise from bottom left): {self.physical_line_tags}"
        )
        _log.info(f"Surface tags (check order???): {self.physical_group_tags}")
        if self._write_file:
            gm.write(f"{self.name}.msh")
        gm.finalize()

    def add_tags(self):
        """Attempt to generate necessary tags (for rectangular meshes) automatically."""
        if self._was_entered:
            if self.dim == 3:
                raise _err.ModelContextError(
                    "cannot generate automatic tags for 3D model."
                )

            self.point_tags = [
                gm.model.geo.addPoint(*point) for point in self.point_constraints
            ]
            for i, point in enumerate(self.point_tags):
                self.line_tags.append(
                    gm.model.geo.addLine(
                        point, self.point_tags[(i + 1) % len(self.point_tags)]
                    )
                )
            self.loop_tags.append(gm.model.geo.addCurveLoop(self.line_tags))
            self.surface_tags.append(gm.model.geo.addPlaneSurface([self.loop_tags[-1]]))

        else:
            raise _err.ModelContextError(
                "cannot generate tags of uninitialized model. Have you used a `with` block?"
            )

    def add_physical_groups(self):
        """Add physical group tags that are used by external tools (e.g. Fluidity)."""
        if self._was_entered:
            if self.dim == 3:
                raise _err.ModelContextError(
                    "cannot generate automatic physical groups for 3D model."
                )
            # First arg to addPhysicalGroup() is the component dimension: 0D, 1D, 2D or 3D
            self.physical_line_tags = [
                gm.model.addPhysicalGroup(1, [line]) for line in self.line_tags
            ]
            self.physical_group_tags = [
                gm.model.addPhysicalGroup(2, [self.surface_tags[-1]])
            ]
        else:
            raise _err.ModelContextError(
                "cannot add physical groups to uninitialized model. Have you used a `with` block?"
            )


def rectangle(
    name: str,
    ref_axes: tuple[str, str],
    center: tuple[float, float],
    width: float,
    height: float,
    resolution: dict,
    **kwargs,
) -> Model:
    """Generate a rectangular (2D) mesh.

    - `name` — the name of the mesh object and the base of the filename (without the
      file extension)
    - `ref_axes` — two letters from {"x", "y", "z"} that set the conventional labels for
      the dimensions of the 2D mesh, and indicate where to insert the dummy index in the
      `custom_constraints` optional keyword arg
    - `width` — size of the mesh along the first dimension according to `ref_axes`
    - `height` — size of the mesh along the second dimension according to `ref_axes`
    - `resolution` — dictionary for convenient definition of standard resolution
      constraints, with float values for any of the keys "global", "north", "south",
      "east", "west", "north-east", "north-west", "south-east" and "south-west"
      (settings are applied in this order!)

    Returns the `Model` object containing mesh information, and writes the mesh to a
    file. The optional keyword argument `custom_constraints` accepts a tuple containing
    1. the indices in `Model.point_constraints` before which to insert the custom
        constraints (by default the four corner points are ordered anti-clockwise from
        the bottom left for meshes with $xᵢ> 0$).
    2. a 2D array of shape Nx3 where rows define constraints `[x1, x2, resolution]`.

    Remaining keyword arguments are passed to the `Model` constructor.

    Examples:

    >>> rect = rectangle(
    ...     "test_rect",
    ...     ("x", "z"),
    ...     center=(0, 0),
    ...     width=1,
    ...     height=1,
    ...     resolution={"global": 1e-2},
    ...     _write_file=False
    ... )
    >>> rect.dim
    2
    >>> rect.name
    'test_rect'
    >>> rect.line_tags
    [1, 2, 3, 4]
    >>> rect.loop_tags
    [1]
    >>> [p[-1] for p in rect.point_constraints]
    [0.01, 0.01, 0.01, 0.01]

    >>> rect = rectangle(
    ...     "test_rect",
    ...     ("x", "z"),
    ...     center=(0, 0),
    ...     width=1,
    ...     height=1,
    ...     resolution={"north": 1e-2, "south": 1e-3},
    ...     _write_file=False
    ... )
    >>> [p[-1] for p in rect.point_constraints]
    [0.001, 0.001, 0.01, 0.01]

    >>> rect = rectangle(
    ...     "test_rect",
    ...     ("x", "z"),
    ...     center=(0, 0),
    ...     width=1,
    ...     height=1,
    ...     resolution={"north-west": 1e-3, "south-east": 1e-2},
    ...     _write_file=False
    ... )
    >>> rect.point_constraints[1][-1]
    0.01
    >>> rect.point_constraints[3][-1]
    0.001
    >>> rect.point_constraints[0][-1] == rect.point_constraints[2][-1]
    True
    >>> rect.point_constraints[0][-1]
    0.0055

    """

    h, v = _geo.to_indices2d(*ref_axes)
    center_h, center_v = center
    point_constraints = np.zeros((4, 4))  # x, y, z, nearby_edge_length
    _loc_map = {
        "global": range(4),
        "north": (2, 3),
        "south": (0, 1),
        "east": (1, 2),
        "west": (0, 3),
        "north-east": (2,),
        "north-west": (3,),
        "south-east": (1,),
        "south-west": (0,),
    }
    for i, p in enumerate(point_constraints):
        p[-1] = np.mean(list(resolution.values()))
        for k, res in resolution.items():
            if i in _loc_map[k]:
                p[-1] = res
        match i:
            case 0:
                p[h] = center_h - width / 2
                p[v] = center_v - height / 2
            case 1:
                p[h] = center_h + width / 2
                p[v] = center_v - height / 2
            case 2:
                p[h] = center_h + width / 2
                p[v] = center_v + height / 2
            case 3:
                p[h] = center_h - width / 2
                p[v] = center_v + height / 2

    custom_constraints = kwargs.pop("custom_constraints", None)
    if custom_constraints is not None:
        dummy_index = ({0, 1, 2} - set([h, v])).pop()
        indices, constraints = custom_constraints
        point_constraints = np.insert(
            point_constraints,
            indices,
            [_utils.add_dim(c, dummy_index) for c in constraints],
            axis=0,
        )

    _log.debug("creating mesh with point constraints:\n\t%s", point_constraints)

    with Model(name, 2, **kwargs) as model:
        model.point_constraints = point_constraints
        model.add_tags()
        model.add_physical_groups()
    return model


# def orthopiped():
#     """Generate an orthopiped (3D “box”) mesh."""
#     ...
#
#
# def annulus():
#     """Generate an annulus (2D) mesh."""
#     ...
#
#
# def annulus_part():
#     """Generate an annulus segment (2D) mesh."""
#     ...
