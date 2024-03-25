"""> PyDRex: Builtin mesh generation helpers using meshio and pygmsh."""

from dataclasses import dataclass, field

import gmsh as gm
import numpy as np

from pydrex import exceptions as _err
from pydrex import geometry as _geo
from pydrex import logger as _log


@dataclass
class Model:
    """An object-oriented gmsh model API.

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
    _write_file: bool = True
    _was_entered: bool = False

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


def rectangle(name, ref_axes, center, width, height, resolution):
    """Generate a rectangular (2D) mesh."""

    # TODO: Support resolution gradients like:
    # resolution_gradient=(1e-2, "radial_grow")  # from 1e-2 at the center to `resolution` at the edges
    # resolution_gradient=(1e-2, "radial_shrink")  # opposite of the above
    # resolution_gradient=(1e-2, "south")  # from `resolution` at the top to 1e-2 at the bottom
    # resolution_gradient=(1e-2, "west")
    # default should be resolution_gradient=None

    h, v = _geo.to_indices(*ref_axes)
    center_h, center_v = center
    point_constraints = np.zeros((4, 4))  # x, y, z, nearby_edge_length
    for i, p in enumerate(point_constraints):
        p[-1] = resolution
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

    with Model(name, 2) as model:
        model.point_constraints = point_constraints
        model.add_tags()
        model.add_physical_groups()

#
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
