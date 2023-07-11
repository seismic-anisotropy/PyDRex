"""> PyDRex: Interpolation callbacks and helpers."""
# NOTE: Module contains delayed imports (search for 'Delayed import' comments).
import itertools as it

import numpy as np
from matplotlib.tri import CubicTriInterpolator
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

from pydrex import core as _core
from pydrex import exceptions as _err
from pydrex import vtk_helpers as _vtk


def default_interpolators(config, coords, vtk_output, mpl_interp=None):
    """Create a dictionary of default interpolator callbacks for PyDRex input.

    Args:
    - `config` (dict) — PyDRex configuration parameters
    - `coords` (2D or 3D NumPy array) — coordinates of the interpolation mesh
    - `vtk_output` (vtk{Un,S}tructuredGrid) — the VTK input grid

    Optionaly pass a string to `mpl_interp` to use
    `matplotlib.tri.CubicTriInterpolator` with the chosen smoothing algorithm.
    Only valid for 2D data with pre-existing triangulations.

    See also `create_interpolators`.

    """
    if config["mesh"]["dimension"] != coords.shape[1]:
        raise _err.MeshError(
            "dimensions of coordinate mesh and interpolation mesh must match."
            + f" You've supplied a {coords.shape[1]}D coordinate mesh,"
            + f" and a {config['mesh']['dimension']}D interpolation mesh."
        )

    is_2d = config["mesh"]["dimension"] == 2
    # Get a `vtkPointData` pointer to the actual data.
    data = vtk_output.GetPointData()

    try:
        deformation_mechanism = _vtk.read_tuple_array(
            data, "DeformationMechanism", skip_z=is_2d
        )
    except LookupError:
        n_nodes = np.product(config["mesh"]["gridnodes"])
        deformation_mechanism = np.empty(n_nodes).fill(_core.Regime.dislocation)

    velocity = _vtk.read_tuple_array(data, "Velocity", skip_z=is_2d)
    velocity_gradient = _vtk.read_tuple_array(data, "VelocityGradient", skip_z=is_2d)

    if is_2d:
        fields = (
            "velocity_x",
            "velocity_z",
            "grad_velocity_xx",
            "grad_velocity_zx",
            "grad_velocity_xz",
            "grad_velocity_zz",
            "deformation_mechanism",
        )

        if mpl_interp is not None:
            # Extract triangle vertex IDs to allow using existing triangulation.
            tri_IDs = []
            for i in range(vtk_output.GetNumberOfCells()):
                assert vtk_output.GetCell(i).GetNumberOfPoints() == 3
                IDs = vtk_output.GetCell(i).GetPointIds()
                tri_IDs.append([IDs.GetId(j) for j in range(IDs.GetNumberOfIds())])

            interpolators = dict(
                zip(
                    fields,
                    it.chain(
                        create_interpolators(
                            CubicTriInterpolator,
                            coords,
                            velocity,
                            triangles=tri_IDs,
                            kind=mpl_interp,
                        ),
                        create_interpolators(
                            CubicTriInterpolator,
                            coords,
                            velocity_gradient,
                            triangles=tri_IDs,
                            kind=mpl_interp,
                        ),
                        create_interpolators(
                            NearestNDInterpolator,
                            coords,
                            deformation_mechanism,
                        ),
                    ),
                )
            )

        else:  # 2D, mpl_interp is None
            interpolators = dict(
                zip(
                    fields,
                    it.chain(
                        *map(
                            create_interpolators,
                            (
                                CloughTocher2DInterpolator,
                                CloughTocher2DInterpolator,
                                NearestNDInterpolator,
                            ),
                            [coords] * 3,
                            (velocity, velocity_gradient, deformation_mechanism),
                        )
                    ),
                )
            )

    else:  # 3D
        interpolators = dict(
            zip(
                [
                    "velocity_x",
                    "velocity_y",
                    "velocity_z",
                    "grad_velocity_xx",
                    "grad_velocity_yx",
                    "grad_velocity_zx",
                    "grad_velocity_xy",
                    "grad_velocity_yy",
                    "grad_velocity_zy",
                    "grad_velocity_xz",
                    "grad_velocity_yz",
                    "grad_velocity_zz",
                    "deformation_mechanism",
                ],
                it.chain(
                    *map(
                        create_interpolators,
                        [NearestNDInterpolator] * 3,
                        [coords] * 3,
                        (velocity, velocity_gradient, deformation_mechanism),
                    )
                ),
            )
        )

    return interpolators


def create_interpolators(interpolator, coords, data, triangles=None, **kwargs):
    """Create interpolator callbacks for data arrays.

    Args:
    - `intepolator` (object) — interpolator class to use
    - `coords` (2D or 3D NumPy array) — coordinates of the finite element mesh
    - `data` (NumPy array) — data used to create the interpolators
    - `triangles` (optional) — 2D triangle vertex indices, see below

    Optional kwargs will be passed to the interpolation constructor.

    Returns a list of interpolator callbacks, one for each data vector component.

    The `triangles` arg is required only for `matplotlib.tri.CubicTriInterpolator`.
    See the documentation of that constructor for details.

    """
    dimension = coords.shape[1]
    assert dimension in (2, 3)

    if dimension == 2:
        if interpolator == NearestNDInterpolator:
            return [interpolator(coords, data, **kwargs)]

        if interpolator == CloughTocher2DInterpolator:
            from scipy.spatial import Delaunay  # NOTE: Delayed import.

            tri = Delaunay(coords)
        elif interpolator == CubicTriInterpolator:
            if triangles is None:
                raise ValueError(
                    "the `triangles` arg is required for CubicTriInterpolator"
                )
            from matplotlib.tri import Triangulation  # NOTE: Delayed import

            tri = Triangulation(coords[:, 0], coords[:, 1], triangles=triangles)

        return [interpolator(tri, data[:, i], **kwargs) for i in range(data.shape[1])]

    return [interpolator(coords, data[:, i], **kwargs) for i in range(data.shape[1])]


def get_velocity(point, interpolators):
    """Interpolates the velocity at a given point."""
    if len(point) == 2:
        return np.array(
            [
                interpolators["velocity_x"](*point),
                -interpolators["velocity_z"](*point),
            ]
        )

    return np.array(
        [
            interpolators["velocity_x"](*point),
            interpolators["velocity_y"](*point),
            -interpolators["velocity_z"](*point),
        ]
    )


def get_velocity_gradient(point, interpolators):
    """Return the interpolated velocity gradient tensor at a given point."""
    if len(point) == 2:
        velocity_gradient = np.array(
            [
                [
                    interpolators["grad_velocity_xx"](*point),
                    -interpolators["grad_velocity_xz"](*point),
                ],
                [
                    -interpolators["grad_velocity_zx"](*point),
                    interpolators["grad_velocity_zz"](*point),
                ],
            ]
        )
    else:
        velocity_gradient = np.array(
            [
                [
                    interpolators["grad_velocity_xx"](*point),
                    interpolators["grad_velocity_xy"](*point),
                    -interpolators["grad_velocity_xz"](*point),
                ],
                [
                    interpolators["grad_velocity_yx"](*point),
                    interpolators["grad_velocity_yy"](*point),
                    -interpolators["grad_velocity_yz"](*point),
                ],
                [
                    -interpolators["grad_velocity_zx"](*point),
                    -interpolators["grad_velocity_zy"](*point),
                    interpolators["grad_velocity_zz"](*point),
                ],
            ]
        )
    assert abs(np.trace(velocity_gradient)) < 1e-15
    return velocity_gradient


def get_deformation_mechanism(point, interpolators):
    """Return the interpolated deformation mechanism ID at the given point."""
    return interpolators["deformation_mechanism"](*point)
