"""> PyDRex: Tests for VTK readers and helpers."""
import numpy as np

from pydrex import vtk_helpers as _vtk


def test_vtk_2d_array_shapes(steady_flow_models):
    for file in steady_flow_models.glob("*.vtu"):
        vtk_output = _vtk.get_output(file)
        data = vtk_output.GetPointData()
        coords = _vtk.read_coord_array(vtk_output, skip_empty=True)
        assert coords.shape[1] == 2

        # Check that there are no empty (all zero) columns.
        assert np.all(coords[:, np.any(coords, axis=0)] == coords)

        # Check that vector data is truncated to 2 components.
        velocity_xz = _vtk.read_tuple_array(data, "Velocity", skip3=True)
        assert velocity_xz.shape[1] == 2

        # Check that matrix data is truncated to 4 components.
        velocity_grad_xz = _vtk.read_tuple_array(data, "VelocityGradient", skip3=True)
        assert velocity_grad_xz.shape[1] == velocity_grad_xz.shape[2] == 2
