"""PyDRex: VTK wrappers and helper functions."""
import pathlib

import numpy as np
from vtk import vtkXMLPUnstructuredGridReader
from vtk import vtkXMLUnstructuredGridReader


def get_output(filename):
    """Get a reference to an unstructured (XML) VTK grid stored in `filename`.

    Only supports modern vtk formats, i.e. .vtu and .pvtu files.

    """
    input_path = pathlib.Path(filename).resolve()
    if input_path.suffix == ".vtu":
        reader = vtkXMLUnstructuredGridReader()
    elif input_path.suffix == ".pvtu":
        reader = vtkXMLPUnstructuredGridReader()
    else:
        raise RuntimeError("the supplied input file format is not supported") from None

    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def read_tuple_array(points, fieldname, skip3=False):
    """Read tuples from VTK `points` into a numpy array.

    Create a numpy array from tuples extracted from a vtkPointData object.
    The returned array has a shape of either (N, d) for vector data,
    or (N, d, d) for matrix data, where N is the number of VTK nodes
    and d is the dimension (3 by default, or 2 if `skip3` is True).

    If `skip3` is True, the domain is assumed to be two-dimensional,
    and any values corresponding to a third dimension are skipped.

    Throws a LookupError if the `fieldname` string does not match any data.

    """
    data = points.GetAbstractArray(fieldname)  # Silently returns None if missing...
    if data is None:
        raise LookupError(f"unable to find data for {fieldname}")

    values = np.array([data.GetTuple(i) for i in range(data.GetNumberOfTuples())])

    if values.shape[1] == 9:  # Matrix data.
        values = values.reshape((values.shape[0], 3, 3))
        if skip3:
            return values[:, :2, :2]
        return values

    if skip3:
        return values[:, :2]
    return values


def read_coord_array(vtkgrid, skip_empty=True, convert_depth=True):
    """Read coordinates from `vtkgrid` into a numpy array.

    Create a numpy array with coordinates extracted from a vtk{Uns,S}tructuredGrid object.

    If `skip_empty` is True (default), columns full of zeros are skipped.
    These are often present in VTK output files from 2D simulations.

    If `depth_conversion` is True (default), the last nonzero column is assumed
    to contain height values which are converted to depth values by subtraction
    from the maximum value.

    """
    data = vtkgrid.GetPoints().GetData()
    _coords = np.array([data.GetTuple(i) for i in range(data.GetNumberOfTuples())])
    if skip_empty:
        coords = _coords[:, np.any(_coords, axis=0)]
    else:
        coords = _coords
    if convert_depth:
        # Convert Z coordinates from height to depth values.
        coords[:, -1] = np.amax(coords[:, -1]) - coords[:, -1]
    return coords
