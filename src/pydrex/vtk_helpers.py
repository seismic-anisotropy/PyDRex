"""PyDRex: VTK wrappers and helper functions."""
import pathlib

import numpy as np
from vtk import vtkXMLUnstructuredGridReader, vtkXMLPUnstructuredGridReader


def get_output(filename):
    """Get a reference to an unstructured (XML) VTK grid stored in `filename`.

    Only supports modern vtk formats, i.e. .vtu and .pvtu files.

    """
    input_path = pathlib.Path(filename)
    if input_path.suffix == ".vtu":
        reader = vtkXMLUnstructuredGridReader()
    elif input_path.suffix == ".pvtu":
        reader = vtkXMLPUnstructuredGridReader()
    else:
        raise RuntimeError("the supplied input file format is not supported") from None

    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def _skip_column_every(a, n):
    """Skip a column in numpy array `a` every `n` indices.
    
    Examples:

    >>> a = np.eye(10)
    >>> _skip_column_every(a, 3)
    array([[1., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1.]])
    >>> _skip_column_every(a, 1)
    array([], shape=(10, 0), dtype=float64)
    
    """
    return a[:, np.mod(np.arange(a.shape[1]), n) != n - 1]


def read_tuple_array(points, fieldname, skip_z=False):
    """Read tuples from VTK `points` into a numpy array.

    Create a numpy array from tuples extracted from a vtkPointData object.

    If `skip_z` is True, the third dimension is assumed to be full of zeros.
    This is the standard behaviour for 2D fluidity model output.
    In this case, indices corresponding to the third dimension are skipped.

    Throws a LookupError if the `fieldname` string does not match any data.

    """
    data = points.GetAbstractArray(fieldname)  # Silently returns None if missing...
    if data is None:
        raise LookupError(f"unable to find data for {fieldname}")

    a = np.array([data.GetTuple(i) for i in range(data.GetNumberOfTuples())])
    if skip_z:
        return _skip_column_every(a, 3)
    return a


def read_coord_array(vtkgrid, skip_z=False, depth_conversion=True):
    """Read coordinates from `vtkgrid` into a numpy array.

    Create a numpy array with coordinates extracted from a vtk{Uns,S}tructuredGrid object.

    If `skip_z` is True, the third dimension is assumed to be full of zeros.
    This is the standard behaviour for 2D fluidity model output.
    In this case, indices corresponding to the third dimension are skipped.

    If `depth_conversion` is False, the last component is returned as a height value.

    Throws an IndexError if the requested dimension are not available.

    """
    data = vtkgrid.GetPoints().GetData()
    a = np.array([data.GetTuple(i) for i in range(data.GetNumberOfTuples())])
    if skip_z:
        coords = _skip_column_every(a, 3)
    else:
        coords = a
    if depth_conversion:
        # Convert Z coordinates from height to depth values.
        coords[:, -1] = np.amax(coords[:, -1]) - coords[:, -1]
    return coords
