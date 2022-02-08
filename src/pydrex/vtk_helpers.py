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


def read_tuple_array(points, fieldname, dimensions):
    """Read tuples from a vtkPointData object into a numpy array.

    Only reads the specified number of `dimensions` (columns).
    Throws a LookupError if `fieldname` does not match any data.

    """
    data = points.GetAbstractArray(fieldname)  # Silently returns None if missing...
    if data is None:
        raise LookupError(f"unable to find data for {fieldname}")

    return np.array([data.GetTuple(i) for i in range(data.GetNumberOfTuples())])[
        :, :dimensions
    ]


def read_coord_array(vtkgrid, dimensions):
    """Read coordinates from a vtk{Uns,S}tructuredGrid into a numpy array.

    Only reads the specified number of `dimensions` (columns).
    Throws an IndexError if the requested dimensions are not available.

    """
    data = vtkgrid.GetPoints().GetData()
    coords = np.array([data.GetTuple(i) for i in range(data.GetNumberOfTuples())])[
        :, :dimensions
    ]
    # Convert Z coordinates from height to depth values.
    z = dimensions - 1
    coords[:, z] = np.amax(coords[:, z]) - coords[:, z]
    return coords
