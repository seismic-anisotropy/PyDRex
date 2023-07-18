"""> PyDRex: Miscellaneous utility methods."""
import numpy as np


def skip_nans(a):
    """Skip NaN values in array."""
    a = np.asarray(a)
    return a[~np.isnan(a)]


def angle_fse_simpleshear(strain):
    """Get angle of FSE long axis anticlockwise from the X axis in simple shear."""
    return np.rad2deg(np.arctan(np.sqrt(strain**2 + 1) + strain))
