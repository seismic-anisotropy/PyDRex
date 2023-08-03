"""> PyDRex: Miscellaneous utility methods."""
from datetime import datetime

import numpy as np


def remove_nans(a):
    """Remove NaN values from array."""
    a = np.asarray(a)
    return a[~np.isnan(a)]


def readable_timestamp(timestamp, tformat="%H:%M:%S"):
    """Convert timestamp in fractional seconds to human readable format."""
    return datetime.fromtimestamp(timestamp).strftime(tformat)


def angle_fse_simpleshear(strain):
    """Get angle of FSE long axis anticlockwise from the X axis in simple shear."""
    return np.rad2deg(np.arctan(np.sqrt(strain**2 + 1) + strain))
