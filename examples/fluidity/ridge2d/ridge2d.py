import configparser
import pathlib

import numpy as np
from pydrex.utils import halfspace

config = configparser.ConfigParser()
config.read(pathlib.PurePath(__file__).with_suffix(".ini"))
init = config["initial conditions"]

PLATE_SPEED = float(init["PLATE_SPEED"])  # in cm/yr
SURFACE_TEMP = float(init["SURFACE_TEMP"])
DIFF_TEMP = float(init["DIFF_TEMP"])
THERM_DIFFUSIVITY = float(init["THERM_DIFFUSIVITY"])
THERM_FIT = init["THERM_FIT"]
HALF_WIDTH = float(init["HALF_WIDTH"])
DEPTH = float(init["DEPTH"])
MIN_AGE = float(init["MIN_AGE"])


def age(x):
    """Calculate age of lithosphere at horizontal location `x`.

    Expects `x` to be equal to half of the domain width at the ridge, not zero.
    This matches what we will get from Fluidity.

    """
    return MIN_AGE * 365.25 * 86400 + np.abs(x / PLATE_SPEED / (100 * 365.25 * 86400))


def temperature(x, z):
    """Calculate temperature at position based on half-space cooling in 2D domain."""
    return halfspace(
        age(x),
        z,
        surface_temp=SURFACE_TEMP,
        diff_temp=DIFF_TEMP,
        diffusivity=THERM_DIFFUSIVITY,
        fit=THERM_FIT,
    )
