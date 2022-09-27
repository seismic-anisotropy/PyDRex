"""PyDRex: logger settings and boilerplate.

Python's logging module is weird and its methods don't allow us to specify
which logger to use, so just using `logging.debug` for example always uses
the "root" logger, which contains a bunch of noise from other imports/modules.

"""
import functools as ft
import logging
import pathlib as pl

import numpy as np

np.set_printoptions(
    formatter={"float_kind": np.format_float_scientific},
    linewidth=1000,
)
np.set_string_function(ft.partial(np.array2string, separator=", "), repr=False)


# To create a new logger we use getLogger as recommeded by the logging docs.
LOGGER = logging.getLogger("pydrex")
# To allow for multiple handlers at different levels, default level must be DEBUG.
LOGGER.setLevel(logging.DEBUG)
# Set up console handler.
LOGGER_CONSOLE = logging.StreamHandler()
# The format string is stored in .formatter._fmt
LOGGER_CONSOLE.setFormatter(
    logging.Formatter("%(levelname)s [%(asctime)s] %(name)s: %(message)s")
)
LOGGER_CONSOLE.setLevel(logging.INFO)
# Turn on console logger by default.
LOGGER.addHandler(LOGGER_CONSOLE)


def logfile_enable(path, level=logging.DEBUG):
    """Enable logging to a file at `path` with given `level`."""
    pl.Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger_file = logging.FileHandler(path, mode="w")
    logger_file.setFormatter(LOGGER_CONSOLE.formatter)
    logger_file.setLevel(level)
    LOGGER.addHandler(logger_file)


def critical(msg, *args, **kwargs):
    """Log a CRITICAL message in PyDRex."""
    LOGGER.critical(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log an ERROR message in PyDRex."""
    LOGGER.error(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Log a WARNING message in PyDRex."""
    LOGGER.warning(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """Log an INFO message in PyDRex."""
    LOGGER.info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """Log a DEBUG message in PyDRex."""
    LOGGER.debug(msg, *args, **kwargs)


def quiet_aliens():
    """Restrict alien loggers 👽 because I'm trying to find MY bugs, thanks."""
    # Only allow warnings or above from root logger.
    logging.getLogger().setLevel(logging.WARNING)
    # Only allow critical stuff from other things.
    for name in logging.Logger.manager.loggerDict.keys():
        if name != "pydrex":
            logging.getLogger(name).setLevel(logging.CRITICAL)
    # Numba is not in the list for some reason, I guess we can leave warnings.
    logging.getLogger("numba").setLevel(logging.WARNING)
