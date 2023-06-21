"""> PyDRex: logger settings and boilerplate.

Python's logging module is weird and its methods don't allow us to specify
which logger to use, so just using `logging.debug` for example always uses
the "root" logger, which spams a bunch of messages from other imports/modules.
Instead, the methods in this module are thin wrappers that use custom
logging objects (`pydrex.logger.LOGGER` and `pydrex.logger.CONSOLE_LOGGER`).
The method `quiet_aliens` can be invoked to suppress most messages
from third-party modules, except critical errors and warnings from Numba.

For most applications, the `logfile_enable` context manager is recommended.
Always use the old printf style formatting for log messages, not fstrings,
otherwise the values will always be converted to strings even when logging is disabled.
Example:

```python
from pydrex import logger as _log
_log.quiet_aliens()
with _log.logfile_enable("my_log_file.log"):
    value = 42
    _log.critical("critical error with value: %s", value)
    _log.error("runtime error with value: %s", value)
    _log.warning("warning with value: %s", value)
    _log.info("information message with value: %s", value)
    _log.debug("verbose debugging message with value: %s", value)
    ... # Construct Minerals, update orientations, etc.

```

"""
import contextlib as cl
import functools as ft
import logging

import numpy as np

from pydrex import io as _io

np.set_printoptions(
    formatter={"float_kind": np.format_float_scientific},
    linewidth=1000,
)
np.set_string_function(ft.partial(np.array2string, separator=", "), repr=False)


class ConsoleFormatter(logging.Formatter):
    """Log formatter that uses terminal color codes."""

    def colorfmt(self, code):
        return f"\033[{code}m%(levelname)s [%(asctime)s]\033[m %(name)s: %(message)s"

    def format(self, record):
        format_specs = {
            logging.CRITICAL: self.colorfmt("1;31"),
            logging.ERROR: self.colorfmt("31"),
            logging.INFO: self.colorfmt("32"),
            logging.WARNING: self.colorfmt("33"),
            logging.DEBUG: self.colorfmt("34"),
        }
        self._style._fmt = format_specs.get(record.levelno)
        return super().format(record)


# To create a new logger we use getLogger as recommended by the logging docs.
LOGGER = logging.getLogger("pydrex")
# To allow for multiple handlers at different levels, default level must be DEBUG.
LOGGER.setLevel(logging.DEBUG)
# Set up console handler.
LOGGER_CONSOLE = logging.StreamHandler()
LOGGER_CONSOLE.setFormatter(ConsoleFormatter(datefmt="%H:%M"))
LOGGER_CONSOLE.setLevel(logging.INFO)
# Turn on console logger by default.
LOGGER.addHandler(LOGGER_CONSOLE)


@cl.contextmanager
def logfile_enable(path, level=logging.DEBUG, mode="w"):
    """Enable logging to a file at `path` with given `level`."""
    logger_file = logging.FileHandler(_io.resolve_path(path), mode=mode)
    logger_file.setFormatter(
        logging.Formatter(
            "%(levelname)s [%(asctime)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger_file.setLevel(level)
    LOGGER.addHandler(logger_file)
    yield
    logger_file.close()


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
