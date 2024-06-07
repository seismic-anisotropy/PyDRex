"""> PyDRex: logger settings and boilerplate.

Python's logging module is weird and its methods don't allow us to specify
which logger to use, so just using `logging.debug` for example always uses
the "root" logger, which spams a bunch of messages from other imports/modules.
Instead, the methods in this module are thin wrappers that use custom
logging objects (`pydrex.logger.LOGGER` and `pydrex.logger.CONSOLE_LOGGER`).
The method `quiet_aliens` can be invoked to suppress most messages
from third-party modules, except critical errors and warnings from Numba.

By default, PyDRex emits INFO level messages to the console.
This can be changed globally by setting the new level with `CONSOLE_LOGGER.setLevel`:

```python
from pydrex import logger as _log
_log.info("this message will be printed to the console")

_log.CONSOLE_LOGGER.setLevel("ERROR")
_log.info("this message will NOT be printed to the console")
_log.error("this message will be printed to the console")
```

To change the console logging level for a particular local context,
use the `handler_level` context manager:

```python
_log.CONSOLE_LOGGER.setLevel("INFO")
_log.info("this message will be printed to the console")

with handler_level("ERROR"):
    _log.info("this message will NOT be printed to the console")

_log.info("this message will be printed to the console")
```

To save logs to a file, the `pydrex.io.logfile_enable` context manager is recommended.
Always use the old printf style formatting for log messages, not fstrings,
otherwise compute time will be wasted on string conversions when logging is disabled:

```python
from pydrex import io as _io
_log.quiet_aliens()  # Suppress third-party log messages except CRITICAL from Numba.
with _io.logfile_enable("my_log_file.log"):  # Overwrite existing file unless mode="a".
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
import sys

import numpy as np

# NOTE: Do NOT import any pydrex submodules here to avoid cyclical imports.

np.set_printoptions(
    formatter={"float_kind": np.format_float_scientific},
    linewidth=1000,
)
np.set_string_function(ft.partial(np.array2string, separator=", "), repr=False)


class ConsoleFormatter(logging.Formatter):
    """Log formatter that uses terminal color codes."""

    def colorfmt(self, code):
        return (
            f"\033[{code}m%(levelname)s [%(asctime)s]\033[m"
            + " \033[1m%(name)s:\033[m %(message)s"
        )

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
CONSOLE_LOGGER = logging.StreamHandler()
CONSOLE_LOGGER.setFormatter(ConsoleFormatter(datefmt="%H:%M"))
CONSOLE_LOGGER.setLevel(logging.INFO)
# Turn on console logger by default.
LOGGER.addHandler(CONSOLE_LOGGER)


def handle_exception(exec_type, exec_value, exec_traceback):
    # Ignore KeyboardInterrupt so ^C (ctrl + C) works as expected.
    if issubclass(exec_type, KeyboardInterrupt):
        sys.__excepthook__(exec_type, exec_value, exec_traceback)
        return
    # Send other exceptions to the logger.
    LOGGER.exception(
        "uncaught exception", exc_info=(exec_type, exec_value, exec_traceback)
    )


# Make our logger handle uncaught exceptions.
sys.excepthook = handle_exception


@cl.contextmanager
def handler_level(level, handler=CONSOLE_LOGGER):
    """Set logging handler level for current context.

    Args:
    - `level` (string) — logging level name e.g. "DEBUG", "ERROR", etc.
      See Python's logging module for details.
    - `handler` (optional, `logging.Handler`) — alternative handler to control instead
      of the default, `CONSOLE_LOGGER`.

    """
    default_level = handler.level
    handler.setLevel(level)
    yield
    handler.setLevel(default_level)


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


def exception(msg, *args, **kwargs):
    """Log a message with level ERROR but retain exception information.

    This function should only be called from an exception handler.

    """
    LOGGER.exception(msg, *args, **kwargs)


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
