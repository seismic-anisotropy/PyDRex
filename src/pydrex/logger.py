"""> PyDRex: logger settings and boilerplate.

.. note:: Always use the old printf style formatting for log messages, not fstrings,
    otherwise compute time may be wasted on string conversions when logging is disabled:

The methods in this module are thin `logging` wrappers that use custom logger objects
(`pydrex.logger.LOGGER` and its `StreamHandler`, `pydrex.logger.CONSOLE_LOGGER`).
For packages that depend on PyDRex, the `pydrex.logger.LOGGER` should be accessed by:

>>> import logging
>>> import pydrex
>>> pydrex_logger = logging.getLogger("pydrex")

Console logs are written at `INFO` level to `sys.stderr` by default:

>>> # ELLIPSIS is <stderr> except in test session.
>>> pydrex_logger.handlers  # doctest: +ELLIPSIS
[<StreamHandler ... (INFO)>]

The following examples use `sys.stdout` instead (`...` represents a timestamp).

>>> import sys
>>> cli_handler = pydrex_logger.handlers[0]
>>> _ = cli_handler.setStream(sys.stdout)  # Doctests don't check stderr.
>>> cli_handler.formatter.color_enabled = False  # Disable colors in output.
>>> pydrex_logger.info("info message")  # doctest: +ELLIPSIS
INFO [...] pydrex: info message
>>> cli_handler.setLevel(logging.ERROR)
>>> pydrex_logger.info("info message")
>>> pydrex_logger.error("error message")  # doctest: +ELLIPSIS
ERROR [...] pydrex: error message

This change persists across successive requests for logger object access:

>>> pydrex_logger = logging.getLogger("pydrex")
>>> cli_handler.level == logging.INFO
False
>>> cli_handler.level == logging.ERROR
True

### Log files and logging contexts

The logging level can also be adjusted using context managers,
both for console output and (optionally) saving logs to a file:

>>> cli_handler.setLevel(logging.INFO)
>>> cli_handler.level == logging.DEBUG
False
>>> with pydrex.io.log_cli_level(logging.DEBUG, cli_handler):
...     cli_handler.level == logging.DEBUG
True
>>> cli_handler.level == logging.DEBUG
False

Usually, a filename should be given to the `pydrex.io.logfile_enable` context manager.
In this example, (open) temporary files and streams are used for demonstration.

>>> import tempfile
>>> import io
>>> kwargs = { "delete": False } if sys.platform == "win32" else {}
>>> tmp = tempfile.NamedTemporaryFile(**kwargs)
>>> pydrex_logger.debug("debug message")
>>> with pydrex.io.logfile_enable(io.TextIOWrapper(tmp.file)):  # doctest: +ELLIPSIS
...     pydrex_logger.debug("debug message in %s", tmp.file.name)
...     with open(tmp.file.name) as f:
...         print(f.readline())
DEBUG [...] pydrex: debug message ...

### Information for PyDRex developers

All PyDRex modules that require logging should use `import pydrex.logger`,
which automatically initialises and registers the PyDRex logger objects if necessary.

The method `quiet_aliens` can be invoked to suppress logging messages from dependencies.

"""

import functools as ft
import logging
import sys

import numpy as np

# NOTE: Do NOT import any pydrex submodules here to avoid cyclical imports.

np.set_printoptions(
    formatter={
        "float_kind": np.format_float_scientific,
        "object": ft.partial(np.array2string, separator=", "),
    },
    linewidth=1000,
)


class ConsoleFormatter(logging.Formatter):
    """Log formatter that uses terminal color codes."""

    def colorfmt(self, code):
        # Color enabled by default, disabled by setting `.color_enabled` = False.
        # Adding this as a constructor arg breaks things...
        if hasattr(self, "color_enabled") and not self.color_enabled:
            return "%(levelname)s [%(asctime)s] %(name)s: %(message)s"
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


def quiet_aliens(root_level=logging.WARNING, level=logging.CRITICAL):
    """Restrict alien loggers 👽.

    .. note:: Primarily intended for internal use (test suite/development).

    - `root_level` sets the level for the "root" logger
    - `level` sets the level for everything else (except "pydrex")

    """
    logging.getLogger().setLevel(root_level)
    for name in logging.Logger.manager.loggerDict.keys():
        if name != "pydrex":
            logging.getLogger(name).setLevel(level)
