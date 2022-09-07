"""PyDRex: logger settings and boilerplate.

Python's logging module is weird and its methods don't allow us to specify
which logger to use, so just using `logging.debug` for example always uses
the "root" logger, which contains a bunch of noise from other imports/modules.

"""
import logging

LOGGER = logging.getLogger("pydrex")
LOGGER_CONSOLE = logging.StreamHandler()
LOGGER_CONSOLE.setFormatter(
    logging.Formatter("%(name)s [%(asctime)s] %(levelname)s: %(message)s")
)
LOGGER.addHandler(LOGGER_CONSOLE)


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
