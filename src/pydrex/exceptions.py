"""> PyDRex: Custom exceptions (subclasses of `pydrex.Error`)."""

# <https://docs.python.org/3.11/tutorial/errors.html#user-defined-exceptions>


class Error(Exception):
    """Base class for exceptions in PyDRex."""


class ConfigError(Error):
    """Exception raised for errors in the input configuration.

    Attributes:
        message — explanation of the error

    """

    def __init__(self, message):  # pylint: disable=super-init-not-called
        self.message = message


class MeshError(Error):
    """Exception raised for errors in the input mesh.

    Attributes:
        message — explanation of the error

    """

    def __init__(self, message):  # pylint: disable=super-init-not-called
        self.message = message


class IterationError(Error):
    """Exception raised for errors in numerical iteration schemes.

    Attributes:
        message — explanation of the error

    """

    def __init__(self, message):  # pylint: disable=super-init-not-called
        # TODO: Add data attribute? Timestep?
        self.message = message


class SCSVError(Error):
    """Exception raised for errors in SCSV file I/O.

    Attributes:
    - message — explanation of the error

    """

    def __init__(self, message):  # pylint: disable=super-init-not-called
        self.message = message
