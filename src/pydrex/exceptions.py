"""PyDRex: Custom exceptions."""
# <https://docs.python.org/3.8/tutorial/errors.html#user-defined-exceptions>


class Error(Exception):
    """Base class for exceptions in PyDRex."""


class MeshError(Error):
    """Exception raised for errors in the input mesh.

    Attributes:
        message — explanation of the error

    """

    def __init__(self, message):  # pylint: disable=super-init-not-called
        self.message = message
