"""> PyDRex: Run doctests for all submodules."""

import doctest
import importlib
import os
import pkgutil

import numpy as np
import pytest

import pydrex
from pydrex import logger as _log
from pydrex.exceptions import Error


def _get_submodule_list():
    # Reset NumPy print options because doctests are just string matches, and typing out
    # so many significant digits in doctests is annoying.
    np.set_printoptions()
    np.set_string_function(None)
    modules = ["pydrex." + m.name for m in pkgutil.iter_modules(pydrex.__path__)]
    modules.remove("pydrex.distributed")
    try:
        from pydrex import mesh
    except ImportError:
        modules.remove("pydrex.mesh")
    return modules


@pytest.mark.parametrize("module", _get_submodule_list())
def test_doctests(module):
    """Run doctests for all submodules."""
    _log.info("running doctests for %s...", module)
    try:
        doctest.testmod(importlib.import_module(module), raise_on_error=True)
    except doctest.DocTestFailure as e:
        if e.test.lineno is None:
            lineno = ""
        else:
            lineno = f":{e.test.lineno + 1 + e.example.lineno}"
        raise Error(
            f"{e.test.name} ({module}{lineno}) failed with:"
            + os.linesep
            + os.linesep
            + e.got
        ) from None
    except doctest.UnexpectedException as e:
        if e.test.lineno is None:
            lineno = ""
        else:
            lineno = f":{e.test.lineno + 1 + e.example.lineno}"
        err_type, err, _ = e.exc_info
        raise Error(
            f"{err_type.__qualname__} encountered in {e.test.name} ({module}{lineno})"
        ) from err
