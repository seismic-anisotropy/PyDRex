"""> PyDRex: Run doctests for all submodules."""

import doctest
import importlib
import os
import pkgutil

import numpy as np
import pydrex
import pytest
from pydrex import logger as _log
from pydrex.exceptions import Error


def _get_submodule_list():
    # Reset NumPy print options because doctests are just string matches, and typing out
    # so many significant digits in doctests is annoying.
    np.set_printoptions()
    np.set_string_function(None)
    modules = ["pydrex." + m.name for m in pkgutil.iter_modules(pydrex.__path__)]
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            modules.remove(module)
    return modules


@pytest.mark.parametrize("module", _get_submodule_list())
def test_doctests(module, capsys, verbose):
    """Run doctests for all submodules."""
    with capsys.disabled():  # Pytest output capturing messes with doctests.
        _log.info("running doctests for %s...", module)
        try:
            n_fails, n_tests = doctest.testmod(
                importlib.import_module(module),
                raise_on_error=True,
                verbose=verbose > 1,  # Run pytest with -vv to show doctest details.
            )
            if n_fails > 0:
                raise AssertionError(
                    f"there were {n_fails} doctest failures from {module}"
                )
        except doctest.DocTestFailure as e:
            if e.test.lineno is None:
                lineno = ""
            else:
                lineno = f":{e.test.lineno + 1 + e.example.lineno}"
            raise AssertionError(
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
            if err_type == NameError:  # Raised on missing optional functions.
                # Issue warning but let the test suite pass.
                _log.warning(
                    "skipping doctest of missing optional symbol in %s", module
                )
            elif err_type == np.core._exceptions._ArrayMemoryError:
                # Faiures to allocate should not be fatal to the doctest test suite.
                _log.warning(
                    "skipping doctests for module %s due to insufficient memory", module
                )
            else:
                raise Error(
                    f"{err_type.__qualname__} encountered in {e.test.name} ({module}{lineno})"
                ) from err
