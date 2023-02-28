"""> Configuration and fixtures for PyDRex tests."""
import pathlib as pl

import matplotlib
import pytest
from _pytest.logging import LoggingPlugin, _LiveLoggingStreamHandler
from numpy import random as rn

from pydrex import logger as _log

matplotlib.use("Agg")  # Stop matplotlib from looking for $DISPLAY in env.
_log.quiet_aliens()  # Stop imported modules from spamming the logs.


# The default pytest logging plugin always creates its own handlers...
class PytestConsoleLogger(LoggingPlugin):
    """Pytest plugin that allows linking up a custom console logger."""

    name = "pytest-console-logger"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")
        capture_manager = config.pluginmanager.get_plugin("capturemanager")
        handler = _LiveLoggingStreamHandler(terminal_reporter, capture_manager)
        handler.setFormatter(_log.LOGGER_CONSOLE.formatter)
        handler.setLevel(_log.LOGGER_CONSOLE.level)
        self.log_cli_handler = handler

    # Override original, which tries to delete some silly globals that we aren't
    # using anymore, this might break the (already quite broken) -s/--capture.
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item):
        self.log_cli_handler.set_when("teardown")
        yield from self._runtest_for(item, "teardown")


# Hook up our logging plugin last,
# it relies on terminalreporter and capturemanager.
@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if config.option.verbose > 0:
        terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")
        capture_manager = config.pluginmanager.get_plugin("capturemanager")
        handler = _LiveLoggingStreamHandler(terminal_reporter, capture_manager)
        handler.setFormatter(_log.LOGGER_CONSOLE.formatter)
        handler.setLevel(_log.LOGGER_CONSOLE.level)
        config.pluginmanager.register(
            PytestConsoleLogger(config), PytestConsoleLogger.name
        )


def pytest_addoption(parser):
    parser.addoption(
        "--outdir",
        metavar="DIR",
        default=None,
        help="output directory in which to store PyDRex figures/logs",
    )


@pytest.fixture
def outdir(request):
    return request.config.getoption("--outdir")


@pytest.fixture
def params_Fraters2021():
    return {
        "stress_exponent": 3.5,
        "deformation_exponent": 1.5,
        "gbm_mobility": 125,
        "gbs_threshold": 0.3,
        "nucleation_efficiency": 5,
        "minerals": ("olivine", "enstatite"),
        "olivine_fraction": 0.7,
        "enstatite_fraction": 0.3,
    }


@pytest.fixture
def params_Kaminski2001_fig5_solid():
    return {
        "stress_exponent": 3.5,
        "deformation_exponent": 1.5,
        "gbm_mobility": 0,
        "gbs_threshold": 0,
        "nucleation_efficiency": 5,
        "minerals": ("olivine",),
        "olivine_fraction": 1,
    }


@pytest.fixture
def params_Kaminski2001_fig5_shortdash():
    return {
        "stress_exponent": 3.5,
        "deformation_exponent": 1.5,
        "gbm_mobility": 50,
        "gbs_threshold": 0,
        "nucleation_efficiency": 5,
        "minerals": ("olivine",),
        "olivine_fraction": 1,
    }


@pytest.fixture
def params_Kaminski2001_fig5_longdash():
    return {
        "stress_exponent": 3.5,
        "deformation_exponent": 1.5,
        "gbm_mobility": 200,
        "gbs_threshold": 0,
        "nucleation_efficiency": 5,
        "minerals": ("olivine",),
        "olivine_fraction": 1,
    }


@pytest.fixture
def params_Kaminski2004_fig4_triangles():
    return {
        "stress_exponent": 3.5,
        "deformation_exponent": 1.5,
        "gbm_mobility": 125,
        "gbs_threshold": 0.4,
        "nucleation_efficiency": 5,
        "minerals": ("olivine",),
        "olivine_fraction": 1.0,
    }


@pytest.fixture
def params_Kaminski2004_fig4_squares():
    return {
        "stress_exponent": 3.5,
        "deformation_exponent": 1.5,
        "gbm_mobility": 125,
        "gbs_threshold": 0.2,
        "nucleation_efficiency": 5,
        "minerals": ("olivine",),
        "olivine_fraction": 1.0,
    }


@pytest.fixture
def params_Kaminski2004_fig4_circles():
    return {
        "stress_exponent": 3.5,
        "deformation_exponent": 1.5,
        "gbm_mobility": 125,
        "gbs_threshold": 0,
        "nucleation_efficiency": 5,
        "minerals": ("olivine",),
        "olivine_fraction": 1.0,
    }


@pytest.fixture
def params_Hedjazian2017():
    return {
        "stress_exponent": 3.5,
        "deformation_exponent": 1.5,
        "gbm_mobility": 10,
        "gbs_threshold": 0.2,  # TODO: Check again, Chris used 0.3?
        "nucleation_efficiency": 5,
        "olivine_fraction": 0.7,
        "enstatite_fraction": 0.3,
    }


@pytest.fixture
def vtkfiles_2d_corner_flow():
    datadir = pl.Path(__file__).parent / ".." / "data" / "vtu"
    return (datadir / "corner2d_2cmyr_5e5x1e5.vtu",)


@pytest.fixture
def rng():
    return rn.default_rng()
