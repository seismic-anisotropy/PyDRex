"""> Configuration and fixtures for PyDRex tests."""
import matplotlib
import pytest
from _pytest.logging import LoggingPlugin, _LiveLoggingStreamHandler
from numpy import random as rn

from pydrex import logger as _log
from pydrex import mock as _mock

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
        _log.LOGGER_PYTEST = handler
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


@pytest.fixture(scope="session")
def outdir(request):
    return request.config.getoption("--outdir")


@pytest.fixture(scope="function")
def console_handler(request):
    if request.config.option.verbose > 0:
        return request.config.pluginmanager.get_plugin(
            "pytest-console-logger"
        ).log_cli_handler
    return _log.LOGGER_CONSOLE


@pytest.fixture
def params_Fraters2021():
    return _mock.PARAMS_FRATERS2021


@pytest.fixture
def params_Kaminski2001_fig5_solid():
    return _mock.PARAMS_KAMINSKI2001_FIG5_SOLID


@pytest.fixture
def params_Kaminski2001_fig5_shortdash():
    return _mock.PARAMS_KAMINSKI2001_FIG5_SHORTDASH


@pytest.fixture
def params_Kaminski2001_fig5_longdash():
    return _mock.PARAMS_KAMINSKI2001_FIG5_LONGDASH


@pytest.fixture
def params_Kaminski2004_fig4_triangles():
    return _mock.PARAMS_KAMINSKI2004_FIG4_TRIANGLES


@pytest.fixture
def params_Kaminski2004_fig4_squares():
    return _mock.PARAMS_KAMINSKI2004_FIG4_SQUARES


@pytest.fixture
def params_Kaminski2004_fig4_circles():
    return _mock.PARAMS_KAMINSKI2004_FIG4_CIRCLES


@pytest.fixture
def params_Hedjazian2017():
    return _mock.PARAMS_HEDJAZIAN2017


@pytest.fixture(params=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def hkl(request):
    return request.param


@pytest.fixture(params=["xz", "yz", "xy"])
def ref_axes(request):
    return request.param


@pytest.fixture
def rng():
    """A seeded RNG for tests to have (more) reproducible results."""
    return rn.default_rng(seed=8816)
