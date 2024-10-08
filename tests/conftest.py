"""> Configuration and fixtures for PyDRex tests."""

import argparse

import matplotlib
import numpy as np
import pytest
from _pytest.logging import LoggingPlugin, _LiveLoggingStreamHandler
from pydrex import io as _io
from pydrex import logger as _log
from pydrex import mock as _mock
from pydrex import utils as _utils
from scipy.spatial.transform import Rotation

from tests import test_vortex_2d as _test_vortex_2d

_log.quiet_aliens()  # Stop imported modules from spamming the logs.
_, HAS_RAY = _utils.import_proc_pool()
if HAS_RAY:
    import ray


# Set up custom pytest CLI arguments.
def pytest_addoption(parser):
    parser.addoption(
        "--outdir",
        metavar="DIR",
        default=None,
        help="output directory in which to store PyDRex figures/logs",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (HPC cluster recommended, large memory requirement)",
    )
    parser.addoption(
        "--runbig",
        action="store_true",
        default=False,
        help="run tests which are fast enough for home PCs but require 16GB RAM",
    )
    parser.addoption(
        "--ncpus",
        default=_utils.default_ncpus(),
        type=int,
        help="number of CPUs to use for tests that support multiprocessing",
    )
    parser.addoption(
        "--fontsize",
        default=None,
        type=int,
        help="set explicit font size for output figures",
    )
    parser.addoption(
        "--markersize",
        default=None,
        type=int,
        help="set explicit marker size for output figures",
    )
    parser.addoption(
        "--linewidth",
        default=None,
        type=int,
        help="set explicit line width for output figures",
    )


class PyDRexLiveLogger(LoggingPlugin):
    """Pytest plugin for custom handling of live logging."""

    name = "pydrex-live-logger"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")
        capture_manager = config.pluginmanager.get_plugin("capturemanager")
        handler = _LiveLoggingStreamHandler(terminal_reporter, capture_manager)
        handler.setFormatter(_log.CONSOLE_LOGGER.formatter)
        handler.setLevel(_log.CONSOLE_LOGGER.level)
        self.log_cli_handler = handler

    # Override original, which tries to use some silly globals for log capture.
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item):
        self.log_cli_handler.set_when("teardown")
        yield from self._runtest_for(item, "teardown")


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "big: mark test as requiring 16GB RAM")

    # Set custom Matplotlib parameters.
    # Alternatively inject a call to `matplotlib.style.use` before starting pytest.
    if config.option.fontsize is not None:
        matplotlib.rcParams["font.size"] = config.option.fontsize
    if config.option.markersize is not None:
        matplotlib.rcParams["lines.markersize"] = config.option.markersize
    if config.option.linewidth is not None:
        matplotlib.rcParams["lines.linewidth"] = config.option.linewidth

    # Hook up our logging plugin last,
    # it relies on terminalreporter and capturemanager.
    # To subclass the logging plugin we also had to break --capture except for -s
    # (--capture=no), so bail if the user tries to set --capture=method.
    if config.option.log_cli_level is not None:
        _log.CONSOLE_LOGGER.setLevel(config.option.log_cli_level)
    if (
        config.option.log_cli_format is not None
        or config.option.log_cli_date_format is not None
    ):
        raise argparse.ArgumentError(
            None,
            message="pydrex test suite does not support changing the CLI logging format",
        ) from None
    if config.option.log_file is not None:
        raise argparse.ArgumentError(
            None,
            message=(
                "pydrex test suite does not support pytest --log-file option, "
                + "use --outdir instead"
            ),
        ) from None
    if config.option.capture != "fd":
        raise argparse.ArgumentError(
            None,
            message=(
                "pydrex test suite only supports pytest 'fd' capture method, "
                + "use --capture=fd"
            ),
        ) from None
    if config.option.verbose > 0 or config.getini("log_cli"):
        config.pluginmanager.register(PyDRexLiveLogger(config), PyDRexLiveLogger.name)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # Don't skip slow tests.
        _log.info("running slow tests with %d CPUs", config.getoption("--ncpus"))
    else:
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if config.getoption("--runbig"):
        pass  # Don't skip big tests.
    else:
        skip_big = pytest.mark.skip(reason="need --runbig option to run")
        for item in items:
            if "big" in item.keywords:
                item.add_marker(skip_big)


@pytest.fixture(scope="session")
def verbose(request):
    if request.config.option.verbose == 0 and request.config.getini("log_cli"):
        return 1
    return request.config.option.verbose


@pytest.fixture(scope="session")
def outdir(request):
    _outdir = request.config.getoption("--outdir")
    yield _outdir
    #  Create combined ensemble figure for 2D cell tests after they have all finished.
    _test_vortex_2d.TestCellOlivineA._make_ensemble_figure(_outdir)


@pytest.fixture(scope="session")
def ncpus(request):
    return max(1, request.config.getoption("--ncpus"))


@pytest.fixture(scope="session")
def ray_session():
    if HAS_RAY:
        # NOTE: Expects a running Ray cluster with a number of CPUS matching --ncpus.
        if not ray.is_initialized():
            ray.init(address="auto")
            _log.info("using Ray cluster with %s", ray.cluster_resources())
        yield
        if ray.is_initialized():
            ray.shutdown()
    yield


@pytest.fixture
def console_handler(request):
    """Return the live logging handler for pydrex.

    If `-v`/`--verbose` is passed to the pytest command,
    this returns the handler of the 'pydrex-live-logger' pytest plugin.
    Otherwise, returns the default pydrex CLI logging handler.

    """
    if request.config.option.verbose > 0 or request.config.getini("log_cli"):
        return request.config.pluginmanager.get_plugin(
            "pydrex-live-logger"
        ).log_cli_handler
    return _log.CONSOLE_LOGGER


@pytest.fixture
def mock():
    return _mock


@pytest.fixture(scope="session")
def orientations_init_y():
    rng = np.random.default_rng(seed=8816)
    return [
        lambda n_grains: None,  # For random orientations.
        lambda n_grains: Rotation.from_euler(  # A girdle around Y.
            "y", [[x * np.pi * 2] for x in rng.random(n_grains)]
        ).as_matrix(),
        lambda n_grains: Rotation.from_euler(  # Clustered orientations.
            "y", [[x * np.pi / 8] for x in rng.random(n_grains)]
        ).as_matrix(),
    ]


@pytest.fixture(scope="session", params=[100, 500, 1000, 5000, 10000])
def n_grains(request):
    return request.param


@pytest.fixture(scope="session", params=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def hkl(request):
    return request.param


@pytest.fixture(scope="session", params=["xz", "yz", "xy"])
def ref_axes(request):
    return request.param


@pytest.fixture(scope="session")
def seeds():
    """1000 unique seeds for ensemble runs that need an RNG seed."""
    return _io.read_scsv(_io.data("rng") / "seeds.scsv").seeds


@pytest.fixture(scope="session")
def seed():
    """Default seed for test RNG."""
    return 8816


@pytest.fixture(scope="session")
def seeds_nearX45():
    """41 seeds which have the initial hexagonal symmetry axis near 45° from X."""
    return _io.read_scsv(_io.data("rng") / "hexaxis_nearX45_seeds.scsv").seeds
