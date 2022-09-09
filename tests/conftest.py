"""Configuration and fixtures for PyDRex tests."""
import pathlib as pl

import matplotlib
import pytest

from pydrex import logger as _log

matplotlib.use("Agg")  # Stop matplotlib from looking for $DISPLAY in env.
_log.quiet_aliens()  # Stop imported modules from spamming the logs.


def pytest_configure(config):
    # Make sure --log-cli-level can be used by default.
    config.option.log_cli = True
    # Make sure pytest never prints more than WARNING messages by default.
    config.option.log_level = "WARNING"
    # Inherit format for "live logs" (--log-cli-level) from our logger.
    config.option.log_cli_format = _log.LOGGER_CONSOLE.formatter._fmt


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
        "dislocation_exponent": 1.5,
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
        "dislocation_exponent": 1.5,
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
        "dislocation_exponent": 1.5,
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
        "dislocation_exponent": 1.5,
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
        "dislocation_exponent": 1.5,
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
        "dislocation_exponent": 1.5,
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
        "dislocation_exponent": 1.5,
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
        "dislocation_exponent": 1.5,
        "gbm_mobility": 10,
        "gbs_threshold": 0.2,  # TODO: Check again, Chris used 0.3?
        "nucleation_efficiency": 5,
        "olivine_fraction": 0.7,
        "enstatite_fraction": 0.3,
    }


@pytest.fixture
def vtkfiles_2d_corner_flow():
    datadir = pl.Path(__file__).parent / ".." / "data" / "vtu"
    return (
        datadir / "2d_corner_flow_2cmyr.vtu",
        datadir / "2d_corner_flow_4cmyr.vtu",
        datadir / "2d_corner_flow_6cmyr.vtu",
        datadir / "2d_corner_flow_8cmyr.vtu",
    )
