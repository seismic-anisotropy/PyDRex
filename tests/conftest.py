import pytest
from pydrex import deformation_mechanism as _defmech
from pydrex import minerals as _minerals


def pytest_addoption(parser):
    parser.addoption(
        "--outdir",
        metavar="DIR",
        default=[None],  # NOTE: `outdir` will just be None, not a list.
        help="output directory in which to store PyDRex figures/logs",
    )


def pytest_generate_tests(metafunc):
    if "outdir" in metafunc.fixturenames:
        metafunc.parametrize("outdir", metafunc.config.getoption("outdir"))


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
        "minerals": ("olivine"),
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
        "minerals": ("olivine"),
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
        "minerals": ("olivine"),
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
        "minerals": ("olivine"),
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
        "minerals": ("olivine"),
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
        "minerals": ("olivine"),
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
