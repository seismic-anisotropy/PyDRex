import pytest

import numpy as np

import pydrex.minerals as _minerals
import pydrex.deformation_mechanism as _defmech


@pytest.fixture
def enstatite_disl_random_5000():
    return _minerals.Mineral(
        _minerals.MineralPhase.enstatite,
        _minerals.EnstatiteFabric.A,
        _defmech.Regime.dislocation,
        5000,
    )


@pytest.fixture
def olivine_A_disl_random_5000():
    return _minerals.Mineral(
        _minerals.MineralPhase.olivine,
        _minerals.OlivineFabric.A,
        _defmech.Regime.dislocation,
        5000,
    )


@pytest.fixture
def enstatite_disl_random_2000():
    return _minerals.Mineral(
        _minerals.MineralPhase.enstatite,
        _minerals.EnstatiteFabric.A,
        _defmech.Regime.dislocation,
        2000,
    )


@pytest.fixture
def olivine_A_disl_random_2000():
    return _minerals.Mineral(
        _minerals.MineralPhase.olivine,
        _minerals.OlivineFabric.A,
        _defmech.Regime.dislocation,
        2000,
    )


@pytest.fixture
def enstatite_disl_random_500():
    return _minerals.Mineral(
        _minerals.MineralPhase.enstatite,
        _minerals.EnstatiteFabric.A,
        _defmech.Regime.dislocation,
        500,
    )


@pytest.fixture
def olivine_A_disl_random_500():
    return _minerals.Mineral(
        _minerals.MineralPhase.olivine,
        _minerals.OlivineFabric.A,
        _defmech.Regime.dislocation,
        500,
    )


@pytest.fixture
def olivine_disl_random_500():
    return [
        _minerals.Mineral(
            _minerals.MineralPhase.olivine,
            fabric,
            _defmech.Regime.dislocation,
            500,
        )
        for fabric in _minerals.OlivineFabric
    ]


@pytest.fixture
def enstatite_disl_random_500():
    return [
        _minerals.Mineral(
            _minerals.MineralPhase.enstatite,
            fabric,
            _defmech.Regime.dislocation,
            500,
        )
        for fabric in _minerals.EnstatiteFabric
    ]


@pytest.fixture
def mock_params_Fraters2021():
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
def mock_params_Kaminski2004_fig4_triangles():
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
def mock_params_Kaminski2004_fig4_squares():
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
def mock_params_Kaminski2004_fig4_circles():
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
def mock_params_Navid2017():
    return {
        "stress_exponent": 3.5,
        "dislocation_exponent": 1.5,
        "gbm_mobility": 10,
        "gbs_threshold": 0.2,  # TODO: Check again, Chris used 0.3?
        "nucleation_efficiency": 5,
        "olivine_fraction": 0.7,
        "enstatite_fraction": 0.3,
    }
