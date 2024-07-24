"""> PyDRex: tests for configuration file format."""

import pathlib as pl

import numpy as np
from pydrex import core as _core
from pydrex import io as _io
from pydrex import velocity as _velocity


def test_steady_specfile():
    """Test TOML spec file for case of input mesh with steady velocity field."""
    config = _io.parse_config(_io.data("specs") / "steady_mesh.toml")
    assert list(config.keys()) == ["name", "input", "output", "parameters"]
    assert config["name"] == "pydrex-spec-steady-mesh"
    _input = config["input"]
    _mesh = _input["mesh"]
    assert _mesh.points.shape == (1705, 3)
    assert list(_mesh.point_data.keys()) == [
        "Pressure",
        "Time",
        "Velocity",
        "VelocityGradient",
    ]
    assert _mesh.cells[0].type == "triangle"
    assert _input["locations_final"].X == (500000.0, 500000.0, 500000.0, 500000.0)
    assert _input["locations_final"].Z == (-20000.0, -40000.0, -60000.0, -80000.0)
    assert _input["strain_final"] == 2.5
    _output = config["output"]
    assert _output["directory"].name == "out_steady"
    assert _output["raw_output"] == [
        _core.MineralPhase.olivine,
        _core.MineralPhase.enstatite,
    ]
    assert _output["diagnostics"] == [
        _core.MineralPhase.olivine,
        _core.MineralPhase.enstatite,
    ]
    assert _output["anisotropy"] == ["Voigt", "hexaxis", "moduli", "%decomp"]
    assert _output["log_level"] == "DEBUG"
    assert config["parameters"] == {
        "phase_assemblage": (_core.MineralPhase.olivine, _core.MineralPhase.enstatite),
        "phase_fractions": [0.7, 0.3],
        "initial_olivine_fabric": _core.MineralFabric.olivine_A,
        "stress_exponent": 1.5,
        "deformation_exponent": 3.5,
        "gbm_mobility": 125,
        "gbs_threshold": 0.3,
        "nucleation_efficiency": 5.0,
        "number_of_grains": 5000,
        "disl_Peierls_stress": 2,
        "disl_prefactors": (1e-16, 1e-17),
        "diff_prefactors": (1e-10, 1e-10),
        "disl_lowtemp_switch": 0.7,
        "disl_activation_energy": 460.0,
        "disl_activation_volume": 12.0,
        "diff_activation_energies": (430.0, 330.0),
        "diff_activation_volumes": (4.0, 4.0),
        "disl_coefficients": (
            4.4e8,
            -5.26e4,
            2.11e-2,
            1.74e-4,
            -41.8,
            4.21e-2,
            -1.14e-5,
        ),
    }


def test_specfile():
    """Test TOML spec file parsing."""
    config = _io.parse_config(_io.data("specs") / "spec.toml")
    assert list(config.keys()) == ["name", "input", "output", "parameters"]
    _input = config["input"]
    # TODO: Add some example mesh/path files and use them in another test.
    assert _input["mesh"] is None
    assert _input["locations_final"] is None
    assert (
        _input["velocity_gradient"][1].func
        == _velocity.simple_shear_2d("Y", "X", 5e-6)[1].func
    )
    assert (
        _input["velocity_gradient"][1].keywords
        == _velocity.simple_shear_2d("Y", "X", 5e-6)[1].keywords
    )
    assert _input["locations_initial"] == _io.read_scsv(
        _io.data("specs") / "start.scsv"
    )
    assert _input["timestep"] == 1e9
    assert _input["paths"] is None
    outdir = pl.Path(_io.data("specs") / "out").resolve()
    _output = config["output"]
    assert _output["directory"] == outdir
    np.testing.assert_array_equal(_output["raw_output"], [_core.MineralPhase.olivine])
    np.testing.assert_array_equal(_output["diagnostics"], [_core.MineralPhase.olivine])
    assert _output["anisotropy"] is True
    assert _output["paths"] is None
    assert _output["log_level"] == "DEBUG"
    assert config["parameters"] == {
        "phase_assemblage": (_core.MineralPhase.olivine,),
        "phase_fractions": (1.0,),
        "initial_olivine_fabric": _core.MineralFabric.olivine_A,
        "stress_exponent": 1.5,
        "deformation_exponent": 3.5,
        "gbm_mobility": 125,
        "gbs_threshold": 0.3,
        "nucleation_efficiency": 5.0,
        "number_of_grains": 2000,
        "disl_Peierls_stress": 2,
        "disl_prefactors": (1e-16, 1e-17),
        "diff_prefactors": (1e-10, 1e-10),
        "disl_lowtemp_switch": 0.7,
        "disl_activation_energy": 460.0,
        "disl_activation_volume": 12.0,
        "diff_activation_energies": (430.0, 330.0),
        "diff_activation_volumes": (4.0, 4.0),
        "disl_coefficients": (
            4.4e8,
            -5.26e4,
            2.11e-2,
            1.74e-4,
            -41.8,
            4.21e-2,
            -1.14e-5,
        ),
    }
