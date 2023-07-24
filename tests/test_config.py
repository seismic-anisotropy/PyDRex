"""> PyDRex: tests for configuration file format."""
import pathlib as pl

import numpy as np

from pydrex import io as _io
from pydrex import core as _core


def test_specfile():
    """Test TOML spec file parsing."""
    config = _io.parse_config(_io.data("specs") / "spec.toml")
    assert list(config.keys()) == ["name", "input", "output", "parameters"]
    _input = config["input"]
    # TODO: Add some example mesh/path files and use them in another test.
    # TODO: Fix asserts here, implement locations_initial parser and
    # built-in analytical velocity fields.
    assert _input["mesh"] is None
    assert _input["locations_final"] is None
    assert _input["velocity_field"] == "simple_shear"  # Should be a funcref?
    assert _input["velocity_dimensions"] == 2
    assert _input["locations_initial"] == "start.scsv"  # Should be an absolute path.
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
        "olivine_fraction": 1.0,
        "enstatite_fraction": 0.0,
        "initial_olivine_fabric": _core.MineralFabric.olivine_A,
        "stress_exponent": 1.5,
        "deformation_exponent": 3.5,
        "gbm_mobility": 125,
        "gbs_threshold": 0.3,
        "nucleation_efficiency": 5.0,
        "number_of_grains": 2000,
    }
