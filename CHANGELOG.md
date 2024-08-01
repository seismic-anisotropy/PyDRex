# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added
- `pydrex.viscosity` module with effective viscosity contributions for
  different deformation mechanisms based on a selection of proposed
  constitutive equations
- Option to use Herzberg et al. 2000 peridotite solidus fit
- Decorator to serialize lexical closures using `dill`
- `pydrex.update_all` to update texture of multiphase aggregates simultaneously
- `get_regime` argument to `update_all`/`update_orientations` to allow for
  temporally variable deformation regimes
- Texture evolution for diffusion creep and yielding regimes (experimental)
- Functions to get peridotite solidus and second order tensor invariants
- Terse SCSV schema parser (only for Python >= 3.12)
- Ability to toggle verbose doctest output using `pytest -vv`

### Changed
- Call signature for steady flow 2D box visualisation function
- Symbol names for default stiffness tensors (now members of
  `minerals.StiffnessTensors` — use your own preferred stiffness tensors by
  passing a custom instance when e.g. calculating Voigt averages)
- Default parameter namespace (moved from `io.DEFAULT_PARAMS` to
  `core.DefaultParams`)

### Fixed
- Handling of enstatite in Voigt averaging
- Handling of optional keyword args in some visualisation functions


## [0.0.1] - 2024-04-24

Alpha release.
