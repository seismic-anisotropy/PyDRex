# PyDRex

<p align="center" style="margin:4%;">
    <img alt="PyDRex logo" src="https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/docs/assets/pydrex.png" width="25%"/>
</p>

#### Simulate crystallographic preferred orientation evolution in polycrystals

This repository contains a Python 3 reimplementation of the D-Rex model
for the evolution of crystallographic preferred orientation in polycrystals.
The code is available for use under the [GNU GPL v3](LICENSE) license.
Documentation is accessible via Python's REPL `help()` and also [online](https://seismic-anisotropy.github.io/PyDRex/).

## Install

Check `requires-python` in `pyproject.toml` for the minimum required Python
version. The software is tested on Linux, MacOS and Windows.

The package is currently not available on PyPi, so installation requires `git`.
Install the package with Python's `pip`:

    pip install git+https://github.com/adigitoleo/PyDRex#egg=pydrex

Alternatively, clone the [source code](https://github.com/seismic-anisotropy/PyDRex).
and execute `pip install "$PWD"` in the top-level folder.
To install additional dependencies required only for the test suite,
use `pip install "$PWD[test]"`.

For a complete development install, including documentation generator dependencies,
use `pip install -e "$PWD[dev]"`.

The package metadata and full list of dependencies are specified in [`pyproject.toml`](pyproject.toml).

Optional mesh generation using [gmsh](https://pypi.org/project/gmsh/) is available,
however the `gmsh` module requires the [`glu`](https://gitlab.freedesktop.org/mesa/glu) library
(that may not be installed by default on all systems).

## Test

Some tests can optionally output figures or extended diagnostics when run locally.
Check the [test suite README](tests/README.md) for details.

## Documentation

The documentation can be built offline using [pdoc](https://github.com/mitmproxy/pdoc),
with the command:

    pdoc -t docs/template -o html --math pydrex tests

which will output the html documentation into a folder called `html`.
The homepage will be `html/index.html`.
See also the provided [GitHub actions workflow](.github/workflows/docs.yml).
