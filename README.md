# PyDRex

#### Simulate crystallographic preferred orientation evolution in polycrystals

This repository contains a Python 3 reimplementation of the D-Rex model
for the evolution of crystallographic preferred orientation in polycrystals.
The code is available for use under the [GNU GPL v3](LICENSE) license.

## Install

The package is currently not available on PyPi,
and must be installed by cloning the [source code](https://github.com/Patol75/PyDRex).
and using `pip install .` (with the dot) in the top-level folder.
To install additional dependencies required only for the test suite,
use `pip install .[test]`.

For a complete development install, including documentation generator dependencies,
use `pip install -e .[test,dev,doc]`.

The package metadata and full list of dependencies are specified in [`setup.cfg`](setup.cfg).

## Test

Some tests can optionally output figures or extended diagnostics when run locally.
Check the [test suite README](tests/README.md) for details.

## Documentation

The documentation can be build using [pdoc](https://github.com/mitmproxy/pdoc),
using the command:

    pdoc -t docs/template -o html --math src/pydrex/ tests/*.py

which will output the html documentation into a folder called `html`.
The homepage will be `html/pydrex.html`.
