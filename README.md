# PyDRex

<p align="center" style="margin:4%;">
    <img alt="PyDRex logo" src="https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/docs/assets/pydrex.png" width="25%"/>
</p>

#### Simulate crystallographic preferred orientation evolution in polycrystals

This repository contains a Python 3 reimplementation of the D-Rex model
for the evolution of crystallographic preferred orientation in polycrystals.
The code is available for use under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
Documentation is accessible via Python's REPL `help()` and also [online](https://seismic-anisotropy.github.io/PyDRex/).

## Installing

Check `requires-python` in `pyproject.toml` for the minimum required Python
version. The software is tested on Linux, MacOS and Windows, however large
simulations require substantial computational resources usually afforded by HPC
Linux clusters. Linux shell scripts for setting up a [Ray](https://www.ray.io/) cluster
on distributed memory PBS systems are provided in the `tools` folder.

Optional mesh generation using [gmsh](https://pypi.org/project/gmsh/) is available,
however the `gmsh` module requires the [`glu`](https://gitlab.freedesktop.org/mesa/glu) library
(which may require manual installation on some systems).

Tagged package versions can be installed from [PyPi](https://pypi.org/project/pydrex/).
To install the latest version, execute:

    pip install pydrex

Optional dependencies can be installed using `package[dependency]` syntax, e.g.

    pip install pydrex[mesh,ray]

For an evolving, bleeding edge variant use the latest commit on `main`:

    pip install git+https://github.com/seismic-anisotropy/PyDRex#egg=pydrex

However, note that pip does not know how to uninstall dependencies of pacakges.
Versioned source distributions, which include tests and examples, are also
available from the [PyPI downloads page](https://pypi.org/project/pydrex/#files).
These include package metadata and a full list of dependencies
declared in the `pyproject.toml` file.

## Testing

Running tests or examples requires a local git clone or unpacked source distribution.

Some tests can optionally output figures or extended diagnostics when run locally.
Testing locally requires a local git clone or source distribution of PyDRex.
Check the `tests/README.md` file for details.

Further examples that demonstrate how PyDRex can be used within geodynamic
simulations are provided in the `examples` folder.
They have their own README file as well.

## Building offline documentation

The documentation can be built offline using [pdoc](https://github.com/mitmproxy/pdoc),
with the command:

    pdoc -t docs/template -o html --math pydrex tests

which will output the html documentation into a folder called `html`.
The homepage will be `html/index.html`.
Note that some submodules depend on optional dependencies,
and should otherwise be excluded:

    pdoc -t docs/template -o html --math pydrex !pydrex.mesh !pydrex.distributed tests

See also the provided [GitHub actions workflow](https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/.github/workflows/docs.yml).

## Contributing

For a development environment, clone the [source code](https://github.com/seismic-anisotropy/PyDRex)
and execute the Bash script `tools/venv_install.sh`.
This will set up a local Python virtual environment with an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
of PyDRex that can be activated/deactivated by following the displayed prompts.
