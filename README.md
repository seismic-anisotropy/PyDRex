# PyDRex

<p align="center" style="margin:50px;">
    <img alt="PyDRex logo" src="https://raw.githubusercontent.com/seismic-anisotropy/PyDRex/main/docs/assets/logo256.png">
</p>

#### Simulate crystallographic preferred orientation evolution in polycrystals

PyDRex is a Python 3 reimplementation of the D-Rex model
for the evolution of crystallographic preferred orientation in polycrystals.
The code is available for use under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
Documentation is accessible via Python's REPL `help()` and also [online](https://seismic-anisotropy.github.io/PyDRex/).

## Installing

Tagged package versions can be installed from [PyPi](https://pypi.org/project/pydrex/).
The minimum required Python version is displayed in the sidebar (search for `Requires: Python`).
PyDRex is tested on Linux, MacOS and Windows,
however large simulations require substantial computational resources
usually only afforded by HPC Linux clusters.
Linux shell scripts for setting up a [Ray](https://www.ray.io/) cluster
on distributed memory PBS systems are provided in the `tools` folder.

Optional mesh generation using [gmsh](https://pypi.org/project/gmsh/) is available,
however the `gmsh` module requires the [`glu`](https://gitlab.freedesktop.org/mesa/glu) library
(which may require manual installation on some systems).

To install the latest version, execute:

    pip install pydrex

Optional dependencies can be installed using `package[dependency]` syntax, e.g.

    pip install pydrex[mesh,ray]

For an evolving, bleeding edge variant use the latest commit on `main`:

    pip install git+https://github.com/seismic-anisotropy/PyDRex#egg=pydrex

However, note that pip does not know how to uninstall dependencies of packages.
Versioned source distributions, which include tests and examples, are also
available from the [PyPI downloads page](https://pypi.org/project/pydrex/#files).
These include package metadata and a full list of dependencies
declared in the `pyproject.toml` file.

## Testing

Running tests or examples requires a local git clone or unpacked source distribution.
Install test suite dependencies with `pip install '.[test]'`.

Some tests can optionally output figures or extended diagnostics when run locally.
Check the `tests/README.md` file for details.

Further examples that demonstrate how PyDRex can be used within geodynamic
simulations are provided in subfolders of the `examples` folder.
They have their own README file as well.

## Building offline documentation

The documentation can be built offline from a git clone or unpacked source distribution.
Install documentation builder dependencies with `pip install '.[doc]'`.
Developers are also recommended to download the [just](https://github.com/casey/just) command runner.
Otherwise, build commands can be found in the provided `justfile`.

Run `just html` from the terminal to generate PyDRex's documentation
(available in the `html` directory), including the API reference.
The homepage will be `html/index.html`.
Alternatively, run `just live_docs` to build and serve the documentation on a `localhost` port.
Follow the displayed prompts to open the live documentation in a browser.
It should automatically reload after changes to the source code.

## Contributing

For a Linux or MacOS development environment, clone the [source code](https://github.com/seismic-anisotropy/PyDRex)
and execute the Bash script `tools/venv_install.sh`.
This will set up a local Python virtual environment with an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
of PyDRex that can be activated/deactivated by following the displayed prompts.

For documentation contributions, please note that
only `h2` headings will be rendered within HTML anchor elements.
Avoid using markdown headings with 3 or more leading hashes in most cases,
as these will not appear in the documentation sidebar nor be linkable.
