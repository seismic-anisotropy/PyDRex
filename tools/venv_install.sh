#!/bin/env bash
set -eu
readonly SCRIPTNAME="${0##*/}"
usage() {
    printf 'Usage: %s [-h]\n' "$SCRIPTNAME"
    printf '       %s [-u]\n' "$SCRIPTNAME"
}
helpf() {
    echo 'Options:'
    echo '-u            Update pydrex and dependencies in the venv'
    echo '              and compile new requirements.txt'
    echo
    echo 'Install local pydrex tree in a Python venv (virtual environment),'
    echo 'in editable mode with development dependencies.'
    echo 'This script should be executed in a context which has access to a'
    echo 'Python binary of the appropriate version (check requires-python in'
    echo 'pyproject.toml). If the necessary version is not available on your'
    echo 'system, you can install it using pyenv'
    echo '(https://github.com/pyenv/pyenv) and this script will prefer the'
    echo 'local pyenv binary over the system one.'
    echo
    echo 'Backups of the last requirements.txt are stored in requirements.bak'
}
warn() { >&2 printf '%s\n' "$SCRIPTNAME: $1"; }

upgrade() {
    . .venv-"${PWD##*/}"/bin/activate
    [ -f requirements.txt ] && mv -i requirements.txt requirements.bak
    pip install --upgrade pip pip-tools && pip-compile --resolver=backtracking && pip-sync
    pip install -e "${PWD}[dev,lsp,doc,test]"
}

if [ $# -eq 0 ]; then  # first install
    [ -d .venv-"${PWD##*/}" ] && exit 0  # venv already exists, no-op
    if [ -z "${PYTHON_BINARY:-}" ]; then
        PYTHON_BINARY="$(2>/dev/null pyenv prefix||printf '/usr')"/bin/python
    fi
    [ "$($PYTHON_BINARY --version|cut -d' ' -f2|cut -d'.' -f1)" -eq 3 ] || {
        warn "Python 3 is required"; exit 1; }
    [ "$($PYTHON_BINARY --version|cut -d' ' -f2|cut -d'.' -f2)" -gt 9 ] || {
        warn "Python 3.10+ is required"; exit 1; }
    $PYTHON_BINARY -m venv .venv-"${PWD##*/}"
    upgrade
    2>/dev/null 1>&2 command -v pre-commit || {
        warn "Skipping installation of pre-commit hooks, which requires the pre-commit command.";
        warn "To install them later, run `pre-commit install` in the local PyDRex clone."
    }
    2>/dev/null 1>&2 command -v pre-commit && pre-commit install
    echo ".python-version" > .git/info/exclude
    {
        echo "*egg-info*";
        echo "*__pycache__/";
        echo ".mypy_cache/";
        echo "/html/"
        echo "/dist/"
        echo "*.mod"
        echo "/.venv-${PWD##*/}";
        echo "/requirements.txt";
        echo "/requirements.bak";
    } >> .git/info/exclude
    echo "$SCRIPTNAME: PyDRex development version installed in .venv-${PWD##*/}"
    echo "$SCRIPTNAME: On Linux, execute 'source .venv-${PWD##*/}/bin/activate' to activate the environment."
    echo "$SCRIPTNAME: Use the command 'deactivate' to subsequently deactivate the environment."
fi

while getopts "hu" OPT; do
    case $OPT in
        h ) usage && helpf; exit 0;;
        u ) upgrade;;
        * ) usage; exit 1;;
    esac
done
