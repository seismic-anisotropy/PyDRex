#!/bin/env sh
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

upgrade() {
    . .venv-"${PWD##*/}"/bin/activate
    [ -f requirements.txt ] && mv -i requirements.txt requirements.bak
    pip install --upgrade pip pip-tools && pip-compile --resolver=backtracking && pip-sync
    pip install -e "$PWD"\[dev\]
}

if [ $# -eq 0 ]; then  # first install
    [ -d .venv-"${PWD##*/}" ] && exit 0  # venv already exists, no-op
    "$(2>/dev/null pyenv prefix||printf '/usr')"/bin/python -m venv .venv-"${PWD##*/}"
    upgrade
fi

while getopts "hu" OPT; do
    case $OPT in
        h ) usage && helpf; exit 0;;
        u ) upgrade;;
        * ) usage; exit 1;;
    esac
done