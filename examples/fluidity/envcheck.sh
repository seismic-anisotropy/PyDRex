#!/bin/sh
readonly SCRIPTNAME="${0##*/}"
usage() {
    printf 'Usage: %s [-h]\n' "$SCRIPTNAME"
    printf '       %s [-f][-m]\n' "$SCRIPTNAME"
    echo
    printf 'Checks the current enviromnent for fluidity and flredecomp executables. '
    printf 'Also verifies the presence of a python3 runtime with access to '
    printf 'the gmsh and fluidity state modules.\n'
}
helpf() {
    echo
    echo 'Options:'
    echo '-f                check for fluidity, flredecomp and fluidity state'
    echo '-m                check for the gmsh module'
}
warn() {
    >&2 printf '%s\n' "${SCRIPTNAME}: $1"
}
tell() {
    printf '%s\n' "${SCRIPTNAME}: $1"
}
is_command() {
    cmdpath="$(command -v $1)"
    if [ -n "$cmdpath" ]; then
        tell "command '$1' is '$cmdpath'"; return 0
    else
        warn "command '$1' not found"; return 1
    fi
}
has_python3_module() {
    python3_version="$(python3 -V)"
    python3 -c "import importlib; importlib.import_module('$1')" 1>&2 2>/dev/null \
        && { tell "successfully imported '$1' from $python3_version"; return 0; } \
        || { warn "failed to import '$1' from $python3_version"; return 1 ;}
}

[ $# -eq 0 ] && usage && exit 1
while getopts "fmh" OPT ; do
    case "$OPT" in
        f )
            is_command fluidity || exit 1
            is_command flredecomp || exit 1
            is_command python3 || exit 1
            has_python3_module "fluidity.state_types" || exit 1
            has_python3_module "tomllib" || exit 1
            has_python3_module "pydrex" || exit 1
            ;;
        m )
            is_command python3 || exit 1
            has_python3_module "gmsh" || exit 1
            ;;
        h ) usage && helpf ; exit 0 ;;
    esac
done
