#!/bin/sh
set -eu
readonly SCRIPTNAME="${0##*/}"
readonly MESH_SMALL='shearbox_1m.msh'
#readonly MESH_LARGE='shearbox_100km.msh'
readonly OPTS_SMALL='shearbox_single_particle.flml'
#readonly OPTS_LARGE='shearbox_500_particles.flml'

warn() { >&2 printf '%s\n' "$SCRIPTNAME: $1" ; }

is_command() { # Check if command exists, for flow control (no stdout messages)
    if 1>/dev/null 2>&1 command -v "$1"; then
        return 0
    else
        warn "command '$1' not found"; return 1
    fi
}

has_python_module() {
    is_command python3 || return 1
    1>/dev/null 2>&1 python3 -m "$1" \
        || { warn "failed to import '$1' from python3" ; return 1 ; }
}

# Check environment for necessary executables and modules.
is_command pydrex || exit 1
is_command fluidity || exit 1
has_python_module "fluidity.state_types" || exit 1
has_python_module "gmsh" || exit 1
is_command mpirun || exit 1
is_command flredecomp || exit 1

# Create output directory, fail if the last simulation wasn't cleaned up.
mkdir out

# Copy or generate mesh files.
if [ -f "src/$MESH_SMALL" ] && [ -f "src/$MESH_LARGE" ] ; then
    cp "src/$MESH_SMALL" out
#    cp "src/$MESH_LARGE" out
else
    >out/gmsh.log python3 ./src/shearbox_gmsh.py -o out
fi

# Copy model options files.
cp "src/$OPTS_SMALL" out
#cp "src/$OPTS_LARGE" out

# Copy auxiliary model files.
cp src/shearbox_pydrex.py out
cp src/shearbox_single.ini out
#cp src/shearbox_500.ini out

# Run the simulations.
( cd out
    fluidity -v2 -l "$OPTS_SMALL"
    #fluidity -v2 -l "$OPTS_LARGE"
    pydrex --ncpus 8 --config shearbox_single.ini "${OPTS_SMALL%.flml}_1.vtu"
    #pydrex --ncpus 8 --config shearbox_500.ini "${OPTS_LARGE%.flml}_1.vtu"
    #>>pydrex.log mpirun -np 8 pydrex --charm "${OPTS_SMALL%.flml}_1.vtu"
    #>>pydrex.log mpirun -np 8 pydrex --charm "${OPTS_LARGE%.flml}_1.vtu"
)
