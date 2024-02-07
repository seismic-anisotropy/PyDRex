#!/bin/bash
set -eu
readonly SCRIPTNAME="${0##*/}"
usage() {
    printf 'Usage: %s [-h]\n' "$SCRIPTNAME"
    printf '       %s ' "$SCRIPTNAME"
    echo '[-o outdir][-n n_runs][-s size3]'
    echo
    echo 'Compare performance of simple shear A-type olivine runs using'
    echo ' ./drex_forward_simpleshear.f90'
    echo 'and a bespoke PyDRex example (without "pytest" overhead).'
    echo 'Each version is run <n_runs> times and resulting statistics'
    echo 'are stored to .stat files in <outdir>. The number of grains to be'
    echo 'used will be <size3>^3, or 15^3=3375 by default. Requires perf(1),'
    echo 'which is part of the Linux kernel project.'
}
warn() { >&2 printf '%s\n' "$SCRIPTNAME: $1"; }
is_command() {
    if 1>/dev/null 2>&1 command -v "$1"; then
        return 0
    else
        warn "command '$1' not found"; return 1
    fi
}
has_pydrex_module() {
    python_version="$(python -V)"
    1>&2 2>/dev/null python -c "import importlib; importlib.import_module('pydrex')" \
        || { warn "cannot find python module 'pydrex' for $python_version"; return 1; }
}
is_command gfortran || exit 1
is_command perf || exit 1
is_command python || exit 1
has_pydrex_module || exit 1
is_command sed || exit 1

run_f90() {
    cp drex_forward_simpleshear.f90 "$OUTDIR"
    >/dev/null pushd "$OUTDIR"
    sed -i -E "s/size3 = [0-9]+/size3 = $SIZE3/" drex_forward_simpleshear.f90
    gfortran -O3 drex_forward_simpleshear.f90 -o run_drex
    perf stat -r $N_RUNS -o drex.stat ./run_drex >/dev/null
    >/dev/null popd
}

run_py3() {
    >/dev/null pushd "$OUTDIR"
    NUMBA_NUM_THREADS=1 perf stat -r $N_RUNS -o pydrex.stat python pydrex_forward_shear.py
    >/dev/null popd
}

clean() {
    # FIXME: This doesn't properly pick up the jobs??
    jobs="$(jobs -rp)"
    if [[ -n "$jobs" ]]; then
        kill -s TERM "$jobs"
    fi
    >/dev/null pushd "$OUTDIR"
    rm -f drex_forward_simpleshear.f90
    rm -f drex_forward_simpleshear.py
    rm -f *.mod
    rm -f run_drex
    >/dev/null popd
}

N_RUNS=10
SIZE3=15
OUTDIR="out"
while getopts "o:n:s:h" OPT; do
    case "$OPT" in
        o ) OUTDIR="$OPTARG" ;;
        n ) N_RUNS="$OPTARG" ;;
        s ) SIZE3="$OPTARG" ;;
        h ) usage; exit 0 ;;
        * ) usage; exit 1 ;;
    esac
done
shift $(( $OPTIND - 1 ))

[ -r "drex_forward_simpleshear.f90" ] || { usage; exit 1; }

mkdir -p "$OUTDIR"
cat << EOF > "$OUTDIR/pydrex_forward_shear.py"
import numpy as np
import numba as nb

from pydrex import core as _core
from pydrex import diagnostics as _diagnostics
from pydrex import io as _io
from pydrex import minerals as _minerals
from pydrex import utils as _utils
from pydrex import velocity as _velocity


shear_direction = np.array([0.0, 1.0, 0.0])
strain_rate = 1e-4
_, get_velocity_gradient = _velocity.simple_shear_2d("Y", "X", strain_rate)
timestamps = np.linspace(0, 1e4, 201)
params = _io.DEFAULT_PARAMS
params["number_of_grains"] = $SIZE3**3
params["gbs_threshold"] = 0
seed=245623452

mineral = _minerals.Mineral(
    phase=_core.MineralPhase.olivine,
    fabric=_core.MineralFabric.olivine_A,
    regime=_core.DeformationRegime.dislocation,
    n_grains=params["number_of_grains"],
    seed=seed,
)
deformation_gradient = np.eye(3)
θ_fse = np.empty_like(timestamps)
θ_fse[0] = 45


@nb.njit
def get_position(t):
    return np.zeros(3)

for t, time in enumerate(timestamps[:-1], start=1):
    deformation_gradient = mineral.update_orientations(
        params,
        deformation_gradient,
        get_velocity_gradient,
        pathline=(time, timestamps[t], get_position),
    )
    _, fse_v = _diagnostics.finite_strain(deformation_gradient)
    θ_fse[t] = _diagnostics.smallest_angle(fse_v, shear_direction)


Cij_and_friends = _diagnostics.elasticity_components(
    _minerals.voigt_averages([mineral], params)
)
angles = [
    _diagnostics.smallest_angle(v, shear_direction)
    for v in Cij_and_friends["hexagonal_axis"]
]
precent_anisotropy = Cij_and_friends["percent_anisotropy"]

EOF

run_f90 &
run_py3 &
wait $(jobs -rp)

# Clean up jobs & intermediate files, also when SIGTERM/HUP/INT is received.
trap "clean" EXIT HUP TERM INT
clean
