#!/bin/bash
set -eu
readonly SCRIPTNAME="${0##*/}"
usage() {
    printf 'Usage: %s [-h]\n' "$SCRIPTNAME"
    printf '       %s ' "$SCRIPTNAME"
    echo '[-o outdir][-n n_runs][-p ncpus]'
    echo
    echo 'Run ensemble of simple shear A-type olivine runs using'
    echo ' ./drex_forward_simpleshear.f90'
}
is_command() {
    if 1>/dev/null 2>&1 command -v "$1"; then
        return 0
    else
        warn "command '$1' not found"; return 1
    fi
}
is_command seq || exit 1
is_command gfortran || exit 1
is_command shuf || exit 1
is_command sed || exit 1
is_command tr || exit 1
is_command wc || exit 1

main() {
    rundir="$OUTDIR"/run_"$1"
    mkdir -p "$rundir"
    cp drex_forward_simpleshear.f90 "$rundir"
    >/dev/null pushd "$rundir"
    seed=$(shuf -i 100000-999999 -n 1)
    sed -i -E "s/state = [0-9]+/state = $seed/" drex_forward_simpleshear.f90
    gfortran drex_forward_simpleshear.f90 -o run_drex_"$1"
    ./run_drex_"$1"|tr -s ' '>out_"$1".txt
    >/dev/null popd
}

clean() {
    >/dev/null pushd "$OUTDIR"
    jobs="$(jobs -rp)"
    if [[ -n "$jobs" ]]; then
        kill -s TERM "$jobs"
    fi
    >/dev/null popd
}

N_RUNS=10
N_CPUS=1
OUTDIR="out"
while getopts "o:n:p:h" OPT; do
    case "$OPT" in
        o ) OUTDIR="$OPTARG" ;;
        n ) N_RUNS="$OPTARG" ;;
        p ) N_CPUS="$OPTARG" ;;
        h ) usage; exit 0 ;;
        * ) usage; exit 1 ;;
    esac
done
shift $(( $OPTIND - 1 ))

[ -r "drex_forward_simpleshear.f90" ] || { usage; exit 1; }

# Run N_RUNS instances on N_CPUS workers, save output.
for run in $(seq 1 $N_RUNS); do
    if [[ $(jobs -rp|wc -l) -ge $N_CPUS ]]; then
        wait $(jobs -rp)
    fi
    main $run &
done
wait $(jobs -rp)

# Clean up background jobs, also when SIGTERM/HUP/INT is received.
trap "clean" EXIT HUP TERM INT
clean
