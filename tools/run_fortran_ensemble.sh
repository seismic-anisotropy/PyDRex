#!/bin/sh
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

main() {
    seed=$(shuf -i 100000-999999 -n 1)
    sed -i -E "s/state = [0-9]+/state = $seed/" drex_forward_simpleshear.f90
    gfortran drex_forward_simpleshear.f90 -o run_drex_"$1"
    ./run_drex_"$1"|tr -s ' '>"${OUTDIR%%/}"/out_"$1".txt
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

mkdir -p "$OUTDIR"
for run in $(seq 1 $N_RUNS); do
    if [[ $(jobs -r -p|wc -l) -ge $N_CPUS ]]; then
        wait -n
    fi
    main $run &
done
wait
rm *.mod
rm run_drex_*
