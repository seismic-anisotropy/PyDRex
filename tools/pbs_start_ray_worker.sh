#!/bin/bash
set -e
readonly SCRIPTNAME="${0##*/}"
warn() { >&2 printf '%s\n' "$SCRIPTNAME: $1"; }
[ $# -gt 0 ] && { warn "do not run this script directly, see pbs_start_ray_cluster.sh instead"; exit 1; }

# This sets up the PBS commands like 'module' which we need below.
source /etc/bashrc

ulimit -s unlimited
USER_CFG=$PBS_O_WORKDIR/.cfg_${PBS_JOBID}
HOSTNAME=`hostname`
JOBDIR=$PBS_JOBFS
cd $JOBDIR

NCPUS=`cat ${USER_CFG}/ncpus`
NGPUS=`cat ${USER_CFG}/ngpus`
IP_HEAD=`cat ${USER_CFG}/ip_head`

module restore  ${USER_CFG}/module_coll >& ${USER_CFG}/worker.${HOSTNAME}.log
module prepend-path PATH "$HOME/.local/bin"
echo "$SCRIPTNAME: starting worker for head node at $IP_HEAD with ${NCPUS} CPUs and ${NGPUS} GPUs" >> ${USER_CFG}/worker.${HOSTNAME}.log
ray start --address=$IP_HEAD  --num-cpus ${NCPUS} --num-gpus ${NGPUS} --block  &>> ${USER_CFG}/worker.${HOSTNAME}.log
