#!/bin/bash
set -u
readonly SCRIPTNAME="${0##*/}"
usage() {
    printf 'Usage: source %s\n' "$SCRIPTNAME"
    echo
    echo "Source this script in your PBS job file to set up a Ray cluster."
    echo "Stop the Ray cluster with 'ray stop' as the terminal line of the job."
    echo "If the environment variable RAY_NWORKERS is set to an integer,"
    echo "only that many workers will be started. Otherwise, the number of"
    echo "workers will be set to PBS_NCPUS."
}
warn() { >&2 printf '%s\n' "$SCRIPTNAME: $1"; }

# Check for the main prerequisites.
[ $# -gt 0 ] && { usage; exit 1; }
module prepend-path PATH "$HOME/.local/bin"
command -v ray || { warn "unknown command 'ray'"; exit 1; }
[ -x "pbs_start_ray_worker.sh" ] || { warn "cannot execute pbs_start_ray_worker.sh"; exit 1; }

USER_CFG=$PBS_O_WORKDIR/.cfg_${PBS_JOBID}

NWORKERS=${RAY_NWORKERS:-${PBS_NCPUS}}
NGPUS=$((PBS_NGPUS/PBS_NNODES))
NCPUS=$((NWORKERS/PBS_NNODES))
unset NWORKERS

# Choose Ray scheduler port in the range 8701-8800.
SCHEDULER_PORT=$(shuf -n 1 -i 8701-8800)
while  `ss -lntupw | grep -q ":$SCHEDULER_PORT" >& /dev/null`; do
    SCHEDULER_PORT=$(shuf -n 1 -i 8701-8800)  # SCHEDULER_PORT is taken.
done

# Choose Ray dashboard port in the range 8801-8900.
DASH_PORT=$(shuf -n 1 -i 8801-8900)
while  `ss -lntupw | grep -q ":$DASH_PORT" >& /dev/null`; do
    DASH_PORT=$(shuf -n 1 -i 8801-8900)  # DASH_PORT is taken.
done

LOG_FILE=${USER_CFG}/head_node.log
if [ ! -d ${USER_CFG} ]; then
    mkdir ${USER_CFG}
fi
touch $LOG_FILE

# Each node needs to load the modules as well.
module save ${USER_CFG}/module_coll
echo "#!/bin/bash" >${USER_CFG}/jobpython
echo "module restore  ${USER_CFG}/module_coll" >> ${USER_CFG}/jobpython
echo "module prepend-path PATH $HOME/.local/bin" >> ${USER_CFG}/jobpython
echo " python \$* " >> ${USER_CFG}/jobpython
chmod 755  ${USER_CFG}/jobpython

# Parameters used to wait for Ray worker connection.
TIMEOUT=300
INTERVAL=2

IP_PREFIX=`hostname -i`
IP_HEAD=${IP_PREFIX}:${SCHEDULER_PORT}

# Set resource numbers (per worker) for ray workers to pick up.
echo ${IP_HEAD} > ${USER_CFG}/ip_head
echo ${NGPUS} > ${USER_CFG}/ngpus
echo ${NCPUS} > ${USER_CFG}/ncpus
if [ ${PBS_NGPUS} -gt 0 ]; then
    GPU_MEM=$(( PBS_VMEM / PBS_NNODES / NGPUS))
    echo ${GPU_MEM} > ${USER_CFG}/mem_proc
else
    PROC_MEM=$(( PBS_VMEM / PBS_NNODES / NCPUS))
    echo ${PROC_MEM} > ${USER_CFG}/mem_proc
fi

# Start Ray scheduler on the head node.
ray start --head --node-ip-address=${IP_PREFIX} --port=${SCHEDULER_PORT}  \
    --dashboard-host=${IP_PREFIX}  --dashboard-port=${DASH_PORT} --num-cpus ${NCPUS} --num-gpus ${NGPUS} &>> ${LOG_FILE}

((t = TIMEOUT))
while [ ! -f ${LOG_FILE} ]; do
    sleep ${INTERVAL}
    ((t -= INTERVAL))
    while ((t <= 2)); do
        warn "scheduler failed to start up within $TIMEOUT seconds, aborting."
        exit 1
    done
done

((t = TIMEOUT))
while ! grep "Ray runtime started." ${LOG_FILE} >& /dev/null; do
    sleep ${INTERVAL}
    ((t -= INTERVAL))
    while ((t <= 2)); do
        warn "no ray runtime established within $TIMEOUT seconds, aborting"
        exit 1
    done
done

# File to store the ssh command that connects to the Ray head node.
if [ ! -e ${USER_CFG}/client_cmd  ]; then
    echo "ssh -N -L $DASH_PORT:`hostname`:$DASH_PORT ${USER}@gadi.nci.org.au " >& $USER_CFG/client_cmd
else
    echo "ssh -N -L $DASH_PORT:`hostname`:$DASH_PORT ${USER}@gadi.nci.org.au " >> $USER_CFG/client_cmd
fi

TOT_NPROCS=0
# Start Ray workers on the remaining nodes.
for node in `cat $PBS_NODEFILE | uniq`; do
    if [ $node != `hostname` ]; then
        pbs_tmrsh ${node} "${PBS_O_WORKDIR}/pbs_start_ray_worker.sh" &
    fi
    if [ ${PBS_NGPUS} -gt 0 ]; then
        TOT_NPROCS=$(( $TOT_NPROCS + $NGPUS ))
    else
        TOT_NPROCS=$(( $TOT_NPROCS + $NCPUS ))
    fi
done

echo "========== RAY cluster resources =========="
if [ ${PBS_NGPUS} -gt 0 ]; then
    echo "RAY NODE: GPU"
    echo "RAY WORKERS: ${NGPUS}/Node, ${TOT_NPROCS} in total."
    echo "RAY MEMORY: $(( GPU_MEM /1024/1024/1024 ))GiB/worker, $(( PBS_VMEM /1024/1024/1024 ))GiB in total."
else
    echo "RAY NODE: CPU"
    echo "RAY WORKERS: ${NCPUS}/Node, ${TOT_NPROCS} in total."
    echo "RAY MEMORY: $(( PROC_MEM /1024/1024/1024 ))GiB/worker, $(( PBS_VMEM /1024/1024/1024 ))GiB in total."
fi
