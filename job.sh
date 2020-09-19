#!/bin/bash

#PBS -P xd2
#PBS -q normal
#PBS -l ncpus=192
#PBS -l mem=768GB
#PBS -l walltime=1:00:00
#PBS -l wd

module load python3/3.8.5 vtk/8.2.0
export PYTHONPATH=$PYTHONPATH:/home/157/td5646/numpy

################################################################################

# Charm4Py
module load openmpi/4.0.2
mpirun -np $PBS_NCPUS python3 PyDRex.py ridge_31.vtu --charm

################################################################################

# Ray
# ip_prefix=`hostname -i`
# port='6379'
# ip_head=${ip_prefix}:${port}
# redis_password=$(uuidgen)
#
# /home/157/td5646/.local/bin/ray start --head --port=${port} \
# --redis-password=$redis_password
# sleep 5
#
# cpusPerNode=48
# for (( n=$cpusPerNode; n<$PBS_NCPUS; n+=$cpusPerNode ))
# do
#   pbsdsh -n $n -v ./startWorkerNode.sh $ip_head $redis_password &
# done
# sleep 5
#
# ./PyDRex.py INPUT --ray --redis-pass $redis_password
#
# /home/157/td5646/.local/bin/ray stop

################################################################################

# Multiprocessing | Only valid on a single-node system
# ./PyDRex.py INPUT --cpus $PBS_NCPUS
