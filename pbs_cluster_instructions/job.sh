#!/bin/bash

#PBS -P xd2
#PBS -q normal
#PBS -l ncpus=192
#PBS -l mem=150GB
#PBS -l walltime=1:00:00
#PBS -l wd

module load python3/3.9.2 vtk/8.2.0

################################## Charm4Py ###################################

module load openmpi/4.1.1
mpirun -x LD_PRELOAD=libmpi.so -np $PBS_NCPUS \
	python3.9 PyDRex.py INPUT --charm

##################################### Ray #####################################

# Ray
# port='6379'
# ip_head=`hostname -i`:$port
# redis_password=$(uuidgen)
#
# /home/157/td5646/.local/bin/ray start --head --port=$port \
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
# python3.9 PyDRex.py INPUT --ray --redis-pass $redis_password
#
# /home/157/td5646/.local/bin/ray stop

############################### Multiprocessing ###############################

# For use on a single-node system only
# python3.9 PyDRex.py INPUT --cpus $PBS_NCPUS
