#!/bin/bash
#PBS -l ncpus=192
#PBS -l mem=600GB
#PBS -l walltime=48:00:00
#PBS -l wd

module load python3/3.7.4 vtk/8.2.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libsvml.so/directory

ip_prefix=`hostname -i`
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

echo parameters: $ip_head $redis_password

/path/to/ray start --head --port=6379 \
--redis-password=$redis_password \
--memory $((120 * 1024 * 1024 * 1024)) \
--object-store-memory $((20 * 1024 * 1024 * 1024)) \
--redis-max-memory $((10 * 1024 * 1024 * 1024)) \
--num-cpus 48 --num-gpus 0
sleep 10

for (( n=48; n<$PBS_NCPUS; n+=48 ))
do
  pbsdsh -n $n -v /path/to/startWorkerNode.sh \
  $ip_head $redis_password &
  sleep 10
done

cd /path/to/working/directory || exit
./PyDRex.py input -m --pw $redis_password

/path/to/ray stop
