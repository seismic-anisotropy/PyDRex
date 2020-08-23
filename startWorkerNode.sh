#!/bin/bash -l

module load python3/3.7.4 vtk/8.2.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libsvml.so/directory

/path/to/ray start --block --address=$1 \
--redis-password=$2 --memory $((120 * 1024 * 1024 * 1024)) \
--object-store-memory $((20 * 1024 * 1024 * 1024)) \
--redis-max-memory $((10 * 1024 * 1024 * 1024)) \
--num-cpus 48 --num-gpus 0

/path/to/ray stop
