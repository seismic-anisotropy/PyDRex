#!/bin/bash -l

module load python3/3.7.4 vtk/8.2.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/157/td5646/.local/lib

/home/157/td5646/.local/bin/ray start --block --address=$1 \
--redis-password=$2 --memory $((90 * 1024 * 1024 * 1024)) \
--object-store-memory $((40 * 1024 * 1024 * 1024)) \
--redis-max-memory $((20 * 1024 * 1024 * 1024)) \
--num-cpus 48 --num-gpus 0

/home/157/td5646/.local/bin/ray stop
