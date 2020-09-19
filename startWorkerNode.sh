#!/bin/bash -l

module load python3/3.8.5 vtk/8.2.0

export PYTHONPATH=$PYTHONPATH:/home/157/td5646/numpy
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/157/td5646/.local/lib

/home/157/td5646/.local/bin/ray start --block --address=$1 \
--redis-password=$2 --num-cpus 48 --num-gpus 0

/home/157/td5646/.local/bin/ray stop
