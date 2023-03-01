#!/bin/bash -l

module load python3/3.9.2 vtk/8.2.0

/home/157/td5646/.local/bin/ray start --block --address=$1 \
--redis-password=$2

/home/157/td5646/.local/bin/ray stop
