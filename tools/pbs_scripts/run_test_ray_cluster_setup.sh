#!/bin/bash
#PBS -P xd2
#PBS -q normalbw
#PBS -l walltime=00:10:00
#PBS -l ncpus=56
#PBS -l mem=256GB
#PBS -l jobfs=400GB
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l wd
#PBS -o test_ray_cluster_setup.log
#PBS -e test_ray_cluster_setup.err
#PBS -N test_ray_cluster_setup

module purge
module load python3/3.11.7 python3-as-python
# NOTE: First run pip install 'ray[default]' in this python environment.

source pbs_start_ray_cluster.sh

python -c 'import ray; ray.init(address="auto"); print(f"nodes in cluster: {ray.nodes()}"); print(f"cluster resources: {ray.cluster_resources()}")'

ray stop
