#!/bin/bash
#PBS -P xd2
#PBS -q normalsr
#PBS -l walltime=10:00:00
#PBS -l ncpus=1
#PBS -l mem=500GB
#PBS -l storage=scratch/xd2+gdata/xd2
#PBS -l wd
#PBS -o run_advection2d.log
#PBS -e run_advection2d.err

source ./load_modules.sh

make -f ./advection2d/Makefile
