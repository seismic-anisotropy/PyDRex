# !!! EDIT THIS SCRIPT !!!
# This is an example script for a particular PBS cluster, and is not intended
# to be a portable solution. Modify as required to set up the appropriate
# environment on the HPC cluster you are using.
module purge
module prepend-path PATH ${HOME}/.local/bin
module append-path PYTHONPATH ${HOME}/vcs/fluidity/python
module append-path PYTHONPATH ${HOME}/vcs/fluidity/tools
module use --append /g/data/xd2/modulefiles
module load cmake/3.24.2 fftw3/3.3.8 gmsh/2.16.0 hdf5/1.10.5p netcdf/4.7.1 \
    openmpi/4.0.1 petsc/3.12.2 python3/3.11.7 szip/2.1.1 udunits/2.2.26 \
    vtk/8.2.0 zoltan/3.83-big-buffers python3-as-python
# module load hdf5/1.12.2p  python3/3.12.1 udunits/2.2.26 netcdf/4.9.2p \
#     vtk/8.2.0 openmpi/4.0.7 cmake/3.24.2 gmsh/4.4.1 fftw3/3.3.8 petsc/3.17.4 \
#     zoltan/3.901 szip/2.1.1 python3-as-python

ulimit -c unlimited
export PETSC_OPTIONS="-no_signal_handler -mg_levels_esteig_ksp_type cg -ksp_chebyshev_esteig_random -mg_levels_esteig_ksp_max_it 30 -pc_gamg_square_graph 100"
