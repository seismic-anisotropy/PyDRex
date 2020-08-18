# PyDRex
Python 3 Implementation of the FORTRAN program D-Rex

The original FORTRAN implementation of D-Rex can be downloaded from the [webpage](http://www.ipgp.fr/~kaminski/web_doudoud/DRex.tar.gz) of Edouard Kaminski. Theoretical considerations about the framework have been detailed in multiple article publications, such as [D-Rex, a program for calculation of seismic anisotropy due to crystal lattice preferred orientation in the convective upper mantle](https://doi.org/10.1111/j.1365-246X.2004.02308.x).

The present implementation (`PyDRex.py`) replicates the original framework and aims to generate the same set of variables output. The calculation of the Infinite Strain Axis (ISA) has been updated to better reflect the data provided in Appendix B of [Kaminski, Ribe and Browaeys (2004)](https://doi.org/10.1111/j.1365-246X.2004.02308.x); the original method is commented out. The code can process both 2-D and 3-D input datasets which describe rocks deformation as a combination of diffusion and dislocation creep. Calculations can be performed in serial or in parallel, on a single machine or an entire cluster, with efficient checkpointing.

The current version expects a VTK file as an input, but any file format can be implemented as long as the necessary fields can be processed by the interpolator objects. Simulation parameters are gathered in `DRexParam.py`.

Parallelization on a cluster is achieved through [Ray](https://github.com/ray-project/ray) and currently implemented for a cluster running PBS Pro (`job.sh` and `startWorkerNode.sh`). An example implementation for Slurm is provided in the Ray [documentation](https://docs.ray.io/en/master/cluster/slurm.html).

The code relies on the following Python packages:
- Numpy
- Scipy
- Numba
- Ray
- VTK

In addition, matplotlib can be used to visualize the outputs and can also provide an alternative for 2-D interpolation on a single machine. To benefit from [SVML](http://numba.pydata.org/numba-doc/latest/user/performance-tips.html#intel-svml) support, the package `icc_rt` also needs to be installed and the path to `libsvml.so` added the `LD_LIBRARY_PATH` environment variable.
