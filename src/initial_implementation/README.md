# PyDRex
Python 3 Implementation of the FORTRAN program D-Rex

The original FORTRAN implementation of D-Rex can be downloaded from the [webpage](http://www.ipgp.fr/~kaminski/web_doudoud/DRex.tar.gz) of Edouard Kaminski. Theoretical considerations about the framework have been detailed in multiple article publications, such as [D-Rex, a program for calculation of seismic anisotropy due to crystal lattice preferred orientation in the convective upper mantle](https://doi.org/10.1111/j.1365-246X.2004.02308.x).

The present implementation (`PyDRex.py`) replicates the original framework and generates the same set of output variables. The code can process both 2-D and 3-D input datasets, and represents rocks deformation as a combination of diffusion and dislocation creep. Calculations can be performed in serial or in parallel, on a single machine or an entire cluster, with efficient checkpointing.

Alternative implementations of translated, original FORTRAN routines are provided (`AlternativeFunctions.py`). In particular, pathlines are now calculated using Scipy's [solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), independently of the time-step of integration in *strain* and, hence, independently of the grid resolution specified to the code. Pathlines generated this way can be visualised using `pathlineIVP`. As a result, *strain* has also been updated to accommodate the changes in *pathline*. Moreover, a version of *strain* using [solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) is included. Despite reproducing the results of the original *strain*, it is not as efficient, probably as a result of a less precise control of the integration time-step. The calculation of the Infinite Strain Axis (ISA) in *eigen* and *isacalc* has been updated to better reflect the analytical equations provided in Appendix B of [Kaminski, Ribe and Browaeys (2004)](https://doi.org/10.1111/j.1365-246X.2004.02308.x).

The current version expects a VTK file as an input, but any file format can be implemented as long as the necessary fields can be processed by the interpolator objects. Simulation parameters are gathered in `DRexParam.py`.

Parallelization on a cluster is achieved through either [Charm4Py](https://github.com/UIUC-PPL/charm4py) or [Ray](https://github.com/ray-project/ray); files used to execute the code on a cluster running PBS Pro are provided (`job.sh` and `startWorkerNode.sh`). Alternatively, example implementations for Slurm are documented both in the [Charm4Py](https://charmpy.readthedocs.io/en/latest/running.html#using-system-job-launchers) and [Ray](https://docs.ray.io/en/master/cluster/slurm.html) documentations.

The code relies on the following Python packages (and their dependencies):
- Numpy
- Scipy
- Numba
- VTK
- Charm4Py/Ray

In addition, `matplotlib` can be used to visualize the outputs (`PlotDRex.py`) and can also be used as an alternative for 2-D interpolation on a single machine. To benefit from [SVML](http://numba.pydata.org/numba-doc/latest/user/performance-tips.html#intel-svml) support, the package `icc_rt` also needs to be installed and the path to `libsvml.so` added the `LD_LIBRARY_PATH` environment variable.
