"""Find seeds that make grains start with an average SCCS direction of 45."""
import numpy as np

from pydrex import diagnostics as _d
from pydrex import logger as _l
from pydrex import minerals as _m

for i in range(10000):
    with _l.handler_level("ERROR"):
        m = _m.Mineral(seed=i, n_grains=4394)
        if (
            np.abs(
                _d.smallest_angle(
                    _d.elasticity_components(
                        _m.voigt_averages([m], {"olivine_fraction": 1})[0]
                    )["hexagonal_axis"][0],
                    [1, 0, 0],
                )
                - 45
            )
            < 0.1
        ):
            print(i)
