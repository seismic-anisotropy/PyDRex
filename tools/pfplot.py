import sys
from pydrex import visualisation as _vis

_vis.polefigures(
    sys.argv[1],
    postfix=sys.argv[2],
    i_range=range(*[int(s) for s in sys.argv[3].split(":")]),
    density=bool(int(sys.argv[4])),
    ref_axes=sys.argv[5],
)
