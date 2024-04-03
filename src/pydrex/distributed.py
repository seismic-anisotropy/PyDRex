"""> PyDRex: Shims and wrappers for distributed-memory multiprocessing."""
import ray

from pydrex import diagnostics as _diagnostics

@ray.remote
def misorientation_indices(*args, **kwargs):
    _diagnostics.misorientation_indices(*args, **kwargs)

@ray.remote
def misorientation_index(*args, **kwargs):
    _diagnostics.misorientation_index(*args, **kwargs)
