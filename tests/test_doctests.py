"""> PyDRex: Run doctests for applicable modules."""

import pydrex


def test_doctests():
    """Run doctests as well."""
    assert pydrex.tensors.__run_doctests().failed == 0
