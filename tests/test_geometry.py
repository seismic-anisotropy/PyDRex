"""> PyDRex: Tests for geometric conversions and projections."""
import numpy as np

from pydrex import geometry as _geo


def test_poles_example(example_outputs, stringify, hkl, ref_axes):
    """Test poles (directions of crystallographic axes) of example data."""
    ref_data = np.load(
        example_outputs / f"example_CPO_poles_{stringify(hkl)}{ref_axes}.npz"
    )
    resampled_data = np.load(example_outputs / "example_CPO_resampled.npz")
    xvals, yvals, zvals = _geo.poles(
        resampled_data["orientations"],
        hkl=hkl,
        ref_axes=ref_axes,
    )
    np.testing.assert_allclose(ref_data["xvals"], xvals, atol=1e-16, rtol=0)
    np.testing.assert_allclose(ref_data["yvals"], yvals, atol=1e-16, rtol=0)
    np.testing.assert_allclose(ref_data["zvals"], zvals, atol=1e-16, rtol=0)


def test_lambert_equal_area(rng):
    """Test Lambert equal area projection."""
    x, y = np.mgrid[-1:1:11j, -1:1:11j]
    x_flat = [j for i in x for j in i]
    y_flat = [j for i in y for j in i]
    # Uniform samples on the unit disk, this is tested in the Shirley doctest example.
    x_disk, y_disk = _geo.shirley_concentric_squaredisk(x_flat, y_flat)
    # Project onto the unit sphere by adding z = ± (1 - r).
    # Then project back onto the disk using Lambert equal-area, should be the same.
    sign = rng.integers(low=0, high=2, size=len(x_disk))
    x_laea, y_laea = _geo.lambert_equal_area(
        x_disk, y_disk, (-1) ** sign * (1 - (x_disk**2 + y_disk**2))
    )
    np.testing.assert_allclose(x_disk, x_laea, atol=1e-15, rtol=0)
    np.testing.assert_allclose(y_disk, y_laea, atol=1e-15, rtol=0)
