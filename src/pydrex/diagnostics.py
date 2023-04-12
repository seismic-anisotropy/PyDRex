r"""> PyDRex: Methods to calculate texture diagnostics.

.. note::
    Calculations expect orientation matrices $a$ to represent passive
    (i.e. alias) rotations, which are defined in terms of the extrinsic ZXZ
    euler angles $ϕ, θ, φ$ as
    $$
    a = \begin{bmatrix}
            \cosφ\cosϕ - \cosθ\sinϕ\sinφ & \cosθ\cosϕ\sinφ + \cosφ\sinϕ & \sinφ\sinθ \cr
            -\sinφ\cosϕ - \cosθ\sinϕ\cosφ & \cosθ\cosϕ\cosφ - \sinφ\sinϕ & \cosφ\sinθ \cr
            \sinθ\sinϕ & -\sinθ\cosϕ & \cosθ
        \end{bmatrix}
    $$
    such that a[i, j] gives the direction cosine of the angle between the i-th
    grain axis and the j-th external axis (in the global Eulerian frame).

"""
import itertools as it

import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation

from pydrex import stats as _st


def bingham_average(orientations, axis="a"):
    """Compute Bingham averages from olivine orientation matrices.

    Returns the antipodally symmetric average orientation
    of the given crystallographic `axis`, or the a-axis by default.
    Valid axis specifiers are "a" for [100], "b" for [010] and "c" for [001].

    See also: [Watson 1966](https://doi.org/10.1086%2F627211),
    [Mardia & Jupp, “Directional Statistics”](https://doi.org/10.1002/9780470316979).


    """
    match axis:
        case "a":
            row = 0
        case "b":
            row = 1
        case "c":
            row = 2
        case _:
            raise ValueError(f"axis must be 'a', 'b', or 'c', not {axis}")

    # https://courses.eas.ualberta.ca/eas421/lecturepages/orientation.html
    # Eigenvector corresponding to largest eigenvalue is the mean direction.
    # SciPy returns eigenvalues in ascending order (same order for vectors).
    # SciPy uses lower triangular entries by default, no need for all components.
    mean_vector = la.eigh(_st._scatter_matrix(orientations, row))[1][:, -1]
    return mean_vector / la.norm(mean_vector)


def symmetry(orientations, axis="a"):
    r"""Compute texture symmetry eigenvalue diagnostics from olivine orientation matrices.

    Compute Point, Girdle and Random symmetry diagnostics
    for ternary texture classification.
    Returns the tuple (P, G, R) where
    $$
    \begin{align\*}
    P &= (λ_{1} - λ_{2}) / N \cr
    G &= 2 (λ_{2} - λ_{3}) / N \cr
    R &= 3 λ_{3} / N
    \end{align\*}
    $$
    with $N$ the sum of the eigenvalues $λ_{1} ≥ λ_{2} ≥ λ_{3}$
    of the scatter (inertia) matrix.

    See e.g. [Vollmer 1990](https://doi.org/10.1130/0016-7606(1990)102%3C0786:AAOEMT%3E2.3.CO;2).

    """
    match axis:
        case "a":
            row = 0
        case "b":
            row = 1
        case "c":
            row = 2
        case _:
            raise ValueError(f"axis must be 'a', 'b', or 'c', not {axis}")

    scatter = _st._scatter_matrix(orientations, row)
    # SciPy uses lower triangular entries by default, no need for all components.
    eigvals_descending = la.eigvalsh(scatter)[::-1]
    sum_eigvals = np.sum(eigvals_descending)
    return (
        (eigvals_descending[0] - eigvals_descending[1]) / sum_eigvals,
        2 * (eigvals_descending[1] - eigvals_descending[2]) / sum_eigvals,
        3 * eigvals_descending[2] / sum_eigvals,
    )


def misorientation_index(orientations, bins=None, system=(2, 4)):
    """Calculate M-index for polycrystal orientations.

    The `bins` argument is passed to `numpy.histogram`.
    If left as `None`, 1° bins will be used as recommended by the reference paper.
    The symmetry system can be specified using the `system` argument.
    The default system is orthorhombic.

    See [Skemer et al. 2005](https://doi.org/10.1016/j.tecto.2005.08.023).

    """
    # Compute and bin misorientation angles from orientation data.
    misorientations_data = misorientation_angles(
        np.array(list(it.combinations(Rotation.from_matrix(orientations).as_quat(), 2)))
    )
    θmax = _st._max_misorientation(system)
    misorientations_count, bin_edges = np.histogram(
        misorientations_data, bins=θmax, range=(0, θmax), density=True
    )

    # Compute counts of theoretical misorientation for an isotropic aggregate,
    # using the same bins (Skemer 2005 recommend 1° bins).
    misorientations_theory = np.array(
        [
            _st.misorientations_random(bin_edges[i], bin_edges[i + 1])
            for i in range(len(misorientations_count))
        ]
    )

    # Equation 2 in Skemer 2005.
    return (θmax / (2 * len(misorientations_count))) * np.sum(
        np.abs(misorientations_theory - misorientations_count)
    )


def coaxial_index(orientations, axis1="b", axis2="a"):
    r"""Calculate coaxial “BA” index for a combination of two crystal axes.

    The BA index of [Mainprice et al. 2015](https://doi.org/10.1144/SP409.8)
    is derived from the eigenvalue `symmetry` diagnostics and measures point vs girdle
    symmetry in the aggregate. $BA \in [0, 1]$ where $BA = 0$ corresponds to a perfect
    axial girdle texture and $BA = 1$ represents a point symmetry texture assuming that
    the random component $R$ is negligible. May be less susceptible to random
    fluctuations compared to the raw eigenvalue diagnostics.

    """
    P1, G1, _ = symmetry(orientations, axis=axis1)
    P2, G2, _ = symmetry(orientations, axis=axis2)
    return 0.5 * (2 - (P1 / (G1 + P1)) - (G2 / (G2 + P2)))


def misorientation_angles(combinations):
    """Calculate the misorientation angles for pairs of rotation quaternions.

    Calculate the angular distance between the rotations `combinations[:, 0]`
    and `combinations[:, 1]`, which are expected to be 1x4 passive (alias)
    rotation quaternions.

    Uses ~25% less memory than the same operation with rotation matrices.

    See also:
    - <https://math.stackexchange.com/questions/90081/quaternion-distance>
    - <https://link.springer.com/article/10.1007/s10851-009-0161-2>


    """
    return 2 * np.rad2deg(
        np.arccos(
            np.abs(
                np.clip(
                    np.sum(combinations[:, 0] * combinations[:, 1], axis=1), -1.0, 1.0
                )
            )
        )
    )


def smallest_angle(vector, axis):
    """Get smallest angle between a unit `vector` and a bidirectional `axis`.

    The axis is specified using either of its two parallel unit vectors.

    """
    angle = np.rad2deg(
        np.arccos(np.dot(vector, axis) / (la.norm(vector) * la.norm(axis)))
    )
    if angle > 90:
        return 180 - angle
    return angle
