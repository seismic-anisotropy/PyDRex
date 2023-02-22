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


def misorientation_index(orientations, bins="doane", system=(2, 4)):
    """Calculate M-index for polycrystal orientations.

    The `bins` argument is passed to `numpy.histogram`.
    The symmetry system can be specified using the `system` argument.
    The default system is orthorhombic.

    See [Skemer et al. 2005](https://doi.org/10.1016/j.tecto.2005.08.023).

    """
    misorientations = misorientation_angles(
        np.array(list(it.combinations(orientations, 2)))
    )
    θmax = _st._max_misorientation(system)
    count_misorientations, _ = np.histogram(
        misorientations, bins="doane", range=(0, θmax)
    )
    print(len(count_misorientations))
    return (1 / 2) * np.sum(
        np.abs(
            [
                _st.misorientations_random(i, i + 1)
                - count_misorientations[i] / len(misorientations)
                for i in range(len(count_misorientations))
            ]
        )
    )


def misorientation_angles(combinations):
    """Calculate the misorientation angles for pairs of rotation matrices.

    Calculate the angular distance between the rotations `combinations[:, 0]`
    and `combinations[:, 1]`, which are expected to be 3x3 passive (alias)
    rotation matrices.

    See also <http://boris-belousov.net/2016/12/01/quat-dist/>.

    """
    return np.rad2deg(
        np.arccos(
            np.clip(
                (
                    np.trace(
                        combinations[:, 0]
                        @ np.transpose(combinations[:, 1], axes=[0, 2, 1]),
                        axis1=1,
                        axis2=2,
                    )
                    - 1.0
                )
                / 2,
                -1.0,
                1.0,
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
