r"""> PyDRex: Methods to calculate texture and strain diagnostics.

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

import functools as ft

import numba as nb
import numpy as np
import scipy.linalg as la

from pydrex import geometry as _geo
from pydrex import logger as _log
from pydrex import stats as _stats
from pydrex import tensors as _tensors
from pydrex import utils as _utils

Pool, HAS_RAY = _utils.import_proc_pool()
if HAS_RAY:
    import ray

    from pydrex import distributed as _dstr


def elasticity_components(voigt_matrices):
    """Calculate elasticity decompositions for the given elasticity tensors.

    Args:
    - `voigt_matrices` (array) — the Nx6x6 Voigt matrix representations of the averaged
      elasticity tensors for a series of polycrystal textures

    Returns a dictionary with the following data series:
    - `'bulk_modulus'` — the computed bulk modulus for each Voigt matrix C
    - `'shear_modulus'` — the computed shear modulus for each Voigt matrix C
    - `'percent_anisotropy'` — for each Voigt matrix C, the total percentage of the
      norm of the elastic tensor ||C|| that deviates from the norm of the "closest"
      isotropic elasticity tensor
    - `'percent_hexagonal'` — for each C, the percentage of ||C|| attributed to
      hexagonally symmetric minerals
    - `'percent_tetragonal'` — for each C, the percentage of ||C|| attributed to
      tetragonally symmetric minerals
    - `'percent_orthorhombic'` — for each C, the percentage of ||C|| attributed to
      orthorhombically symmetric minerals
    - `'percent_monoclinic'` — for each C, the percentage of ||C|| attributed to
      monoclinically symmetric minerals
    - `'percent_triclinic'` — for each C, the percentage of ||C|| attributed to
      triclinically "symmetric" minerals (no mirror planes)
    - `'hexagonal_axis'` — for each C, the axis of hexagonal symmetry for the "closest"
      hexagonally symmetric approximation to C, a.k.a. the "transverse isotropy" axis

    .. note::
        Only 5 symmetry classes are relevant for elasticity decomposition,
        compared to the usual 6 used to describe crystal families.
        Crystals with cubic symmetry contribute to the isotropic elasticity tensor,
        because the lattice spacing is identical in all orthogonal directions.
        Note also that the trigonal crystal *system* is not a crystal family
        (it belongs to the hexagonal family).

    """
    n_matrices = len(voigt_matrices)
    out = {
        "bulk_modulus": np.empty(n_matrices),
        "shear_modulus": np.empty(n_matrices),
        "percent_anisotropy": np.empty(n_matrices),
        "percent_hexagonal": np.empty(n_matrices),
        "percent_tetragonal": np.empty(n_matrices),
        "percent_orthorhombic": np.empty(n_matrices),
        "percent_monoclinic": np.empty(n_matrices),
        "percent_triclinic": np.empty(n_matrices),
        "hexagonal_axis": np.empty((n_matrices, 3)),
    }
    for m, matrix in enumerate(voigt_matrices):
        voigt_matrix = _tensors.upper_tri_to_symmetric(matrix)
        stiffness_dilat, stiffness_deviat = _tensors.voigt_decompose(voigt_matrix)
        K = np.trace(stiffness_dilat) / 9  # Bulk modulus
        G = (np.trace(stiffness_deviat) - 3 * K) / 10  # Shear modulus
        out["bulk_modulus"][m] = K
        out["shear_modulus"][m] = G

        # Appendix A5, Browaeys & Chevrot, 2004.
        # The isotropic stiffness vector is independent of the coordinate system.
        isotropic_vector = np.hstack(
            (
                np.repeat(K + 4 * G / 3, 3),
                np.repeat(np.sqrt(2) * (K - 2 * G / 3), 3),
                np.repeat(2 * G, 3),
                np.repeat(0, 12),
            )
        )
        voigt_vector = _tensors.voigt_matrix_to_vector(voigt_matrix)
        out["percent_anisotropy"][m] = (
            la.norm(voigt_vector - isotropic_vector) / la.norm(voigt_vector) * 100
        )

        # Search for SCCS axes (orthogonal axes defined intrinsically by
        # eigenstrains of the elastic tensor).
        unpermuted_SCCS = np.empty((3, 3))
        eigv_dij = la.eigh(stiffness_dilat)[1]
        eigv_vij = la.eigh(stiffness_deviat)[1]
        # Averaging of eigenvectors to get the symmetry axes.
        for i in range(3):
            index_vij = 0  # [i(+1?)] of v_ij = eigvect that is nearest to the d_ij one.
            angle = 10  # Initialise to any number > 2π radians.
            for j in range(3):
                # Calculate angle between a pair of eigenvectors.
                # One eigenvector is taken from v_ij and one from d_ij.
                # Do not distinguish between vectors in opposite directions.
                dot_eigvects = np.dot(eigv_dij[:, i], eigv_vij[:, j])
                angle_eigvects = smallest_angle(eigv_dij[:, i], eigv_vij[:, j])
                if angle_eigvects < angle:
                    angle = angle_eigvects
                    # NOTE: Differences between ASPECT And original implementation:
                    # - In C++, std::copysign(1.0, 0.0) returns -1.
                    # - In Fortran 90, sign(1.0, 0.0) returns 1.
                    # - ASPECT implementation uses 'j' instead of 'j+1', doesn't seem to
                    # make a difference using the equivalent change for vec_SCSS = ...
                    index_vij = np.sign(dot_eigvects) * j if dot_eigvects != 0 else j
                    # index_vij = (
                    #     np.sign(dot_eigvects) * (j + 1)
                    #     if dot_eigvects != 0
                    #     else (j + 1)
                    # )

            # Add/subtract the nearest v_ij eigenvector multiplied by the signed index,
            # which effectively rotates the d_ij eigenvector, and then normalise it.
            vec_SCCS = (
                eigv_dij[:, i] + index_vij * eigv_vij[:, int(abs(index_vij))]
            ) / 2
            # vec_SCSS = (
            #     eigv_dij[:, i] + index_vij * eigv_vij[:, int(abs(index_vij) - 1)]
            # ) / 2
            vec_SCCS /= la.norm(vec_SCCS)
            unpermuted_SCCS[:, i] = vec_SCCS

        # Determine SCCS permutation that gives best hexagonal approximation.
        # This is achieved by minimising the "distance" between the voigt vector and its
        # projection onto the hexagonal symmetry subspace.
        # The elastic tensor is rotated into this system for decomposition.
        elastic_tensor = _tensors.voigt_to_elastic_tensor(voigt_matrix)
        distance = la.norm(voigt_vector)
        for i in range(3):
            permuted_SCCS = unpermuted_SCCS[:, [(i + j) % 3 for j in range(3)]]
            rotated_voigt_matrix = _tensors.elastic_tensor_to_voigt(
                _tensors.rotate(elastic_tensor, permuted_SCCS.transpose())
            )
            rotated_voigt_vector = _tensors.voigt_matrix_to_vector(rotated_voigt_matrix)
            mono_and_higher_vector = _tensors.mono_project(rotated_voigt_vector)
            tric_vector = rotated_voigt_vector - mono_and_higher_vector
            ortho_and_higher_vector = _tensors.ortho_project(mono_and_higher_vector)
            tetr_and_higher_vector = _tensors.tetr_project(ortho_and_higher_vector)
            hex_and_higher_vector = _tensors.hex_project(tetr_and_higher_vector)
            mono_vector = mono_and_higher_vector - ortho_and_higher_vector
            ortho_vector = ortho_and_higher_vector - tetr_and_higher_vector
            tetr_vector = tetr_and_higher_vector - hex_and_higher_vector
            hex_vector = hex_and_higher_vector - isotropic_vector

            δ = la.norm(rotated_voigt_vector - hex_and_higher_vector)
            if δ < distance:
                distance = δ

                percent = 100 / la.norm(voigt_vector)
                # print("\n", _tensors.voigt_vector_to_matrix(hex_vector))
                out["percent_triclinic"][m] = la.norm(tric_vector) * percent
                out["percent_monoclinic"][m] = la.norm(mono_vector) * percent
                out["percent_orthorhombic"][m] = la.norm(ortho_vector) * percent
                out["percent_tetragonal"][m] = la.norm(tetr_vector) * percent
                out["percent_hexagonal"][m] = la.norm(hex_vector) * percent
                # Last SCCS axis is always the hexagonal symmetry axis.
                out["hexagonal_axis"][m, ...] = permuted_SCCS[:, 2]
    return out


def bingham_average(orientations, axis="a"):
    """Compute Bingham average of orientation matrices.

    Returns the antipodally symmetric average orientation
    of the given crystallographic `axis`, or the a-axis by default.
    Valid axis specifiers are "a" for [100], "b" for [010] and "c" for [001].

    See also: [Watson (1966)](https://doi.org/10.1086%2F627211),
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
    mean_vector = la.eigh(_stats._scatter_matrix(np.asarray(orientations), row))[1][
        :, -1
    ]
    return mean_vector / la.norm(mean_vector)


def finite_strain(deformation_gradient, driver="ev"):
    """Extract measures of finite strain from the deformation gradient.

    Extracts the largest principal strain value and the vector defining the axis of
    maximum extension (longest semiaxis of the finite strain ellipsoid) from the 3x3
    deformation gradient tensor.

    """
    # Get eigenvalues and eigenvectors of the left stretch (Cauchy-Green) tensor.
    B_λ, B_v = la.eigh(
        deformation_gradient @ deformation_gradient.transpose(),
        driver=driver,
    )
    # Stretch ratio is sqrt(λ) - 1, the (-1) is so that undeformed strain is 0 not 1.
    return np.sqrt(B_λ[-1]) - 1, B_v[:, -1]


def symmetry_pgr(orientations, axis="a"):
    r"""Compute texture symmetry eigenvalue diagnostics from grain orientation matrices.

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

    See e.g. [Vollmer (1990)](https://doi.org/10.1130/0016-7606(1990)102%3C0786:AAOEMT%3E2.3.CO;2).

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

    scatter = _stats._scatter_matrix(orientations, row)
    # SciPy uses lower triangular entries by default, no need for all components.
    eigvals_descending = la.eigvalsh(scatter)[::-1]
    sum_eigvals = np.sum(eigvals_descending)
    return (
        (eigvals_descending[0] - eigvals_descending[1]) / sum_eigvals,
        2 * (eigvals_descending[1] - eigvals_descending[2]) / sum_eigvals,
        3 * eigvals_descending[2] / sum_eigvals,
    )


def misorientation_indices(
    orientation_stack,
    system: _geo.LatticeSystem,
    bins=None,
    ncpus=None,
    pool=None,
):
    """Calculate M-indices for a series of polycrystal textures.

    Calculate M-index using `misorientation_index` for a series of texture snapshots.
    The `orientation_stack` is a NxMx3x3 array of orientations where N is the number of
    texture snapshots and M is the number of grains.

    Uses either Ray or the Python multiprocessing library to calculate texture indices
    for multiple snapshots simultaneously. The arguments `ncpus` and `pool` are only
    relevant the latter option: if `ncpus` is `None` the number of CPU cores to use is
    chosen automatically based on the maximum number available to the Python
    interpreter, otherwise the specified number of cores is requested. Alternatively, an
    existing instance of `multiprocessing.Pool` can be provided.

    If Ray is installed, it will be automatically preferred. In this case, the number of
    processors (actually Ray “workers”) should be set upon initialisation of the Ray
    cluster (which can be distributed over the network).

    See `misorientation_index` for documentation of the remaining arguments.

    """
    if ncpus is not None and pool is not None:
        _log.warning("ignoring `ncpus` argument because a Pool was provided")
    m_indices = np.empty(len(orientation_stack))
    _run = ft.partial(
        misorientation_index,
        system=system,
        bins=bins,
    )
    if pool is None:
        if ncpus is None:
            ncpus = _utils.default_ncpus()
        with Pool(processes=ncpus) as pool:
            for i, out in enumerate(pool.imap(_run, orientation_stack)):
                m_indices[i] = out
    else:
        if HAS_RAY:
            m_indices = np.array(
                ray.get(
                    [
                        _dstr.misorientation_index.remote(
                            ray.put(a), system=system, bins=bins
                        )
                        for a in orientation_stack
                    ]
                )
            )
        else:
            for i, out in enumerate(pool.imap(_run, orientation_stack)):
                m_indices[i] = out
    return m_indices


def misorientation_index(orientations, system: _geo.LatticeSystem, bins=None):
    r"""Calculate M-index for polycrystal orientations.

    The `bins` argument is passed to `numpy.histogram`.
    If left as `None`, 1° bins will be used as recommended by the reference paper.
    The `symmetry` argument specifies the lattice system which determines intrinsic
    symmetry degeneracies and the maximum allowable misorientation angle.
    See `_geo.LatticeSystem` for supported systems.

    .. warning::
        This method must be able to allocate an array of shape
        $ \frac{N!}{2(N-2)!}× M^{2} $
        for N the length of `orientations` and M the number of symmetry operations for
        the given `system`.

    See [Skemer et al. (2005)](https://doi.org/10.1016/j.tecto.2005.08.023).

    """
    θmax = _stats._max_misorientation(system)
    misorientations_count, bin_edges = _stats.misorientation_hist(
        orientations, system, bins
    )
    # Compute counts of theoretical misorientation for an isotropic aggregate,
    # using the same bins (Skemer 2005 recommend 1° bins).
    misorientations_theory = np.array(
        [
            _stats.misorientations_random(bin_edges[i], bin_edges[i + 1], system)
            for i in range(len(misorientations_count))
        ]
    )
    # Equation 2 in Skemer 2005.
    return (θmax / (2 * len(misorientations_count))) * np.sum(
        np.abs(misorientations_theory - misorientations_count)
    )


def coaxial_index(orientations, axis1="b", axis2="a"):
    r"""Calculate coaxial “BA” index for a combination of two crystal axes.

    The BA index of [Mainprice et al. (2015)](https://doi.org/10.1144/SP409.8)
    is derived from the eigenvalue `symmetry` diagnostics and measures point vs girdle
    symmetry in the aggregate. $BA ∈ [0, 1]$ where $BA = 0$ corresponds to a perfect
    axial girdle texture and $BA = 1$ represents a point symmetry texture assuming that
    the random component $R$ is negligible. May be less susceptible to random
    fluctuations compared to the raw eigenvalue diagnostics.

    """
    P1, G1, _ = symmetry_pgr(orientations, axis=axis1)
    P2, G2, _ = symmetry_pgr(orientations, axis=axis2)
    return 0.5 * (2 - (P1 / (G1 + P1)) - (G2 / (G2 + P2)))


@nb.njit(fastmath=True)
def smallest_angle(vector, axis, plane=None):
    """Get smallest angle between a unit `vector` and a bidirectional `axis`.

    The axis is specified using either of its two parallel unit vectors.
    Optionally project the vector onto the `plane` (given by its unit normal)
    before calculating the angle.

    Examples:

    >>> from numpy import asarray as Ŋ
    >>> smallest_angle(Ŋ([1e0, 0e0, 0e0]), Ŋ([1e0, 0e0, 0e0]))
    0.0
    >>> smallest_angle(Ŋ([1e0, 0e0, 0e0]), Ŋ([0e0, 1e0, 0e0]))
    90.0
    >>> smallest_angle(Ŋ([1e0, 0e0, 0e0]), Ŋ([0e0, -1e0, 0e0]))
    90.0
    >>> smallest_angle(Ŋ([1e0, 0e0, 0e0]), Ŋ([np.sqrt(2), np.sqrt(2), 0e0]))
    45.0
    >>> smallest_angle(Ŋ([1e0, 0e0, 0e0]), Ŋ([-np.sqrt(2), np.sqrt(2), 0e0]))
    45.0

    """
    if plane is not None:
        _vector = vector - plane * np.dot(vector, plane)
    else:
        _vector = vector
    angle = np.rad2deg(
        np.arccos(
            np.clip(
                np.asarray(  # https://github.com/numba/numba/issues/3469
                    np.dot(_vector, axis)
                    / (np.linalg.norm(_vector) * np.linalg.norm(axis))
                ),
                -1,
                1,
            )
        )
    )
    if angle > 90:
        return 180 - angle
    return angle
