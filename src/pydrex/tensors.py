"""> PyDRex: Tensor operation functions and helpers.

For Voigt notation, the symmetric 6x6 matrix representation is used,
which assumes that the fourth order tensor being represented as such is also symmetric.
The vectorial notation uses 21 components which are the independent components of the
symmetric 6x6 matrix.

"""

import numba as nb
import numpy as np

PERMUTATION_SYMBOL = np.array(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)


@nb.njit(fastmath=True)
def voigt_decompose(matrix):
    """Decompose elastic tensor (as 6x6 Voigt matrix) into distinct contractions.

    Return the only two independent contractions of the elastic tensor given as a 6x6
    Voigt `matrix`. With reference to the full 4-th order elastic tensor, the
    contractions are defined as:
    - $d_{ij} = C_{ijkk}$ (dilatational stiffness tensor)
    - $v_{ij} = C_{ijkj}$ (deviatoric stiffness tensor)

    Any vector which is an eigenvector of both $d_{ij}$ and $v_{ij}$ is always normal to
    a symmetry plane of the elastic medium.

    See Equations 3.4 & 3.5 in [Browaeys & Chevrot (2004)](https://doi.org/10.1111/j.1365-246X.2004.02415.x).

    """
    # 1. Compose dᵢⱼ = Cᵢⱼₖₖ (dilatational stiffness tensor)
    # Eq. 3.4 in Browaeys & Chevrot, 2004.
    stiffness_dilat = np.empty((3, 3))
    for i in range(3):
        stiffness_dilat[i, i] = matrix[:3, i].sum()
    stiffness_dilat[0, 1] = stiffness_dilat[1, 0] = matrix[:3, 5].sum()
    stiffness_dilat[0, 2] = stiffness_dilat[2, 0] = matrix[:3, 4].sum()
    stiffness_dilat[1, 2] = stiffness_dilat[2, 1] = matrix[:3, 3].sum()
    # 2. Compose vᵢⱼ = Cᵢⱼₖⱼ (deviatoric stiffness tensor)
    # Eq. 3.5, Browaeys & Chevrot, 2004.
    stiffness_deviat = np.empty((3, 3))
    stiffness_deviat[0, 0] = matrix[0, 0] + matrix[4, 4] + matrix[5, 5]
    stiffness_deviat[1, 1] = matrix[1, 1] + matrix[3, 3] + matrix[5, 5]
    stiffness_deviat[2, 2] = matrix[2, 2] + matrix[3, 3] + matrix[4, 4]
    stiffness_deviat[0, 1] = matrix[0, 5] + matrix[1, 5] + matrix[3, 4]
    stiffness_deviat[0, 2] = matrix[0, 4] + matrix[2, 4] + matrix[3, 5]
    stiffness_deviat[1, 2] = matrix[1, 3] + matrix[2, 3] + matrix[4, 5]
    stiffness_deviat = upper_tri_to_symmetric(stiffness_deviat)
    return stiffness_dilat, stiffness_deviat


@nb.njit(fastmath=True)
def mono_project(voigt_vector):
    """Project 21-component `voigt_vector` onto monoclinic symmetry subspace.

    Monoclinic symmetry is characterised by 13 independent elasticity components.

    See [Browaeys & Chevrot (2004)](https://doi.org/10.1111/j.1365-246X.2004.02415.x).

    """
    out = voigt_vector.copy()
    inds = (9, 10, 12, 13, 15, 16, 18, 19)
    for i in inds:
        out[i] = 0
    return out


@nb.njit(fastmath=True)
def ortho_project(voigt_vector):
    """Project 21-component `voigt_vector` onto orthorhombic symmetry subspace.

    Orthorhombic symmetry is characterised by 9 independent elasticity components.

    See [Browaeys & Chevrot (2004)](https://doi.org/10.1111/j.1365-246X.2004.02415.x).

    """
    out = voigt_vector.copy()
    out[9:] = 0
    return out


@nb.njit(fastmath=True)
def tetr_project(voigt_vector):
    """Project 21-component `voigt_vector` onto tetragonal symmetry subspace.

    Tetragonal symmetry is characterised by 6 independent elasticity components.

    See [Browaeys & Chevrot (2004)](https://doi.org/10.1111/j.1365-246X.2004.02415.x).

    """
    out = ortho_project(voigt_vector)
    for i, j in ((0, 1), (3, 4), (6, 7)):
        for k in range(2):
            out[i + k] = 0.5 * (voigt_vector[i] + voigt_vector[j])
    return out


@nb.njit(fastmath=True)
def hex_project(voigt_vector):
    """Project 21-component `voigt_vector` onto hexagonal symmetry subspace.

    Hexagonal symmetry (a.k.a. transverse isotropy) is characterised by 5 independent
    elasticity components.

    See [Browaeys & Chevrot (2004)](https://doi.org/10.1111/j.1365-246X.2004.02415.x).

    """
    x = voigt_vector
    out = np.zeros(21)
    out[0] = out[1] = 3 / 8 * (x[0] + x[1]) + x[5] / 4 / np.sqrt(2) + x[8] / 4
    out[2] = x[2]
    out[3] = out[4] = (x[3] + x[4]) / 2
    out[5] = (x[0] + x[1]) / 4 / np.sqrt(2) + 3 / 4 * x[5] - x[8] / 2 / np.sqrt(2)
    out[6] = out[7] = (x[6] + x[7]) / 2
    out[8] = (x[0] + x[1]) / 4 - x[5] / 2 / np.sqrt(2) + x[8] / 2
    return out


@nb.njit(fastmath=True)
def upper_tri_to_symmetric(arr):
    """Create symmetric array using upper triangle of input array.

    Examples:

    >>> import numpy as np
    >>> upper_tri_to_symmetric(np.array([
    ...         [ 1.,  2.,  3.,  4.],
    ...         [ 0.,  5.,  6.,  7.],
    ...         [ 0.,  0.,  8.,  9.],
    ...         [ 9.,  0.,  0., 10.]
    ... ]))
    array([[ 1.,  2.,  3.,  4.],
           [ 2.,  5.,  6.,  7.],
           [ 3.,  6.,  8.,  9.],
           [ 4.,  7.,  9., 10.]])

    """
    # <https://stackoverflow.com/questions/58718365/fast-way-to-convert-upper-triangular-matrix-into-symmetric-matrix>
    upper_tri = np.triu(arr)
    return np.where(upper_tri, upper_tri, upper_tri.transpose())


@nb.njit(fastmath=True)
def voigt_to_elastic_tensor(matrix):
    """Create 4-th order elastic tensor from an equivalent Voigt matrix.

    See also: `elastic_tensor_to_voigt`.

    """
    tensor = np.empty((3, 3, 3, 3))
    for p in range(3):
        for q in range(3):
            delta_pq = 1 if p == q else 0
            i = (p + 1) * delta_pq + (1 - delta_pq) * (7 - p - q) - 1
            for r in range(3):
                for s in range(3):
                    delta_rs = 1 if r == s else 0
                    j = (r + 1) * delta_rs + (1 - delta_rs) * (7 - r - s) - 1
                    tensor[p, q, r, s] = matrix[i, j]
    return tensor


@nb.njit(fastmath=True)
def elastic_tensor_to_voigt(tensor):
    """Create a 6x6 Voigt matrix from an equivalent 4-th order elastic tensor."""
    matrix = np.zeros((6, 6))
    matrix_indices = np.zeros((6, 6))
    for p in range(3):
        for q in range(3):
            delta_pq = 1 if p == q else 0
            i = (p + 1) * delta_pq + (1 - delta_pq) * (7 - p - q) - 1
            for r in range(3):
                for s in range(3):
                    delta_rs = 1 if r == s else 0
                    j = (r + 1) * delta_rs + (1 - delta_rs) * (7 - r - s) - 1
                    matrix[i, j] += tensor[p, q, r, s]
                    matrix_indices[i, j] += 1
    matrix /= matrix_indices
    return (matrix + matrix.transpose()) / 2


@nb.njit(fastmath=True)
def voigt_matrix_to_vector(matrix):
    """Create the 21-component Voigt vector equivalent to the 6x6 Voigt matrix."""
    vector = np.zeros(21)
    for i in range(3):
        vector[i] = matrix[i, i]
        vector[i + 3] = np.sqrt(2) * matrix[(i + 1) % 3, (i + 2) % 3]
        vector[i + 6] = 2 * matrix[i + 3, i + 3]
        vector[i + 9] = 2 * matrix[i, i + 3]
        vector[i + 12] = 2 * matrix[(i + 2) % 3, i + 3]
        vector[i + 15] = 2 * matrix[(i + 1) % 3, i + 3]
        vector[i + 18] = 2 * np.sqrt(2) * matrix[(i + 1) % 3 + 3, (i + 2) % 3 + 3]
    return vector


@nb.njit(fastmath=True)
def voigt_vector_to_matrix(vector):
    """Create the 6x6 matrix representation of the 21-component Voigt vector.

    See also: `voigt_matrix_to_vector`.

    """
    matrix = np.zeros((6, 6))
    for i in range(3):
        matrix[i, i] = vector[i]
        matrix[i + 3, i + 3] = 0.5 * vector[i + 6]

    matrix[1, 2] = 1 / np.sqrt(2) * vector[3]
    matrix[0, 2] = 1 / np.sqrt(2) * vector[4]
    matrix[0, 1] = 1 / np.sqrt(2) * vector[5]

    matrix[0, 3] = 0.5 * vector[9]
    matrix[1, 4] = 0.5 * vector[10]
    matrix[2, 5] = 0.5 * vector[11]
    matrix[2, 3] = 0.5 * vector[12]

    matrix[0, 4] = 0.5 * vector[13]
    matrix[1, 5] = 0.5 * vector[14]
    matrix[1, 3] = 0.5 * vector[15]

    matrix[2][4] = 0.5 * vector[16]
    matrix[0][5] = 0.5 * vector[17]
    matrix[4][5] = 0.5 * 1 / np.sqrt(2) * vector[18]
    matrix[3][5] = 0.5 * 1 / np.sqrt(2) * vector[19]
    matrix[3][4] = 0.5 * 1 / np.sqrt(2) * vector[20]
    return upper_tri_to_symmetric(matrix)


@nb.njit(fastmath=True)
def rotate(tensor, rotation):
    """Rotate 4-th order tensor using a 3x3 rotation matrix."""
    rotated_tensor = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for L in range(3):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                for d in range(3):
                                    rotated_tensor[i, j, k, L] += (
                                        rotation[i, a]
                                        * rotation[j, b]
                                        * rotation[k, c]
                                        * rotation[L, d]
                                        * tensor[a, b, c, d]
                                    )
    return rotated_tensor
