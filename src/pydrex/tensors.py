"""PyDRex: Tensor operation functions and helpers."""
import numba as nb
import numpy as np


PERMUTATION_SYMBOL = np.array(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)


def upper_tri_to_symmetric(arr):
    """Create symmetric array using upper triangle of input array."""
    # <https://stackoverflow.com/questions/58718365/fast-way-to-convert-upper-triangular-matrix-into-symmetric-matrix>
    upper_tri = np.triu(arr)
    return np.where(upper_tri, upper_tri, upper_tri.transpose())


def Voigt_to_elastic_tensor(matrix):
    """Create 4-th order elastic tensor from an equivalent Voigt matrix."""
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


def elastic_tensor_to_Voigt(tensor):
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


def Voigt_matrix_to_vector(matrix):
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


@nb.njit
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
