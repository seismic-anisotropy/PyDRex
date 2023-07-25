"""> PyDRex: Steady-state solutions of velocity gradients for various flows."""
import numpy as np

def simple_shear_2d(direction, deformation_plane, strain_rate):
    """Return simple shear velocity gradient callable f(x) for the given parameters."""
    shear = (direction, deformation_plane)
    match shear:
        case ("X", "Y") | ([1, 0, 0], [0, 1, 0]):
            shear = (0, 1)
        case ("X", "Z") | ([1, 0, 0], [0, 0, 1]):
            shear = (0, 2)
        case ("Y", "X") | ([0, 1, 0], [1, 0, 0]):
            shear = (1, 0)
        case ("Y", "Z") | ([0, 1, 0], [0, 0, 1]):
            shear = (1, 2)
        case ("Z", "X") | ([0, 0, 1], [1, 0, 0]):
            shear = (2, 0)
        case ("Z", "Y") | ([0, 0, 1], [0, 1, 0]):
            shear = (2, 1)

    def _velocity_gradient(x):
        grad_v = np.zeros((3, 3))
        grad_v[shear[0], shear[1]] = 2 * strain_rate
        return grad_v

    return _velocity_gradient
