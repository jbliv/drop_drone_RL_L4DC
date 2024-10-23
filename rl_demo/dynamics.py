import numpy as np


def double_integrator_dynamics(
        x: np.ndarray,
        u: np.ndarray,
        **kwargs,
) -> np.ndarray:
    """Double Integrator Dynamics
    dot x = A @ x + B @ u
    """
    """Drag Force equation is given by:
    Fd = .5*ro*Cd*A*v^2"""
    ro = 1.2 #approximate
    surf = 3.14 # given a sphere with radius of 0.5 m 
    Cd = 0.5 #sphere

    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32)
    B = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
    ], dtype=np.float32)

    # drag_force = np.array([
    #     -0.5*ro*surf*Cd*x[:,2]**2,
    #     0.5*ro*surf*Cd*x[:,3]**2
    # ])
    return np.einsum("ij,kj->ki",A, x) + np.einsum("ij,kj->ki",B, u - np.array([0, 9.8])) 