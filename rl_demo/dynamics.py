import numpy as np


def double_integrator_dynamics(
        x: np.ndarray,
        u: np.ndarray,
        a: np.ndarray,
) -> np.ndarray:
    """Double Integrator Dynamics
    dot x = A @ x + B @ u """
    
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32)
    C = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32)

    B = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
    ])
    target_speed = -5
    # Compute the state evolution based on the dynamics



    dynamics = np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u - np.array([0, 9.81]))  # Apply system dynamics A @ x
    dynamics[:,3] = np.where(a==1,0,dynamics[:,3])
    return dynamics
         