import numpy as np
from config import config 


def double_integrator_dynamics(
        x: np.ndarray,
        u: np.ndarray,
) -> np.ndarray:
    """Double Integrator Dynamics
    dot x = A @ x + B @ u """
    
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32)

    B = np.array([
        [0, 0],
        [0, 0],
        [(1 / config["drone_mass"]), 0],
        [0, 1 / config["drone_mass"]],
    ], dtype=np.float32)
    
    # Compute the state evolution based on the dynamics
    dynamics = np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u - np.array([0, 9.81*config["drone_mass"]]))
    
    return dynamics
         