import numpy as np
from config import config

def double_integrator_dynamics(
        x: np.ndarray,
        u: np.ndarray,
        Cd: float = 0.5, 
        area: float = 0.1,  
        air_density: float = 1.225,  
        wind_speed: float = 5.0,  
        **kwargs,
) -> np.ndarray:
    """Double Integrator Dynamics with wind and drag"""
    relative_v_x = x[:, 2] - wind_speed
    drag_x = 0.5 * air_density * Cd * area * (relative_v_x ** 2)
    drag_x = -np.sign(relative_v_x) * drag_x / config["drone_mass"]
    dims = config["dimensions"]
    # drag = np.zeros_like(x))
    # drag[:,2] = drag_x  

    # print("U size")
    # print(u.size)
    # print(u)
    if dims == 2:
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
        

        #dynamics = np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u - np.array([0, 9.81*config["drone_mass"]])) + drag
        dynamics1 = np.einsum("ij,kj->ki", A, x)
        dynamics2 = np.einsum("ij,kj->ki", B, u - np.array([0, 9.81*config["drone_mass"]]))
        dynamics2[:,2] += drag_x
        dynamics = dynamics1 + dynamics2
    elif dims == 3:
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)

        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [(1 / config["drone_mass"]), 0, 0],
            [0, 1 / config["drone_mass"], 0],
            [0, 0, 1 / config["drone_mass"]],
        ], dtype=np.float32)
        

        #dynamics = np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u - np.array([0, 9.81*config["drone_mass"]])) + drag
        dynamics1 = np.einsum("ij,kj->ki", A, x)
        dynamics2 = np.einsum("ij,kj->ki", B, u - np.array([0, 0, 9.81*config["drone_mass"]]))
        # dynamics2[:,2] += drag_x
        dynamics = dynamics1 + dynamics2
        

    return dynamics
