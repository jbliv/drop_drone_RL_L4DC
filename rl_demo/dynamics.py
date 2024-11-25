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
    dims = config["dimensions"]

    if dims == 2:
        relative_velo = x[:, 2] - wind_speed
    else:
        relative_velo = x[:,3:5] - wind_speed

    horizontal_drag = 0.5 * air_density * Cd * area * (relative_velo ** 2)
    horizontal_drag = -np.sign(relative_velo) * horizontal_drag / config["drone_mass"]
    
    if dims == 2:
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.float32)
        B = np.array([
            [0],
            [0],
            [1/config["drone_mass"]],
            [0],
        ], dtype=np.float32)
        
        #dynamics = np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u - np.array([0, 9.81*config["drone_mass"]])) + drag
        #dynamics = np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u - np.array([0, 9.81*config["drone_mass"]])) + drag
        dynamics1 = np.einsum("ij,kj->ki", A, x)
        dynamics2 = np.outer(u,B)
        dynamics2[:,2] += horizontal_drag
        dynamics1[:,3] -= 9.81
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
            [0,0],
            [0,0],
            [0,0],
            [1/config["drone_mass"], 0],
            [0,1/config["drone_mass"]],
            [0,0],

        ], dtype=np.float32)
        

        #dynamics = np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u - np.array([0, 9.81*config["drone_mass"]])) + drag
        dynamics1 = np.einsum("ij,kj->ki", A, x)
        dynamics2 = np.matmul(u,B.T)

        dynamics2[:,4:6] += horizontal_drag

        dynamics1[:,5] -= 9.81
        dynamics = dynamics1 + dynamics2
        
    return dynamics
