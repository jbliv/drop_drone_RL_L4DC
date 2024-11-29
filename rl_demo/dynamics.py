import numpy as np
from config import config


def double_integrator_dynamics(
    x: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    wind_speed: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Double Integrator Dynamics with wind and drag"""
    dims = config["dimensions"]

    relative_v_x = x[:, 3]  # - wind_speed[:, 0]
    relative_v_y = x[:, 4]  # - wind_speed[:, 1]
    relative_v_z = x[:, 5]

    v_ang_xy = np.arctan2(relative_v_y, relative_v_x)
    v_ang_z = np.arctan2(relative_v_z, np.sqrt(relative_v_x**2 + relative_v_y**2))

    # Effective drag coefficients and areas
    effective_Cd = (
        config["Cd_x"] * np.abs(np.cos(v_ang_xy))
        + config["Cd_y"] * np.abs(np.sin(v_ang_xy))
        + config["Cd_z"] * np.abs(np.sin(v_ang_z))
    )
    effective_area = (
        config["area_x"] * np.abs(np.cos(v_ang_xy))
        + config["area_y"] * np.abs(np.sin(v_ang_xy))
        + config["area_z"] * np.abs(np.sin(v_ang_z))
    )

    # Drag magnitudes
    drag_mag_x = (
        0.5 * config["air_density"] * effective_Cd * effective_area * (relative_v_x**2)
    )
    drag_mag_y = (
        0.5 * config["air_density"] * effective_Cd * effective_area * (relative_v_y**2)
    )
    drag_mag_z = (
        0.5 * config["air_density"] * effective_Cd * effective_area * (relative_v_z**2)
    )

    # Drag forces in each direction
    drag_x = -np.sign(relative_v_x) * drag_mag_x * np.abs(np.cos(v_ang_xy))
    drag_y = -np.sign(relative_v_y) * drag_mag_y * np.abs(np.sin(v_ang_xy))
    drag_z = -np.sign(relative_v_z) * drag_mag_z * np.abs(np.sin(v_ang_z)) * 0

    if dims == 2:
        A = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        B = np.array(
            [
                [0, 0],
                [0, 0],
                [(1 / config["drone_mass"]), 0],
                [0, 1 / config["drone_mass"]],
            ],
            dtype=np.float32,
        )

        u_new = np.hstack((u, np.zeros((u.shape[0], 1))))

        # dynamics = np.einsum("ij,kj->ki", A, x) + np.einsum("ij,kj->ki", B, u - np.array([0, 9.81*config["drone_mass"]])) + drag
        dynamics1 = np.einsum("ij,kj->ki", A, x)
        dynamics2 = np.einsum(
            "ij,kj->ki", B, u_new - np.array([0, 9.81 * config["drone_mass"]])
        )
        dynamics = dynamics1 + dynamics2
    elif dims == 3:
        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        B = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [(1 / config["drone_mass"]), 0, 0],
                [0, 1 / config["drone_mass"], 0],
                [0, 0, 1 / config["drone_mass"]],
            ],
            dtype=np.float32,
        )

        parachute = d * (
            9.81 * config["drone_mass"]
            + config["p_gain"]
            * config["drone_mass"]
            * (config["target_speed"] - x[:, dims * 2 - 1])
        )
        u = np.column_stack((u, parachute))

        u[:, 0] += drag_x
        u[:, 1] += drag_y
        u[:, 2] += drag_z * (1 - d)

        dynamics1 = np.einsum("ij,kj->ki", A, x)
        dynamics2 = np.einsum(
            "ij,kj->ki", B, u - np.array([0, 0, 9.81 * config["drone_mass"]])
        )

        dynamics = dynamics1 + dynamics2
    return dynamics
