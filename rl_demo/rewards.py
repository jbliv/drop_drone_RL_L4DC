import numpy as np
import pdb
from config import config


def double_integrator_rewards(
    ID, x: np.ndarray, old_x: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Double Integrator Reward Function
    2-Dimensions
    State:
    x: [x, y, dot x, dot y, Tx, Ty, gx, gy, D] -> [num_envs, 9]
    Action:
    u: [Tx, Ty, deploy] -> [num_envs, 3]

    3-Dimensions
    State:
    x: [x, y, z, dot x, dot y, dot z, Tx, Ty, Tz, gx, gy, gz, D] -> [num_envs, 13]
    Action:
    u: [Tx, Ty, Tz, deploy] -> [num_envs, 4]
    """

    dims = config["dimensions"]
    target_range = config["target_distance"]

    # # task rewards
    # pose_error = np.linalg.norm(x[:, dims * 3:dims * 4] - x[:, 0:dims], axis=1)
    # tracking_std = ID
    # goal_tracking = np.exp(-pose_error ** 2 / tracking_std)

    # closer_std = ID / 100
    # goal_closer = np.exp(-np.sqrt(np.abs(pose_error))/closer_std)

    # goal_kinda = np.where(pose_error < 1000, 1, 0) * np.where(x[:, dims] < 10, 1, 0) * (1000 - pose_error) / 1000
    # goal_almost = np.where(pose_error < 100, 1, 0) * np.where(x[:, dims] < 100, 1, 0) * (100 - pose_error) / 100
    # goal_reached = np.where(pose_error < target_range, 1, 0)

    # # regularizing rewards
    # effort_penalty = -np.linalg.norm(u, axis=1) ** 2
    # action_rate = -np.linalg.norm(u - x[:, dims * 2:dims * 3], axis=1)

    # reward = (10 * goal_closer + 5 * goal_tracking +
    #           100 * goal_reached + 25 * goal_almost + 10 * goal_kinda +
    #           0.0001 * effort_penalty + 0.0001 * action_rate)
    old_dist = np.linalg.norm(old_x[:, 0 : dims - 1], axis=1)
    new_dist = np.linalg.norm(x[:, 0 : dims - 1], axis=1)
    goal_closer = np.where(old_dist > new_dist, old_dist - new_dist, 0)
    # goal_closer = np.linalg.norm(old_x[:, 0:dims], axis=1)  / np.linalg.norm(old_x[:, 0:dims] - x[:, 0:dims], axis=1)
    pose_error = np.linalg.norm(x[:, dims * 2 : dims * 3] - x[:, 0:dims], axis=1)
    goal_reached = np.where(pose_error < target_range, 1, 0)
    parachute_early = np.where(x[:, dims - 1] > 500, 1 - x[:, -1], 1)
    parachute_late = np.where(x[:, dims - 1] < 400, x[:, -1] - 1, 1)

    # regularizing rewards
    effort_penalty = -(np.linalg.norm(u, axis=1))
    action_rate = -np.linalg.norm(u - x[:, dims * 3 : dims * 4 - 1], axis=1)

    reward = (
        5 * goal_reached
        + 0.1 * goal_closer * parachute_early * parachute_late
        + 0.0001 * effort_penalty
        + 0.0001 * action_rate
    )

    return reward
