import numpy as np
import pdb
from config import config
def double_integrator_rewards(ID, x: np.ndarray, u: np.ndarray) -> np.ndarray:

target_range = config["target_distance"]


def double_integrator_rewards(x: np.ndarray, u: np.ndarray) -> np.ndarray:

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

    # task rewards
    pose_error = np.linalg.norm(x[:, dims * 3:dims * 4] - x[:, 0:dims], axis=1)
    tracking_std = ID
    goal_tracking = np.exp(-pose_error ** 2 / tracking_std)

    closer_std = 10
    goal_closer = np.exp(-np.sqrt(np.abs(pose_error))/closer_std)

    goal_reached = np.where(pose_error < target_range, 1, 0)

    # regularizing rewards
    effort_penalty = -np.linalg.norm(u, axis=1) ** 2
    action_rate = -np.linalg.norm(u - x[:, dims * 2:dims * 3], axis=1)
    
    reward = 20 * goal_closer + 10 * goal_tracking + 100 * goal_reached + 0.0001 * effort_penalty + 0.0001 * action_rate
    return reward


