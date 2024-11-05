import numpy as np
from config import config

target_range = config["target_distance"]


def double_integrator_rewards(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Double Integrator Reward Function
    State:
    x: [x, y, dot x, dot y, Tx, Ty, gx, gy] -> [num_envs, 8]
    Action:
    u: [Tx, Ty] -> [num_envs, 2]
    """
    # task rewards
    pose_error = np.linalg.norm(x[:, 6:8] - x[:, 0:2], axis=1)
    tracking_std = 100
    goal_tracking = np.exp(-pose_error ** 2 / tracking_std)

    closer_std = 10
    goal_closer = np.exp(-np.sqrt(np.abs(pose_error))/closer_std)

    goal_reached = np.where(pose_error < target_range, 1, 0)

    # regularizing rewards
    effort_penalty = -np.linalg.norm(u, axis=1) ** 2
    action_rate = -np.linalg.norm(u - x[:, 4:6], axis=1)
    
    reward = 20 * goal_closer + 10 * goal_tracking + 100 * goal_reached + 0.0001 * effort_penalty + 0.0001 * action_rate
    return reward