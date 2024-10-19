import numpy as np


def double_integrator_rewards(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Double Integrator Reward Function
    State:
    x: [x, y, dot x, dot y, Tx, Ty, gx, gy] -> [num_envs, 8]
    Action:
    u: [Tx, Ty] -> [num_envs, 2]
    """
    # task rewards
    pose_error = np.linalg.norm(x[:, 6:8] - x[:, 0:2], axis=1)
    tracking_std = 3000
    goal_tracking = 10*np.exp(-pose_error**2/tracking_std)
    goal_reached = np.where(pose_error < 1, 1, 0)
    #wall_hit = np.where((x[:, 0] == 300) | (x[:,0] == -300), 1, 0)

    # regularizing rewards
    effort_penalty = -np.linalg.norm(u, axis=1) ** 2
    # action_rate = -np.linalg.norm(u - x[:, 4:6], axis=1)

    return 10 * goal_reached + goal_tracking + .5 * effort_penalty