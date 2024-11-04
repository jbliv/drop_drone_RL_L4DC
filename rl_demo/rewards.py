import numpy as np
import pdb
def double_integrator_rewards(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Double Integrator Reward Function
    State:
    x: [x, y, dot x, dot y, Tx, Ty, gx, gy] -> [num_envs, 8]
    Action:
    u: [Tx, Ty, deploy] -> [num_envs, 2]
    """
    # task rewards
    
    pose_error = np.linalg.norm(x[:, 7:9] - x[:, 0:2], axis = 1)
    tracking_std = 2000
    
    goal_tracking = 500 * np.exp(-(1.5*pose_error/tracking_std)**2)
    goal_reached = np.where(pose_error < 1, 1, 0)
    # ground_hit = np.where((x[:, 1] == 0) & (pose_error > 5), -1, 0)
    wall_hit = np.where((x[:,0] == -1000) | (x[:,0] == 1000), -1, 0)

    #TODO: change the goal tracking based on the intial distance from the target

    # regularizing rewards
    effort_penalty = -np.linalg.norm(u, axis=1) ** 2
    # action_rate = -np.linalg.norm(u - x[:, 4:6], axis=1)
    #have net positive rewards
    return 0.1 * goal_tracking + 10 * goal_reached + wall_hit + 0.01 * effort_penalty
    



