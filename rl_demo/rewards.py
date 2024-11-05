import numpy as np
import pdb
from config import config as cfg
def double_integrator_rewards(ID, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Double Integrator Reward Function
    State:
    x: [x, y, dot x, dot y, Tx, Ty,D, gx, gy] -> [num_envs, 8]
    Action:
    u: [Tx, Ty, deploy] -> [num_envs, 2]
    """
    # task rewards
    
    pose_error = x[:, 7] - x[:, 0]
    tracking_std = ID

    
    goal_tracking = 500 * np.exp(-(3*pose_error/tracking_std)**2)
    goal_reached = np.where(pose_error < 1, 10, 0)
    # ground_hit = np.where((x[:, 1] == 0) & (pose_error > 5), -1, 0)
    wall_hit = np.where((x[:,0] == cfg["env_range"]["x"][0]) | (x[:,0] == cfg["env_range"]["x"][1]), -1, 0)

    #TODO: change the goal tracking based on the intial distance from the target

    # regularizing rewards
    effort_penalty = -np.linalg.norm(u, axis=1) ** 2
    # action_rate = -np.linalg.norm(u - x[:, 4:6], axis=1)
    #have net positive rewards
    return goal_tracking + wall_hit
    



