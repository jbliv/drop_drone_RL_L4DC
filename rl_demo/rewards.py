import numpy as np
import pdb
from config import config
def double_integrator_rewards(ID, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Double Integrator Reward Function
    State:
    2D
        x: [x, y, dot x, dot y, Tx, Ty, gx, gy, deploy] -> [num_envs, 9]
        Action:
        u: [Tx, deploy] -> [num_envs, 2]

    3D
        x: [x, y, z, dot x, dot y, dot z, Tx, Ty, gx, gy, gy, deploy] -> [num_envs, 12]
        Action:
        u: [Tx, Ty, deploy] -> [num_envs, 3]

    """
    # task rewards
    dims = config["dimensions"]
    pose_error = np.linalg.norm(x[:, dims * 3 - 1:dims * 4 - 1] - x[:, 0:dims], axis=1)
    tracking_std = ID # Initial Distance of agent from target


    #constant 2.5 reduces variance(Gaussian lingo) so that rewards aren't given initially
    goal_tracking = tracking_std*np.exp(-(2.5*pose_error/tracking_std)**2)

    #Potential Additions
    # effort_penalty = -np.linalg.norm(u, axis=1)**2
    # action_rate = -np.linalg.norm(u - x[:, 4:6], axis=1)
    # goal_reached = np.where(pose_error < 5, 1, 0)
    # wall_hit = np.where(x[:,0] < config["env_range"][x][0] | x[:,0] > config["env_range"][x][1])

    reward = goal_tracking
    return reward 
    