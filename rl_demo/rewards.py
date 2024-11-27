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
    goal_tracking = tracking_std*np.exp(-(2.5*pose_error/tracking_std)**2)
    reward = goal_tracking
    return reward 

# closer_std = ID / 100
#     goal_closer = np.exp(-np.sqrt(np.abs(pose_error))/closer_std)

#     goal_kinda = np.where(pose_error < 100, 1, 0) * np.where(x[:, dims] < 10, 1, 0) * (100 - pose_error) / 100
#     goal_almost = np.where(pose_error < 10, 1, 0) * np.where(x[:, dims] < 10, 1, 0) * (10 - pose_error) / 10
#     goal_reached = np.where(pose_error < target_range, 1, 0)

# =======
#     deploy_bad = np.where(x[:, dims] > 75, 1, 0) * np.where(x[:, -1], 1, 0)
#     deploy_good = np.where(x[:, dims] < 50, 1, 0) * np.where(x[:, -1], 1, 0)

#     # regularizing rewards
#     effort_penalty = -np.linalg.norm(u[:, 0: dims], axis=1) ** 2
#     action_rate = -np.linalg.norm(u[:, 0: dims] - x[:, dims * 2:dims * 3], axis=1)
    
#     reward = (10 * goal_closer + 5 * goal_tracking + 
#               100 * goal_reached + 25 * goal_almost + 10 * goal_kinda + 
#               100 * deploy_good - 2500 * deploy_bad + 
#               0.001 * effort_penalty + 0.0001 * action_rate) # + np.where(pose_error > ID, -1000, 0)
#     return reward


# >>>>>>> main
