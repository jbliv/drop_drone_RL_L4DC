import torch

config = {
    # Hyperparameters and policy controls
    "num_envs": 1_000,
    "device": "cuda",
    "seed": 0,
    "rollout_steps": 150,  # how man time steps per rollout before training
    "minibatch_size": 5000,
    "max_steps": 1_000_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32, 32],
    ),
    "verbose": 1,
    # Simulation and env variables
    "sim_dt": 0.01,
    "policy_dt": 0.1,
    "dimensions": 3,
    "max_time": 100,
    "env_range": {
        "x": (-100000, 100000),
        "y": (-100000, 100000),
        "z": (0, 1500),
    },
    # Drone Variables
    "drone_mass": 2.3,
    "max_effort": 100,
    "p_gain": 1.5,  # Proportional gain for parachute modeling. In theory anything higher than drone mass is an unstable system
    "drone_ic_range": {
        "x": (-2000, 2000),
        "y": (-2000, 2000),
        "xy_range": (-50, -50),
        "z": (1000, 1000),
        # "vx": (0, 0),
        # "vy": (0, 0),
        # "vz": (0, 0),
        "vx": (45, 45),
        "vy": (45, 45),
        "vz": (-5, -5),
    },
    # Goal Variables
    "goal_ic_range": {
        "x": (0, 0),
        "y": (0, 0),
        "z": (0, 0),
    },
    "target_distance": 5,
    "target_speed": -5,
    # Wind Variables
    "max_wind": 5.0,
    "Cd_x": 0.8,
    "Cd_y": 0.8,
    "Cd_z": 0.35,
    "area_x": 0.5,
    "area_y": 0.5,
    "area_z": 0.3,
    "air_density": 1.225,
    # Plotting Variables
    "plot_frequency": 0,
    "gif_steps/frame": 10,
    "gif_speed": 4,
}
