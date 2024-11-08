import torch

config = {

    "num_envs": 1_000,
    "device": "cuda",
    "seed": 0,
    "sim_dt": 0.01,
    "policy_dt": 0.1,
    "max_effort": 100,
    "dimensions": 3, 
    "env_range": {
        "x": (-10000, 10000),
        "y": (-10000, 10000),
        "z": (0, 1500),
    },
    "goal_ic_range": {
        "x": (0, 0),
        "y": (0, 0),
        "z": (0, 0),
    },
    "drone_ic_range": {
        "x": (-2000, 2000),
        "y": (-2000, 2000),
        "xy_range": (-50, -50),
        "z": (1000, 1000),
        "vx": (45, 45),
        "vy": (45, 45),
        "vz": (0, 0),
    },

    "drone_mass": 2.3,
    "target_distance": 5,
    "target_speed": -5,
    "max_time": 100,

    "max_steps": 1_000_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32, 32],
    ),
    "verbose": 1,
    "rollout_steps": 1500, # how man time steps per rollout before training
    "minibatch_size": 5000,
    "plot_frequency": 0,
    "gif_steps/frame": 10,
    "gif_speed": 4

}

