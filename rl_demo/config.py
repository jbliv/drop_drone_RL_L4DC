import torch

config = {

    "num_envs": 5_0,
    "device": "cuda",
    "seed": 0,
    "sim_dt": 0.01,
    "policy_dt": 0.1,
    "max_effort": 100,
    "dimensions": 3, 
    "env_range": {
        "x": (-100000, 100000),
        "y": (-100000, 100000),
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
        "xy_range": (-5.0, -5.0),
        "z": (100, 100),
        "vx": (4.5, 4.5),
        "vy": (4.5, 4.5),
        "vz": (-6, -6),
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
    "rollout_steps": 200, # how man time steps per rollout before training
    "minibatch_size": 50000,
    "plot_frequency": 0,
    "gif_steps/frame": 10,
    "gif_speed": 1

}
