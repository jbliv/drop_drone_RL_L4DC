import torch

config = {
    "num_envs": 1000,
    "device": "cpu",
    "seed": 0,
    "sim_dt": 0.01,
    "policy_dt": 0.1,
    "max_effort": 10,
    "env_range": {
        "x": (-500, 500),
        "y": (0, 1500),
    },
    "goal_ic_range": {
        "x": (450,500),
        "y": (0, 0),
    },
    "drone_ic_range": {
        "x_range": (0, 0),
        "x": (-500,500),
        "y": (500,500),
        "vx": (0, 0),
        "vy": (0, 0),
    },
    "drone_mass": 1,
    "target_distance": 5,
    "target_speed": -5,
    "max_time": 100, # End of episode
    "max_steps": 1_000_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32],
    ),
    "verbose": 1,
    "rollout_steps": 100, # how man time steps per rollout before training
    "minibatch_size": 2500,
    "plot_frequency": 0,
}
