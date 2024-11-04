import torch

config = {
    "num_envs": 1_000,
    "device": "cpu",
    "seed": 0,
    "sim_dt": 0.01,
    "policy_dt": 0.01,
    "max_effort": 100,
    "env_range": {
        "x": (-1000, 1000),
        "y": (0, 915),
    },
    "goal_ic_range": {
        "x": (0, 1000),
        "y": (0, 0),
    },
    "drone_ic_range": {
        "x": (-1000, 0),
        "y": (915, 915),
        "vx": (45, 45),
        "vy": (0, 0),
    },
    "target_distance": 1,
    "target_speed": -5,
    "max_time": 300, # End of episode
    "max_steps": 1_000_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32],
    ),
    "verbose": 1,
    "rollout_steps": 100, # how man time steps per rollout before training
    "minibatch_size": 1000,
    "plot_frequency": 0,
}
