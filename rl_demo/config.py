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
        "y": (0, 1000),
    },
    "goal_ic_range": {
        "x": (125, 750),
        "y": (0, 0),
    },
    "drone_ic_range": {
        "x": (-950, -500),
        "y": (915, 915),
        "vx": (45, 45),
        "vy": (-1, -1),
    },
    "target_distance": 1,
    "target_speed": -5,
    "max_time": 50, # End of episode
    "max_steps": 1_000_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32],
    ),
    "verbose": 1,
    "rollout_steps": 50, # how man time steps per rollout before training
    "minibatch_size": 2500,
    "plot_frequency": 0,
}
