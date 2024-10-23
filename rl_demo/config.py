import torch

config = {
    "num_envs": 1_000,
    "device": "cpu",
    "seed": 0,
    "sim_dt": 0.01,
    "policy_dt": 0.01,
    "max_effort": 300,
    "range": {
        "x": (-1000, 1000),
        "y": (0, 915),
    },
    "target_range": {
        "x": (0,1000),
        "y": (0,0)
    },
    "ic_range": {
        "y": (915,915),
        "vx": (44, 44),
        "vy": (0, 0),
    },
    "wind": {
        "wind_mean": (5,2),
        "wind_std" : (1,1)
    },
    "target_distance": 1,
    "target_speed": 5,
    "max_time": 120,
    "max_steps": 1_000_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32],
    ),
    "verbose": 1,
    "rollout_steps": 50,
    "minibatch_size": 1000,
}
