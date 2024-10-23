import torch

config = {
    "num_envs": 1_000,
    "device": "cuda",
    "seed": 0,
    "sim_dt": 0.01,
    "policy_dt": 0.01,
    "max_effort": 100,
    "env_range": {
        "x": (-10000, 10000),
        "y": (0, 5000),
    },
    "goal_ic_range": {
        "x": (-20, 20),
        "y": (0, 100),
    },
    "drone_ic_range": {
        "x": (-2000, 2000),
        "x_range": (-20, 20),
        "y": (900, 1000),
        "vx": (-45, 45),
        "vy": (0, 0),
    },
    "target_distance": 1,
    "target_speed": 0.0,
    "max_time": 50,
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
