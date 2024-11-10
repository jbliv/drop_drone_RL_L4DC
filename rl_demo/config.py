import torch

config = {
    "num_envs": 1000,
    "device": "cpu",
    "seed": 0,
    "sim_dt": 0.01,
    "policy_dt": 0.1,
    "max_effort": 220,
    "dimensions": 3,
    "env_range": {
        "x": (-500,500),
        "y": (-500, 500),
        "z": (0,1500)
    },
    "goal_ic_range": {
        "x": (300,500),
        "y": (300, 500),
        "z": (0,0)
    },
    "drone_ic_range": {
        "x_range": (0, 100),
        "xy_range": (-50, -50),
        "x": (-200,0),
        "y": (-200,0),
        "vx": (45, 45),
        "vy": (45, 45),
        "vz":(0,0),
        "z": (750,1000)
    },
    "drone_mass": 23,
    "target_distance": 1,
    "target_speed": -5,
    "max_time": 100, # End of episode
    "max_steps": 1_000_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32, 32],
    ),
    "verbose": 1,
    "rollout_steps": 200, # how man time steps per rollout before training
    "minibatch_size": 2000,
    "plot_frequency": 0,
    "gif_steps/frame": 10,
    "gif_speed": 4
}