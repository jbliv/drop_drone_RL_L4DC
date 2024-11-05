from typing import Any, Callable, Dict, Iterable, List,  Optional, Type

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv, VecEnvStepReturn, VecEnvObs, VecEnvIndices
)

from config import config
from dynamics import double_integrator_dynamics
from rewards import double_integrator_rewards
from utils import rk4


class RK4Env(VecEnv):
    metadata = {
        "render_mode": ["human"],
        "render_fps": int(1 / config["policy_dt"])
    }
    actions: np.ndarray

    def __init__(
            self,
            num_envs: int,
            num_obs: int = 8,  # [x, y, dot x, dot y, Tx, Ty, gx, gy]
            num_actions: int = 2,  # [Tx, Ty]
            config: Dict = config,
            dynamics_func: Callable = double_integrator_dynamics,
            rew_func: Callable = double_integrator_rewards,
    ) -> None:
        self.cfg = config
        self.dynamics = dynamics_func
        self.rew_func = rew_func
        self.plotting_tracker = 0
        self.plot_uploaded = False

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_obs,),
            dtype=np.float32,
            seed=self.cfg["seed"],
        )
        self.action_space = gym.spaces.Box(
            low=-self.cfg["max_effort"],
            high=self.cfg["max_effort"],
            shape=(num_actions,),
            dtype=np.float32,
            seed=self.cfg["seed"],
        )

        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._seeds: List[Optional[int]] = [None for _ in range(num_envs)]
        self._options: List[Dict[str, Any]] = [{} for _ in range(num_envs)]

        self.num_envs = num_envs
        self.buf_obs = np.zeros((self.num_envs, num_obs))
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [
            {} for _ in range(self.num_envs)
        ]

        self.sim_dt = self.cfg["sim_dt"]
        self.decimation = int(self.cfg["policy_dt"] / self.cfg["sim_dt"])
        self.target_distance = self.cfg["target_distance"]
        self.target_speed = self.cfg["target_speed"]
        self.rng = np.random.default_rng(self.cfg["seed"])
        self.t = np.zeros((self.num_envs,), dtype=np.float32)
        self.max_time = self.cfg["max_time"]

        self.obs_hist = np.zeros(
            (int(self.max_time / self.cfg["policy_dt"]), num_obs),
            dtype=np.float32,
        )
        self.plot = None  # save plots for env[0]
        self.counter = 0

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for _ in range(self.decimation):
            self.buf_obs[:, 0:4] = rk4(
                self.dynamics,
                self.buf_obs[:, 0:4],
                self.sim_dt,
                u=self.actions
            )
        self.buf_rews = self.rew_func(self.buf_obs, self.actions)
        self.buf_obs[:, 4:6] = self.actions
        self.obs_hist[self.counter] = self.buf_obs[0]
        terminated = \
            (self.buf_obs[:, 0] < self.cfg["env_range"]["x"][0]) | \
            (self.buf_obs[:, 0] > self.cfg["env_range"]["x"][1]) | \
            (self.buf_obs[:, 1] < self.cfg["env_range"]["y"][0]) | \
            (self.buf_obs[:, 1] > self.cfg["env_range"]["y"][1])
        truncated = (
            (np.linalg.norm(
                self.buf_obs[:, 0:2], axis=1
            ) < self.target_distance) # &
            # (np.linalg.norm(
            #     self.buf_obs[:, 2:4], axis=1
            # ) < self.target_speed)
            ) | \
            (self.t > self.max_time)
        self.buf_dones = terminated | truncated
        for idx in range(self.num_envs):
            self.buf_infos[idx]["TimeLimit.truncated"] = \
                truncated[idx] and not terminated[idx]
            if self.buf_dones[idx]:
                self.buf_infos[idx]["terminal_observation"] = self.buf_obs[idx]
                self.buf_infos[idx]["episode"] = {
                    "r": self.buf_rews[idx],
                    "l": self.t[idx]
                }
            self.reset_infos[idx]["success"] = self.buf_dones[idx]
        self.t += self.decimation * self.sim_dt
        self.counter = min(self.counter + 1, self.obs_hist.shape[0] - 1)
        reset_idx = np.argwhere(self.buf_dones).flatten()
        if reset_idx.size > 0:
            self.buf_obs[reset_idx], self.t[reset_idx] = \
                self.reset_idx(reset_idx)

        return (
            np.copy(self.buf_obs),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            self.buf_infos.copy()  # deepcopy(self.buf_infos)
        )

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        self.step_async(actions)
        return self.step_wait()
    
    def uploaded(self, val: bool):
        self.plot_uploaded = val

    def reset_idx(self, indices: VecEnvIndices = None) -> VecEnvObs:
        idx = self._get_indices(indices)
        # save trajectory for env[0]
        
        if 0 in idx and self.counter > 1:
            self.plot = self.render()
            self.plot_uploaded = False
            self.counter = 0
        
        # if 0 in idx and self.counter > 1:
        #     if self.plotting_tracker >= self.cfg["plot_frequency"]:
        #         print("Plotted")
        #         self.plot = self.render()
        #         self.plotting_tracker = 0
        #     else:
        #         print(self.plotting_tracker)
        #         self.plotting_tracker += 1
        #     self.counter = 0
        gx = self.rng.uniform(
            low=self.cfg["goal_ic_range"]["x"][0],
            high=self.cfg["goal_ic_range"]["x"][1],
            size=(len(idx), 1),
        )
        gy = self.rng.uniform(
            low=self.cfg["goal_ic_range"]["y"][0],
            high=self.cfg["goal_ic_range"]["y"][1],
            size=(len(idx), 1),
        )
        vx0 = self.rng.uniform(
            low=self.cfg["drone_ic_range"]["vx"][0],
            high=self.cfg["drone_ic_range"]["vx"][1],
            size=(len(idx), 1),
        )
        vy0 = self.rng.uniform(
            low=self.cfg["drone_ic_range"]["vy"][0],
            high=self.cfg["drone_ic_range"]["vy"][1],
            size=(len(idx), 1),
        )
        y0 = self.rng.uniform(
            low=self.cfg["drone_ic_range"]["y"][0],
            high=self.cfg["drone_ic_range"]["y"][1],
            size=(len(idx), 1),
        )
        x0 = []
        for condition in range(len(idx)):
            # print("here")
            t1 = (vy0[condition][0] + math.sqrt(vy0[condition][0]**2 - 2 * -9.81 * (y0[condition][0] - gy[condition][0]))) / -9.81
            # print(t1)
            t2 = (vy0[condition][0] - math.sqrt(vy0[condition][0]**2 - 2 * -9.81 * (y0[condition][0] - gy[condition][0]))) / -9.81
            # print(t1)
            t = max(t1, t2)
            # print(t)
            x_initial = gx[condition][0] - vx0[condition][0] * t
            # print("VeloX: " vx0[condition][0] ++ vx_initial)
            current_x = self.rng.uniform(
            low=x_initial + self.cfg["drone_ic_range"]["x_range"][0],
            high=x_initial + self.cfg["drone_ic_range"]["x_range"][1])
            x0.append([current_x])

        print("generated IC's")
        T = np.zeros((len(idx), 2), dtype=np.float32)
        obs = np.concatenate((x0, y0, vx0, vy0, T, gx, gy), axis=1)
        t = np.zeros((len(idx),), dtype=np.float32)
        return obs, t

    def reset(self, seed=None, options=None) -> VecEnvObs:
        idx = self._get_indices(None)
        self.buf_obs, self.t = self.reset_idx(idx)
        for idx in range(self.num_envs):
            self.reset_infos[idx] = {}
        return self.buf_obs

    def render(self) -> matplotlib.figure.Figure:
        if 1 == 2:
            pass
        else:
            obs_plot = self.obs_hist[:self.counter]

            # Calculate the magnitude of the combined velocity vector
            velocity_magnitude = np.sqrt(obs_plot[:, 2]**2 + obs_plot[:, 3]**2)

            # Create a color map based on velocity magnitude
            norm = plt.Normalize(velocity_magnitude.min(), velocity_magnitude.max())
            colors = plt.cm.viridis(norm(velocity_magnitude))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Subplot 1: Trajectory with velocity magnitude gradient
            for i in range(1, len(obs_plot)):
                ax1.plot(obs_plot[i-1:i+1, 0], obs_plot[i-1:i+1, 1], color=colors[i-1], linewidth=2)

            # Plot initial and goal points
            ax1.scatter(obs_plot[0, 0], obs_plot[0, 1], color='red', s=100, label='Initial Point')
            ax1.scatter(obs_plot[0, 6], obs_plot[0, 7], color='blue', s=100, label='Goal Point')

            # Add colorbar for velocity magnitude
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax1, label='Velocity Magnitude')

            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.legend()
            ax1.set_title('Trajectory with Velocity Magnitude Gradient')

            # Define arrow locations and properties
            #wind_speed = 5.0  
            #wind_direction = np.array([1, 0])  
            #arrow_locations = [(200, 800), (500, 600), (100, 400), (300, 200)]  

            #for arrow_x, arrow_y in arrow_locations:
               # ax1.arrow(
                   # arrow_x, arrow_y,
                   # wind_direction[0] * wind_speed * 0.1,
                   # wind_direction[1] * wind_speed * 0.1,
                   # head_width=5, head_length=40, fc="black", ec="black"
              #  )

            # Time axis for thrust plot
#             time = np.linspace(0, len(obs_plot[:, 1]) * self.sim_dt, len(obs_plot[:, 1]))


            time = np.linspace(0, len(obs_plot[:, 1]) * self.sim_dt, len(obs_plot[:, 1]))

            # Subplot 2: Thrust vs Y-location
            ax2.plot(time, obs_plot[:, 4], label='X-Thrust', color='orange')
            ax2.plot(time, obs_plot[:, 5], label='Y-Thrust', color='purple')

            ax2.set_xlabel('Time')
            ax2.set_ylabel('Thrust Value')
            ax2.legend()
            ax2.set_title('Thrust vs Time')

            plt.tight_layout()
            print("Plot made")
        return fig


    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        return indices

    # Required to subclass stable_baselines3.common.vec_env.base_vec_env.VecEnv
    # Do not change below:

    def close(self) -> None:
        return

    def env_is_wrapped(
            self,
            wrapper_class: Type[gym.Wrapper],
            indices: VecEnvIndices = None
    ) -> List[bool]:
        idx = self._get_indices(indices)
        return [False] * len(idx)

    def env_method(
            self,
            method_name: str,
            *method_args,
            indices: VecEnvIndices = None,
            **method_kwargs
    ) -> List[Any]:
        raise NotImplementedError

    def get_attr(
            self,
            attr_name: str,
            indices: VecEnvIndices = None
    ) -> List[Any]:
        raise NotImplementedError

    def set_attr(
            self,
            attr_name: str,
            indices: VecEnvIndices = None
    ) -> List[Any]:
        raise NotImplementedError


if __name__ == "__main__":
    import time
    n = 10_000
    T = 10
    env = RK4Env(n, 8, 2, config)
    u = np.array([[0]*n, [0.]*n], dtype=np.float32).T
    now = time.time()
    for _ in range(T):
        obs, rew, done, info = env.step(u)
    print(f"{int(n*T/(time.time() - now)):_d} steps/second")