from typing import Any, Callable, Dict, Iterable, List, Optional, Type

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import pdb
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvStepReturn,
    VecEnvObs,
    VecEnvIndices,
)
from config import config
from dynamics import double_integrator_dynamics
from rewards import double_integrator_rewards
from utils import rk4


class RK4Env(VecEnv):
    metadata = {"render_mode": ["human"], "render_fps": int(1 / config["policy_dt"])}
    actions: np.ndarray

    def __init__(
        self,
        num_envs: int,
        test: bool = False,
        gif: bool = False,
        num_obs: int = config["dimensions"]
        * 4,  # [x, (y), z, dot x, (dot y), dot z, gx, (gy), gz, Tx, (Ty), deployed]
        num_actions_continuous: int = config["dimensions"] - 1,  # [Tx, Ty, (Tz)]
        num_actions_disc: int = 1,  # [0,1] 1 for slow down to 5 m/s, 0 for continue normal tracking
        config: Dict = config,
        dynamics_func: Callable = double_integrator_dynamics,
        rew_func: Callable = double_integrator_rewards,
    ) -> None:
        self.cfg = config
        self.dims = self.cfg["dimensions"]
        self.dynamics = dynamics_func
        self.rew_func = rew_func
        self.plotting_tracker = 0
        self.plot_uploaded = False
        self.test = test
        self.gif = gif

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_obs,),
            dtype=np.float32,
            seed=self.cfg["seed"],
        )

        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low=-self.cfg["max_effort"],
                    high=self.cfg["max_effort"],
                    shape=(num_actions_continuous,),
                    dtype=np.float32,
                ),
                gym.spaces.Discrete(num_actions_disc),
            )
        )

        self.action_space = gym.spaces.flatten_space(self.action_space)
        self.reset_infos: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._seeds: List[Optional[int]] = [None for _ in range(num_envs)]
        self._options: List[Dict[str, Any]] = [{} for _ in range(num_envs)]

        self.num_envs = num_envs

        self.flip_discrete = np.zeros((self.num_envs), dtype=bool)
        self.discrete_action = np.zeros((self.num_envs), dtype=np.float32)
        self.wind = np.zeros((self.num_envs), dtype=np.float32)

        self.buf_obs = np.zeros((self.num_envs, num_obs))
        self.obs_prev = np.zeros((self.num_envs, num_obs))
        self.dt = np.zeros((self.num_envs), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

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

    def step_async(self, actions: np.ndarray = None) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:

        self.obs_prev = np.copy(self.buf_obs)
        self.continuous_action = self.actions[:, 0 : self.dims - 1]
        semi_discrete_action = np.where(self.actions[:, self.dims - 1] > 0.5, 1, 0)
        discrete_action = np.where(
            self.buf_obs[:, self.dims - 1] > self.cfg["parachute_max"],
            0,
            semi_discrete_action,
        )
        self.discrete_action = discrete_action
        self.buf_obs[:, -1] = self.discrete_action
        self.buf_obs[:, -1] = np.where(
            self.obs_prev[:, -1] == 1, 1, self.buf_obs[:, -1]
        )

        self.continuous_action[:, :] = np.where(
            self.buf_obs[:, -1][:, np.newaxis] == 1,
            self.continuous_action[:, :],
            [0, 0],
        )

        for _ in range(self.decimation):
            self.buf_obs[:, 0 : self.dims * 2] = rk4(
                self.dynamics,
                self.buf_obs[:, 0 : self.dims * 2],
                self.sim_dt,
                u=self.continuous_action,
                d=self.buf_obs[:, -1],
                wind_speed=self.wind,
            )
        filler = 0  # self.initial_distance not being declared for some reason
        self.buf_rews = self.rew_func(
            filler, self.buf_obs, self.obs_prev, self.continuous_action
        )
        self.buf_obs[:, self.dims * 3 : self.dims * 4 - 1] = self.continuous_action
        self.obs_hist[self.counter] = self.buf_obs[0]
        if self.dims == 2:
            terminated = (
                (self.buf_obs[:, 0] < self.cfg["env_range"]["x"][0])
                | (self.buf_obs[:, 0] > self.cfg["env_range"]["x"][1])
                | (self.buf_obs[:, 1] < self.cfg["env_range"]["z"][0])
                | (self.buf_obs[:, 1] > self.cfg["env_range"]["z"][1])
            )
            truncated = (
                (np.linalg.norm(self.buf_obs[:, 0:2], axis=1) < self.target_distance)
                & (np.linalg.norm(self.buf_obs[:, 2:4], axis=1) < self.target_speed)
            ) | (self.t > self.max_time)
        elif self.dims == 3:
            terminated = (
                (self.buf_obs[:, 0] < self.cfg["env_range"]["x"][0])
                | (self.buf_obs[:, 0] > self.cfg["env_range"]["x"][1])
                | (self.buf_obs[:, 1] < self.cfg["env_range"]["y"][0])
                | (self.buf_obs[:, 1] > self.cfg["env_range"]["y"][1])
                | (self.buf_obs[:, 2] < self.cfg["env_range"]["z"][0])
                | (self.buf_obs[:, 2] > self.cfg["env_range"]["z"][1])
            )
            truncated = (
                (
                    np.linalg.norm(self.buf_obs[:, 0 : self.dims], axis=1)
                    < self.target_distance
                )
                & (
                    np.linalg.norm(self.buf_obs[:, self.dims : self.dims * 2], axis=1)
                    < self.target_speed
                )
            ) | (self.t > self.max_time)
        self.buf_dones = terminated | truncated
        for idx in range(self.num_envs):
            self.buf_infos[idx]["TimeLimit.truncated"] = (
                truncated[idx] and not terminated[idx]
            )
            if self.buf_dones[idx]:
                self.buf_infos[idx]["terminal_observation"] = self.buf_obs[idx]
                self.buf_infos[idx]["episode"] = {
                    "r": self.buf_rews[idx],
                    "l": self.t[idx],
                }
            self.reset_infos[idx]["success"] = self.buf_dones[idx]
        self.t += self.decimation * self.sim_dt
        self.counter = min(self.counter + 1, self.obs_hist.shape[0] - 1)
        reset_idx = np.argwhere(self.buf_dones).flatten()
        if reset_idx.size > 0:
            self.buf_obs[reset_idx], self.t[reset_idx] = self.reset_idx(reset_idx)

        return (
            np.copy(self.buf_obs),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            self.buf_infos.copy(),  # deepcopy(self.buf_infos)
        )

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        self.step_async(actions)
        return self.step_wait()

    def uploaded(self, val: bool):
        self.plot_uploaded = val
        if self.plot_uploaded:
            plt.close()

    def reset_idx(self, indices: VecEnvIndices = None) -> VecEnvObs:

        idx = self._get_indices(indices)

        if 0 in idx and self.counter > 1:
            self.plot = self.render()
            self.plot_uploaded = False
            self.counter = 0

        self.wind = self.rng.uniform(
            low=-self.cfg["max_wind"],
            high=self.cfg["max_wind"],
            size=(len(idx), 1),
        )

        gx = self.rng.uniform(
            low=self.cfg["goal_ic_range"]["x"][0],
            high=self.cfg["goal_ic_range"]["x"][1],
            size=(len(idx), 1),
        )
        gz = self.rng.uniform(
            low=self.cfg["goal_ic_range"]["z"][0],
            high=self.cfg["goal_ic_range"]["z"][1],
            size=(len(idx), 1),
        )
        vx0 = self.rng.uniform(
            low=self.cfg["drone_ic_range"]["vx"][0],
            high=self.cfg["drone_ic_range"]["vx"][1],
            size=(len(idx), 1),
        )
        vz0 = self.rng.uniform(
            low=self.cfg["drone_ic_range"]["vz"][0],
            high=self.cfg["drone_ic_range"]["vz"][1],
            size=(len(idx), 1),
        )
        z0 = self.rng.uniform(
            low=self.cfg["drone_ic_range"]["z"][0],
            high=self.cfg["drone_ic_range"]["z"][1],
            size=(len(idx), 1),
        )
        x0 = []
        for condition in range(len(idx)):
            t1 = (
                vz0[condition][0]
                + math.sqrt(
                    vz0[condition][0] ** 2
                    - 2 * -9.81 * (z0[condition][0] - gz[condition][0])
                )
            ) / -9.81
            t2 = (
                vz0[condition][0]
                - math.sqrt(
                    vz0[condition][0] ** 2
                    - 2 * -9.81 * (z0[condition][0] - gz[condition][0])
                )
            ) / -9.81
            t = max(t1, t2)
            x_initial = gx[condition][0] - vx0[condition][0] * t
            current_x = self.rng.uniform(
                low=x_initial + self.cfg["drone_ic_range"]["xy_range"][0],
                high=x_initial + self.cfg["drone_ic_range"]["xy_range"][1],
            )
            x0.append([current_x])

        if self.dims == 3:
            vy0 = self.rng.uniform(
                low=self.cfg["drone_ic_range"]["vy"][0],
                high=self.cfg["drone_ic_range"]["vy"][1],
                size=(len(idx), 1),
            )
            gy = self.rng.uniform(
                low=self.cfg["goal_ic_range"]["y"][0],
                high=self.cfg["goal_ic_range"]["y"][1],
                size=(len(idx), 1),
            )
            y0 = []
            for condition in range(len(idx)):
                t1 = (
                    vz0[condition][0]
                    + math.sqrt(
                        vz0[condition][0] ** 2
                        - 2 * -9.81 * (z0[condition][0] - gz[condition][0])
                    )
                ) / -9.81
                t2 = (
                    vz0[condition][0]
                    - math.sqrt(
                        vz0[condition][0] ** 2
                        - 2 * -9.81 * (z0[condition][0] - gz[condition][0])
                    )
                ) / -9.81
                t = max(t1, t2)
                y_initial = gy[condition][0] - vy0[condition][0] * t
                current_y = self.rng.uniform(
                    low=y_initial + self.cfg["drone_ic_range"]["xy_range"][0],
                    high=y_initial + self.cfg["drone_ic_range"]["xy_range"][1],
                )
                y0.append([current_y])

        T = np.zeros((len(idx), self.dims - 1), dtype=np.float32)
        D = np.zeros((len(idx), 1), dtype=int)
        if self.dims == 2:
            obs = np.concatenate((x0, z0, vx0, vz0, gx, gz, T, D), axis=1)
        elif self.dims == 3:
            obs = np.concatenate((x0, y0, z0, vx0, vy0, vz0, gx, gy, gz, T, D), axis=1)
        t = np.zeros((len(idx),), dtype=np.float32)

        return obs, t

    def reset(self, seed=None, options=None) -> VecEnvObs:

        idx = self._get_indices(None)
        self.buf_obs, self.t = self.reset_idx(idx)
        print("declared")
        self.initial_distance = np.linalg.norm(
            self.buf_obs[:, self.dims * 2 : self.dims * 3]
            - self.buf_obs[:, 0 : self.dims],
            axis=1,
        )
        for idx in range(self.num_envs):
            self.reset_infos[idx] = {}
        return self.buf_obs

    def render(self) -> matplotlib.figure.Figure:
        if self.dims == 2:
            obs_plot = self.obs_hist[: self.counter]

            # Calculate the magnitude of the combined velocity vector
            velocity_magnitude = np.sqrt(obs_plot[:, 2] ** 2 + obs_plot[:, 3] ** 2)

            # Create a color map based on velocity magnitude
            norm = plt.Normalize(velocity_magnitude.min(), velocity_magnitude.max())
            colors = plt.cm.viridis(norm(velocity_magnitude))

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

            # Subplot 1: Trajectory with velocity magnitude gradient
            for i in range(1, len(obs_plot)):
                ax1.plot(
                    obs_plot[i - 1 : i + 1, 0],
                    obs_plot[i - 1 : i + 1, 1],
                    color=colors[i - 1],
                    linewidth=2,
                )

            # Plot initial and goal points
            ax1.scatter(
                obs_plot[0, 0],
                obs_plot[0, 1],
                color="red",
                s=100,
                label="Initial Point",
            )
            ax1.scatter(
                obs_plot[0, 6], obs_plot[0, 7], color="blue", s=100, label="Goal Point"
            )
            # if self.discrete_action[0] == 1:
            #     loc_discx = self.buf_obs[0,0]
            #     loc_discy = self.buf_obs[0,1]
            #     ax1.scatter(loc_discx, loc_discy, color = "red", marker = "X", label="Deploy Position")
            # Add colorbar for velocity magnitude
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax1, label="Velocity Magnitude")

            ax1.set_xlabel("X Position")
            ax1.set_ylabel("Z Position")
            ax1.legend()
            ax1.set_title("Trajectory with Velocity Magnitude Gradient")

            time = np.linspace(
                0, len(obs_plot[:, 1]) * self.cfg["policy_dt"], len(obs_plot[:, 1])
            )
            # Subplot 2: Thrust vs Y-location
            ax2.plot(time, obs_plot[:, 4], label="X-Thrust", color="orange")
            ax2.plot(time, obs_plot[:, 5], label="Z-Thrust", color="purple")

            ax2.set_xlabel("Time")
            ax2.set_ylabel("Thrust Value")
            ax2.legend()
            ax2.set_title("Thrust vs Time")

            ax3.plot(time, obs_plot[:, 3])
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Z-Velocity")
            ax3.set_title("Z-Velocity vs Time")
            plt.tight_layout()
            if self.test and not self.gif:
                plt.show()
                plt.close()
            if self.gif:
                # Calculate the magnitude of the combined velocity vector
                velocity_magnitude = np.sqrt(obs_plot[:, 2] ** 2 + obs_plot[:, 3] ** 2)

                # Create a color map based on velocity magnitude
                norm = plt.Normalize(velocity_magnitude.min(), velocity_magnitude.max())
                colors = plt.cm.viridis(norm(velocity_magnitude))

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Plot initial and goal points
                ax1.scatter(
                    obs_plot[0, 0],
                    obs_plot[0, 1],
                    color="red",
                    s=100,
                    label="Initial Point",
                )
                ax1.scatter(
                    obs_plot[0, 6],
                    obs_plot[0, 7],
                    color="blue",
                    s=100,
                    label="Goal Point",
                )
                # if self.discrete_action[0] == 1:
                #     loc_discx = self.buf_obs[0,0]
                #     loc_discy = self.buf_obs[0,1]
                #     ax1.scatter(loc_discx, loc_discy, color = "red", marker = "X", label="Deploy Position")
                # Add colorbar for velocity magnitude
                sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax1, label="Velocity Magnitude")

                ax1.set_xlabel("X Position")
                ax1.set_ylabel("Z Position")
                ax1.set_xlim(
                    min(min(obs_plot[:, 0]), obs_plot[0, 6]) - 100,
                    max(max(obs_plot[:, 0]), obs_plot[0, 6]) + 100,
                )
                ax1.set_ylim(
                    min(min(obs_plot[:, 1]), obs_plot[0, 7]) - 100,
                    max(max(obs_plot[:, 1]), obs_plot[0, 7]) + 100,
                )
                ax1.legend()
                ax1.set_title("Trajectory with Velocity Magnitude Gradient")

                ax2.set_xlabel("Time")
                ax2.set_ylabel("Thrust Value")
                ax2.set_xlim(min(time) - 1, max(time) - 1)
                ax2.set_ylim(
                    min(min(obs_plot[:, 4]), min(obs_plot[:, 5])) - 1,
                    max(max(obs_plot[:, 4]), max(obs_plot[:, 5])) + 1,
                )
                ax2.legend()
                ax2.set_title("Thrust vs Time")

                def update(frame):
                    start_frame = int((frame - 1) * self.cfg["gif_steps/frame"])
                    if start_frame < 1:
                        start_frame = 1
                    frame *= self.cfg["gif_steps/frame"]
                    frame = int(frame)
                    for i in range(start_frame, frame):
                        ax1.plot(
                            obs_plot[i - 1 : i + 1, 0],
                            obs_plot[i - 1 : i + 1, 1],
                            color=colors[i - 1],
                            linewidth=2,
                        )

                    time = np.linspace(
                        0,
                        len(obs_plot[0 : frame + 1, 1]) * self.cfg["policy_dt"],
                        len(obs_plot[0 : frame + 1, 1]),
                    )
                    # Subplot 2: Thrust vs Y-location
                    ax2.plot(
                        time,
                        obs_plot[0 : frame + 1, 4],
                        label="X-Thrust",
                        color="orange",
                    )
                    ax2.plot(
                        time,
                        obs_plot[0 : frame + 1, 5],
                        label="Z-Thrust",
                        color="purple",
                    )

                    return

                anim = FuncAnimation(
                    fig,
                    update,
                    frames=np.arange(0, len(obs_plot) / self.cfg["gif_steps/frame"]),
                    interval=self.cfg["gif_steps/frame"]
                    * self.cfg["policy_dt"]
                    * 1000
                    / self.cfg["gif_speed"],
                )

                # save a gif of the animation using the writing package from magick

                anim.save("gifs/policy.gif", dpi=80, writer="pillow")
            print("Plot made")
        elif self.dims == 3:
            obs_plot = self.obs_hist[: self.counter]

            # Calculate the magnitude of the combined velocity vector
            velocity_magnitude = np.sqrt(
                obs_plot[:, self.dims] ** 2
                + obs_plot[:, self.dims + 1] ** 2
                + obs_plot[:, self.dims + 2] ** 2
            )

            # Create a color map based on velocity magnitude
            norm = plt.Normalize(velocity_magnitude.min(), velocity_magnitude.max())
            colors = plt.cm.viridis(norm(velocity_magnitude))

            fig = plt.figure(figsize=(21, 6))
            ax1 = fig.add_subplot(131, projection="3d")
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            # Subplot 1: Trajectory with velocity magnitude gradient
            for i in range(1, len(obs_plot)):
                ax1.plot(
                    obs_plot[i - 1 : i + 1, 0],
                    obs_plot[i - 1 : i + 1, 1],
                    obs_plot[i - 1 : i + 1, 2],
                    color=colors[i - 1],
                    linewidth=3,
                )

            # Plot initial and goal points
            ax1.scatter(
                obs_plot[0, 0],
                obs_plot[0, 1],
                obs_plot[0, 2],
                color="red",
                s=100,
                label="Initial Point",
            )
            ax1.scatter(
                obs_plot[0, self.dims * 2],
                obs_plot[0, self.dims * 2 + 1],
                obs_plot[0, self.dims * 2 + 2],
                color="blue",
                s=100,
                label="Goal Point",
            )

            # Add colorbar for velocity magnitude
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax1, label="Velocity Magnitude")

            ax1.set_xlabel("X Position")
            ax1.set_ylabel("Y Position")
            ax1.set_zlabel("Z Position")
            ax1.legend()
            ax1.set_title("Trajectory with Velocity Magnitude Gradient")

            time = np.linspace(
                0, len(obs_plot[:, 1]) * self.cfg["policy_dt"], len(obs_plot[:, 1])
            )

            # Subplot 2: Thrust vs Y-location
            ax2.plot(time, obs_plot[:, self.dims * 3], label="X-Thrust", color="orange")
            ax2.plot(
                time, obs_plot[:, self.dims * 3 + 1], label="Y-Thrust", color="green"
            )
            ax2.plot(time, 100 * obs_plot[:, -1], label="Parachute", color="purple")

            ax2.set_xlabel("Time")
            ax2.set_ylabel("Thrust Value")
            ax2.legend()
            ax2.set_title("Thrust vs Time")

            ax3.plot(time, obs_plot[:, self.dims * 2 - 1])
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Z-Velocity")
            ax3.set_title("Z-Velocity vs Time")

            plt.tight_layout()
            if self.test and not self.gif:
                plt.show()
                plt.close()
            if self.gif:
                # Calculate the magnitude of the combined velocity vector
                velocity_magnitude = np.sqrt(
                    obs_plot[:, self.dims] ** 2 + obs_plot[:, self.dims + 1] ** 2,
                    obs_plot[:, self.dims + 2] ** 2,
                )

                # Create a color map based on velocity magnitude
                norm = plt.Normalize(velocity_magnitude.min(), velocity_magnitude.max())
                colors = plt.cm.viridis(norm(velocity_magnitude))

                fig = plt.figure(figsize=(14, 6))
                ax1 = fig.add_subplot(121, projection="3d")
                ax2 = fig.add_subplot(122)

                # Plot initial and goal points
                ax1.scatter(
                    obs_plot[0, 0],
                    obs_plot[0, 1],
                    obs_plot[0, 2],
                    color="red",
                    s=100,
                    label="Initial Point",
                )
                ax1.scatter(
                    obs_plot[0, self.dims * 2],
                    obs_plot[0, self.dims * 2 + 1],
                    obs_plot[0, self.dims * 2 + 2],
                    color="blue",
                    s=100,
                    label="Goal Point",
                )

                sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax1, label="Velocity Magnitude")

                ax1.set_xlabel("X Position")
                ax1.set_ylabel("Y Position")
                ax1.set_zlabel("Z Position")
                ax1.set_xlim(
                    min(min(obs_plot[:, 0]), obs_plot[0, 6]) - 100,
                    max(max(obs_plot[:, 0]), obs_plot[0, 6]) + 100,
                )
                ax1.set_ylim(
                    min(min(obs_plot[:, 1]), obs_plot[0, 7]) - 100,
                    max(max(obs_plot[:, 1]), obs_plot[0, 7]) + 100,
                )
                ax1.set_zlim(
                    min(min(obs_plot[:, 2]), obs_plot[0, 8]) - 100,
                    max(max(obs_plot[:, 2]), obs_plot[0, 8]) + 100,
                )
                ax1.legend()
                ax1.set_title("Trajectory with Velocity Magnitude Gradient")

                ax2.set_xlabel("Time")
                ax2.set_ylabel("Thrust Value")
                ax2.set_xlim(min(time) - 1, max(time) - 1)
                ax2.set_ylim(
                    min(min(obs_plot[:, 9]), min(obs_plot[:, 10])) - 1,
                    max(max(obs_plot[:, 9]), max(obs_plot[:, 10])) + 1,
                )
                ax2.legend()
                ax2.set_title("Thrust vs Time")

                def update(frame):
                    start_frame = int((frame - 1) * self.cfg["gif_steps/frame"])
                    if start_frame < 1:
                        start_frame = 1
                    frame *= self.cfg["gif_steps/frame"]
                    frame = int(frame)
                    for i in range(start_frame, frame):
                        ax1.plot(
                            obs_plot[i - 1 : i + 1, 0],
                            obs_plot[i - 1 : i + 1, 1],
                            obs_plot[i - 1 : i + 1, 2],
                            color=colors[i - 1],
                            linewidth=3,
                        )

                    time = np.linspace(
                        0,
                        len(obs_plot[0 : frame + 1, 1]) * self.cfg["policy_dt"],
                        len(obs_plot[0 : frame + 1, 1]),
                    )
                    # Subplot 2: Thrust vs Y-location
                    ax2.plot(
                        time,
                        obs_plot[0 : frame + 1, 9],
                        label="X-Thrust",
                        color="orange",
                    )
                    ax2.plot(
                        time,
                        obs_plot[0 : frame + 1, 10],
                        label="Y-Thrust",
                        color="green",
                    )
                    # ax2.plot(time, obs_plot[0:frame+1, 8], label='Y-Thrust', color='purple')

                    return

                anim = FuncAnimation(
                    fig,
                    update,
                    frames=np.arange(0, len(obs_plot) / self.cfg["gif_steps/frame"]),
                    interval=self.cfg["gif_steps/frame"]
                    * self.cfg["policy_dt"]
                    * 1000
                    / self.cfg["gif_speed"],
                )

                # save a gif of the animation using the writing package from magick

                anim.save("gifs/policy.gif", dpi=80, writer="pillow")
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
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        idx = self._get_indices(indices)
        return [False] * len(idx)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        raise NotImplementedError

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        raise NotImplementedError

    def set_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        raise NotImplementedError


if __name__ == "__main__":
    import time

    n = 10
    T = 10
    env = RK4Env(n, config)
    u = np.array([[0] * n, [0.0] * n, [0] * n], dtype=np.float32).T
    d = np.array([0] * n, dtype=np.int32).T
    now = time.time()
    for _ in range(T):
        obs, rew, done, info = env.step(u)
    print(f"{int(n*T/(time.time() - now)):_d} steps/second")
