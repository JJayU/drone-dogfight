import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os


class CrazyflieEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 0.005

        xml_path = '/home/ws/src/drone_mujoco/model/scene.xml'
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        if self.render_mode == "human":
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Actions: 4 silniki (0.0 - 1.0)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: pos (3) + vel (3) + quat (4) + ang vel (3) = 13
        obs_high = np.array([np.inf] * 13, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.target_position = np.array([0.5, -0.0, 1.5])  # cel do utrzymania
        
        self.no_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        self.no_steps = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, 0.0, 1.0)
        mujoco.mj_step(self.model, self.data)
        
        self.no_steps += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = bool(self._check_termination(obs))
        truncated = False
        if self.no_steps > 1000:
            truncated = True
        info = {}

        if self.render_mode == "human" and self.viewer and self.viewer.is_alive:
            self.viewer.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pos = self.data.qpos[0:3]
        lin_vel = self.data.qvel[0:3]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        return np.concatenate([pos, lin_vel, quat, ang_vel]).astype(np.float32)

    def _compute_reward(self, obs):
        target_quat = np.array([1, 0, 0, 0])
        current_quat = obs[6:10]
        
        quat_error = 1 - np.abs(np.dot(target_quat, current_quat))
        
        ang_vel = obs[10:13]
        ang_vel_penalty = 0.01 * np.linalg.norm(ang_vel)
        
        pos_err = np.linalg.norm(obs[0:3] - self.target_position)
        
        reward = -5.0 * quat_error - ang_vel_penalty - 10 * pos_err
        
        return reward

    def _check_termination(self, obs):
        x = obs[0]
        y = obs[1]
        z = obs[2]
        
        return z < 0.2 or z > 3.0 or x < -2.0 or x > 2.0 or y < -2.0 or y > 2.0

    def render(self):
        if self.viewer:
            self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
