import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


class CrazyflieEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.dt = 1.0 / 200.0  # 200 Hz

        xml_path = '/home/ws/src/drone_mujoco/model/scene.xml'
        
        # Check if file exists before loading
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found at: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set simulation timestep to match our control frequency
        self.model.opt.timestep = self.dt

        self.viewer = None
        if self.render_mode == "human":
            try:
                import mujoco_viewer
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            except ImportError:
                print("Warning: mujoco_viewer not available. Install with: pip install mujoco-viewer")

        # Actions: collective thrust (0.0 - 1.0), body rates (roll, pitch, yaw) in rad/s
        action_low = np.array([0.0, -5.0, -5.0, -5.0], dtype=np.float32)
        action_high = np.array([1.0, 5.0, 5.0, 5.0], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation: pos (3) + vel (3) + RPY (3) + ang vel (3) + local_dist (3) + target_yaw (1) = 16
        obs_high = np.array([np.inf] * 16, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.target_position = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0  # Target yaw in radians
        
        self.no_steps = 0
        self.prev_dist_to_target = 0
        
        self.target_hold_steps = int(2.0 / self.dt)  # 2 seconds at 30Hz = 60 steps
        self.at_target_counter = 0

        # Initialize random seed
        self.np_random = None

    def _mujoco_quat_to_scipy(self, mj_quat):
        """Convert MuJoCo quaternion [w,x,y,z] to scipy format [x,y,z,w]"""
        return np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize random number generator
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
            
        mujoco.mj_resetData(self.model, self.data)

        # Reset drone position and velocity
        self.data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # MuJoCo identity quaternion [w,x,y,z]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        self.no_steps = 0
        
        # Set random target position
        self.target_position = np.array([
            self.np_random.uniform(-3.0, 3.0),
            self.np_random.uniform(-3.0, 3.0),
            self.np_random.uniform(0.5, 2.0)
        ])
        
        # Random target yaw
        self.target_yaw = self.np_random.uniform(-np.pi, np.pi)
        
        # Forward simulation to settle physics
        mujoco.mj_forward(self.model, self.data)
        
        self.prev_dist_to_target = np.linalg.norm(self.data.qpos[0:3] - self.target_position)

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Ensure action is numpy array with correct dtype
        action = np.array(action, dtype=np.float32)
        
        thrust = np.clip(action[0], 0.0, 1.0)
        roll_rate = np.clip(action[1], -5.0, 5.0)
        pitch_rate = np.clip(action[2], -5.0, 5.0)
        yaw_rate = np.clip(action[3], -5.0, 5.0)
        
        # Motor mixing for quadcopter X configuration
        k_mix = 0.15  # mixing coefficient
        
        m1 = thrust + k_mix * (-roll_rate + pitch_rate - yaw_rate)  # front-right
        m2 = thrust + k_mix * (-roll_rate - pitch_rate + yaw_rate)  # back-right  
        m3 = thrust + k_mix * (roll_rate - pitch_rate - yaw_rate)   # back-left
        m4 = thrust + k_mix * (roll_rate + pitch_rate + yaw_rate)   # front-left
        
        # Clip motor commands
        motor_commands = np.array([m1, m2, m3, m4], dtype=np.float32)
        motor_commands = np.clip(motor_commands, 0.0, 1.0)
        print(f"Motor commands: {motor_commands}")
        # Apply control
        if len(self.data.ctrl) >= 4:
            self.data.ctrl[:4] = motor_commands
        else:
            self.data.ctrl[:] = motor_commands[:len(self.data.ctrl)]
        
        # Step the simulation
        for iters in range(4):
            mujoco.mj_step(self.model, self.data)
        
        self.no_steps += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = bool(self._check_termination(obs))
        
        truncated = self.no_steps > 12000
        info = {}
        
        # Check if target reached
        pos = self.data.qpos[0:3]
        current_yaw = self._get_yaw_from_quat(self.data.qpos[3:7])
        yaw_error = self._wrap_angle(self.target_yaw - current_yaw)
        
        at_target = (np.linalg.norm(pos - self.target_position) < 0.2 and abs(yaw_error) < 0.1)
        if at_target:
            self.at_target_counter += 1
            if self.at_target_counter == self.target_hold_steps:
                reward += 1000.0
                # Set new target
                self.target_position = np.array([
                    self.np_random.uniform(-3.0, 3.0),
                    self.np_random.uniform(-3.0, 3.0),
                    self.np_random.uniform(0.5, 2.0)
                ])
                self.target_yaw = self.np_random.uniform(-np.pi, np.pi)
                self.at_target_counter = 0
        else:
            self.at_target_counter = 0

        self.prev_dist_to_target = np.linalg.norm(self.data.qpos[0:3] - self.target_position)
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pos = self.data.qpos[0:3].copy()
        lin_vel = self.data.qvel[0:3].copy()
        mj_quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()
        
        # Convert MuJoCo quaternion to scipy format
        scipy_quat = self._mujoco_quat_to_scipy(mj_quat)
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        try:
            scipy_quat = scipy_quat / np.linalg.norm(scipy_quat)  # Normalize quaternion
            rpy = R.from_quat(scipy_quat).as_euler('xyz', degrees=False)
        except (ValueError, np.linalg.LinAlgError):
            # Handle invalid quaternion
            rpy = np.array([0.0, 0.0, 0.0])
        
        # Calculate local distance to target
        global_dist = self.target_position - pos
        try:
            rot = R.from_quat(scipy_quat).as_matrix()
            local_dist = rot.T @ global_dist
        except (ValueError, np.linalg.LinAlgError):
            # Handle invalid quaternion
            local_dist = global_dist
        
        # Add target yaw to observation
        current_yaw = rpy[2]
        yaw_error = self._wrap_angle(self.target_yaw - current_yaw)

        obs = np.concatenate([pos, lin_vel, rpy, ang_vel, local_dist, [yaw_error]])
        return obs.astype(np.float32)

    def _get_yaw_from_quat(self, mj_quat):
        """Extract yaw from quaternion"""
        try:
            scipy_quat = self._mujoco_quat_to_scipy(mj_quat)
            scipy_quat = scipy_quat / np.linalg.norm(scipy_quat)
            return R.from_quat(scipy_quat).as_euler('xyz', degrees=False)[2]
        except (ValueError, np.linalg.LinAlgError):
            return 0.0

    def _wrap_angle(self, angle):
        """Normalize angle to range [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _compute_reward(self, obs):
        pos = obs[0:3]
        lin_vel = obs[3:6]
        rpy = obs[6:9]
        ang_vel = obs[9:12]
        current_yaw = rpy[2]
        
        # Position error
        pos_error = np.linalg.norm(pos - self.target_position)
        
        # Orientation error - we want to maintain roll=0, pitch=0
        roll_pitch_error = np.linalg.norm(rpy[0:2])
        
        # Yaw error
        yaw_error = self._wrap_angle(self.target_yaw - current_yaw)
        
        # Velocity penalties
        lin_vel_penalty = 0.01 * np.linalg.norm(lin_vel)
        ang_vel_penalty = 0.01 * np.linalg.norm(ang_vel)
        
        # Distance change reward
        current_dist = np.linalg.norm(pos - self.target_position)
        delta_dist = self.prev_dist_to_target - current_dist
        
        # Composite reward function
        reward = (
            -6.0 * pos_error +           # main penalty for distance from target
            -5.0 * roll_pitch_error +     # penalty for deviation from level
            -5.0 * abs(yaw_error) +       # penalty for yaw error
            -lin_vel_penalty +            # penalty for excessive linear velocity
            -3*ang_vel_penalty +            # penalty for excessive angular velocity
            2.0 * delta_dist              # reward for approaching target
        )
        
        # Large penalty for termination
        if self._check_termination(obs):
            reward -= 1000.0
        
        return reward

    
    def _check_termination(self, obs):
        x, y, z = obs[0:3]
        # Check if drone is out of bounds
        # return False
        return z < 0.2 or z > 5.0 or x < -5.0 or x > 5.0 or y < -5.0 or y > 5.0

    def render(self):
        if self.viewer and hasattr(self.viewer, 'is_alive') and self.viewer.is_alive:
            self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None