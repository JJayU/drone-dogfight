import numpy as np
import mujoco
from env.drone_sim import Sim

# Environment class for the drone simulation, used for testing MPC controllers.

class DroneEnv:
    def __init__(self, use_gui=False):
        self.sim = Sim(use_gui=use_gui)
        self.action_space = np.array([0.0, 0.0, 0.0, 0.0])  
        self.observation_space = np.zeros(7)  
        self.done = False

    def reset(self):
        # Reset the simulation state
        self.sim.data.qpos[:] = 0.0  # Reset positions
        self.sim.data.qvel[:] = 0.0  # Reset velocities
        self.sim.data.qpos[2] = 2.0  # Set initial height

        return self._get_observation()

    def step(self, action):
        # Apply the action (motor commands)
        self.sim.set_control(action)

        # Step the simulation
        mujoco.mj_step(self.sim.model, self.sim.data)

        # Get the new state
        observation = self._get_observation()

        # Calculate reward (example: penalize distance from origin)
        x, y, z = self.sim.data.qpos[0:3]
        reward = -np.sqrt(x**2 + y**2 + (z - 1.0)**2)  # Penalize distance from (0, 0, 1)

        # Check termination condition (example: if the drone falls below a certain height)
        self.done = z < 0.2

        return observation, reward, False, False, {} #self.done, {}

    def render(self):
        if self.sim.viewer and self.sim.viewer.is_alive:
            self.sim.viewer.render()
            
    def quaternion_to_euler(self, q):
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def _get_observation(self):
        # Combine position and orientation into an observation
        position = self.sim.data.qpos[0:3]
        speed = self.sim.data.qvel[0:3]
        orientation = self.sim.data.qpos[3:7]
        # Convert quaternion to Euler angles
        euler_angles = self.quaternion_to_euler(orientation)
        # Calculate angular velocity in roll, pitch, yaw (RPY) space
        angular_velocity = self.sim.data.qvel[3:6]
        # Combine position, speed, orientation, and angular velocity into a single observation vector
        observation = np.concatenate([position, speed, euler_angles, angular_velocity])
        return observation