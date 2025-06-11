import gymnasium as gym
import numpy as np
import mujoco
import os
from env.drone_env import DroneEnv
from controllers.mpc_drone import MPCController

# This script tests the MPC controller for a drone in a MuJoCo simulation.

n_steps = 1000
env = DroneEnv(use_gui=True)

controller = MPCController(env)

initial_state = env.reset()

episode_reward = 0.
for i in range(n_steps):
    # Target state
    yref = np.array([5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    
    action = controller.compute_control(yref)
    next_observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
print(f"Episode reward: {episode_reward}")
