import gymnasium as gym
import numpy as np

from controllers.mpc import MPCController

n_steps = 200
env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

controller = MPCController(env)

initial_state = [-np.pi, 0.0]
env.reset()
env.unwrapped.state = np.array(initial_state)

episode_reward = 0.
for i in range(n_steps):
    action = controller.compute_control()
    next_observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
print(f"Episode reward: {episode_reward}")
