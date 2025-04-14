from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from sim import CrazyflieEnv

env = CrazyflieEnv(render_mode=None)
check_env(env)

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
model = PPO.load("ppo_crazyflie_up1mx1", env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=1_000_000, progress_bar=True)
model.save("ppo_crazyflie")
