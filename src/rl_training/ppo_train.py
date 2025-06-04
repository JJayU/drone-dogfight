from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from sim import CrazyflieEnv
import os

models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = CrazyflieEnv(render_mode=None)
env.reset()

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
model = PPO.load(f"{models_dir}/2000000", env, verbose=1, tensorboard_log="./logs/")   

TIMESTEPS = 100_000
iters = 0

while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
