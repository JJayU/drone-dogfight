from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from sim import CrazyflieEnv
import os

# --- Katalogi ---
models_dir = "models/PPO"
logdir = "logs/PPO"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# --- Inicjalizacja środowiska ---
env = CrazyflieEnv(render_mode=None)
check_env(env)  # Opcjonalnie: sprawdzenie zgodności Gym
env.reset()

# --- Konfiguracja modelu ---
ppo_config = {
    "policy": "MlpPolicy",
    "env": env,
    "verbose": 1,
    "tensorboard_log": logdir,
    "learning_rate": 3e-4,       
    "ent_coef": 0.01,           
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
}

# --- Wczytaj lub stwórz model ---
load_path = f"{models_dir}/zmiejszeniev2"
if os.path.exists(load_path + ".zip"):
    print(f"Loading model from {load_path}")
    model = PPO.load(load_path, env=env, tensorboard_log=logdir)
else:
    print("Creating new model")
    model = PPO(**ppo_config)

# --- Trening ---
TIMESTEPS = 100_000
iters = 0

while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
