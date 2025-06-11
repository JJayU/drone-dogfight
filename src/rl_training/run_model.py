from stable_baselines3 import PPO
from sim import CrazyflieEnv
import time

# Uruchamiamy środowisko z GUI
env = CrazyflieEnv(render_mode="human")

models_dir = "models/PPO"

# Ładujemy model
model = PPO.load(f"{models_dir}/51100000", env=env)

# Reset środowiska
obs, _ = env.reset()



for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
    
    # time.sleep(env.dt)  # żeby render nie był za szybki
    
    # print(obs[0:3])
    
    print(obs[0:3])  # Wyświetlamy pozycję drona
    print(f"Obecny yaw drona: {obs[8]}")   # yaw to trzeci element z RPY (indeks 8)
    print(f"loc dist: {obs[14]}")  
    print(f"error Yaw targetu: {obs[15]}")

env.close()
