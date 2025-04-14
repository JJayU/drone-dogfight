from stable_baselines3 import PPO
from sim import CrazyflieEnv
import time

# Uruchamiamy środowisko z GUI
env = CrazyflieEnv(render_mode="human")

# Ładujemy model
model = PPO.load("ppo_crazyflie", env=env)

# Reset środowiska
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
    
    # time.sleep(env.dt)  # żeby render nie był za szybki
    
    print(obs[0:3])

env.close()
