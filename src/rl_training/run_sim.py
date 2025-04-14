from sim import CrazyflieEnv

env = CrazyflieEnv(render_mode=False)
obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    action = [0, 0, 0, 0]
    obs, reward, terminated, _, _ = env.step(action)
    print(reward)
    if terminated:
        obs = env.reset()
