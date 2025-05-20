from ers_risk_slap_env import ERSRiskSlapEnv
from reflex_bot import ReflexBot

env = ERSRiskSlapEnv()
bot = ReflexBot(env)
obs = env.reset()
done = False


while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Done: {done}")

print("-----------------------------")
env = ERSRiskSlapEnv()
obs = env.reset()
done = False

while not done:
    action = bot.select_action()
    obs, reward, done, info = env.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
