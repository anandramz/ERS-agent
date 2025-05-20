import numpy as np
import torch
from dqn_agent import DQNAgent
from ers_risk_slap_env import ERSRiskSlapEnv
import matplotlib.pyplot as plt

# Hyperparameters
episodes = 1000
target_update_freq = 10
save_path = "dqn_ers_weights.pth"

# Initialize environment and agent
env = ERSRiskSlapEnv()
agent = DQNAgent()

# Logging
episode_rewards = []
episode_epsilons = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        total_reward += reward

    if episode % target_update_freq == 0:
        agent.update_target()

    episode_rewards.append(total_reward)
    episode_epsilons.append(agent.epsilon)

    print(f"Episode {episode:3d} | Total Reward: {total_reward:5.2f} | Epsilon: {agent.epsilon:.3f}")

# Save trained weights
torch.save(agent.q_net.state_dict(), save_path)
print(f"Model saved to {save_path}")

# Plot rewards and epsilon
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(1, 2, 2)
plt.plot(episode_epsilons)
plt.title("Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")

plt.tight_layout()
plt.show()
