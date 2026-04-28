import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.q_learning import QLearningAgent
from src.utils.logger import EpisodeLogger

env = gym.make("LunarLander-v3")

obs, _ = env.reset(seed=0)

n_actions = env.action_space.n

config = {
    "gamma": 0.99,
    "lr": 0.1,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
}

agent = QLearningAgent(n_actions, config)
logger = EpisodeLogger("logs/q_learning.csv")

episode = 0
ep_return = 0
ep_len = 0
ep_returns = []

for step in range(200000):
    action = agent.select_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    agent.update(obs, action, reward, next_obs, float(done))

    obs = next_obs
    ep_return += reward
    ep_len += 1

    if done:
        ep_returns.append(ep_return)
        logger.log(episode, ep_return, ep_len, terminated, truncated, info)

        print(f"Episode {episode} | reward={ep_return:.1f} | epsilon={agent.epsilon:.3f}")

        if episode >= 100:
            mean_100 = sum(ep_returns[-100:]) / 100
            if mean_100 >= 200:
                print("Solved!")
                break

        obs, info = env.reset()
        ep_return = 0
        ep_len = 0
        episode += 1

env.close()