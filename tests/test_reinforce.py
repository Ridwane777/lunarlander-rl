import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from src.agents.reinforce_agent import ReinforceAgent
from src.utils.logger import EpisodeLogger

env = gym.make("LunarLander-v3")
obs, info = env.reset(seed=0)

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

config = {"gamma": 0.99, "lr": 0.001}
agent = ReinforceAgent(obs_dim, n_actions, config)
logger = EpisodeLogger("logs/reinforce.csv")

episode = 0
ep_return = 0
ep_len = 0
ep_returns = []

for step in range(200000):
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    agent.store_reward(reward)
    ep_return += reward
    ep_len += 1

    if terminated or truncated:
        loss = agent.update()
        ep_returns.append(ep_return)
        logger.log(episode, ep_return, ep_len, terminated, truncated, info)
        print(f"Episode {episode} | reward={ep_return:.1f} | loss={loss:.3f}")

        if episode >= 100:
            mean_100 = sum(ep_returns[-100:]) / 100
            if mean_100 >= 200:
                print(f"Resolu a l'episode {episode} !")
                break

        ep_return = 0
        ep_len = 0
        episode += 1
        obs, info = env.reset()

logger.close()
env.close()