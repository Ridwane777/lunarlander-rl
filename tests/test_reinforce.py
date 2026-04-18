import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from src.agents.reinforce_agent import ReinforceAgent
from src.utils.logger import EpisodeLogger
import numpy as np

env = gym.make("LunarLander-v3")
obs, info = env.reset(seed=0)

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

config = {"gamma": 0.99, "lr": 0.001}
agent = ReinforceAgent(obs_dim, n_actions, config)
logger = EpisodeLogger("logs/reinforce.csv")

episode = 0
ep_return = 0

for step in range(200000):
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    agent.store_reward(reward)
    ep_return += reward

    if terminated or truncated:
        loss = agent.update()
        logger.log(episode, ep_return, step, terminated, truncated, info)
        print(f"Episode {episode} | reward={ep_return:.1f} | loss={loss:.3f}")
        ep_return = 0
        episode += 1
        obs, info = env.reset()

logger.close()
env.close()