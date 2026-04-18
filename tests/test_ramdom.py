import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from src.agents.random_agent import RandomAgent
from src.utils.logger import EpisodeLogger

env = gym.make("LunarLander-v3")
agent = RandomAgent(env)
logger = EpisodeLogger("logs/random.csv")

obs, info = env.reset(seed=0)
ep_return, ep_len, episode = 0, 0, 0

for _ in range(10000):
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    ep_return += reward
    ep_len += 1

    if terminated or truncated:
        logger.log(episode, ep_return, ep_len, terminated, truncated, info)
        print(f"Episode {episode} | reward={ep_return:.1f} | steps={ep_len}")
        ep_return, ep_len = 0, 0
        episode += 1
        obs, info = env.reset()

logger.close()
env.close()