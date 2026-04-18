import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from src.agents.heuristic_agent import HeuristicAgent
from src.utils.logger import EpisodeLogger

env = gym.make("LunarLander-v3")
agents = HeuristicAgent(env)
logger = EpisodeLogger("logs/heuristic2.csv")

obs, info = env.reset(seed=0)
ep_return, ep_len, episode = 0, 0, 0

for _ in range(10000):
    action = agents.select_action(obs)
    obs, reward, truncated, terminated, info = env.step(action)
    ep_len += 1
    ep_return += reward

    if terminated or truncated:
        logger.log(episode, ep_return, ep_len, terminated, truncated, info)
        print(f"Episode {episode} | reward={ep_return:.1f} | steps={ep_len}")
        ep_return, ep_len = 0, 0
        episode += 1
        obs, info = env.reset()

logger.close()
env.close()

