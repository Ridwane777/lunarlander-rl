import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.dqn_agent import DQNAgent
from src.utils.logger import EpisodeLogger

env = gym.make("LunarLander-v3")
obs, info = env.reset(seed=0)

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

config_hard = {
    "gamma": 0.99,
    "lr": 0.0005,
    "batch_size": 64,
    "buffer_size": 100000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "target_update_freq": 500,
    "soft_update": False,
    "tau": 0.005,
}

config_soft = {
    "gamma": 0.99,
    "lr": 0.0005,
    "batch_size": 64,
    "buffer_size": 100000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "target_update_freq": 500,
    "soft_update": True,
    "tau": 0.005,
}

agent = DQNAgent(obs_dim, n_actions, config_soft)
logger = EpisodeLogger("logs/dqn_soft.csv")

episode = 0
ep_return = 0
ep_len = 0
ep_returns = []

for step in range(500000):
    action = agent.select_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    agent.store(obs, action, reward, next_obs, float(done))
    loss = agent.update()

    obs = next_obs
    ep_return += reward
    ep_len += 1

    if done:
        ep_returns.append(ep_return)
        logger.log(episode, ep_return, ep_len, terminated, truncated, info)

        if loss:
            print(f"Episode {episode:4d} | reward={ep_return:8.1f} | epsilon={agent.epsilon:.3f} | steps={ep_len:4d} | loss={loss:.3f}")
        else:
            print(f"Episode {episode:4d} | reward={ep_return:8.1f} | epsilon={agent.epsilon:.3f} | steps={ep_len:4d} | filling buffer...")

        # Critère de succès
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