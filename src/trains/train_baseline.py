import os
import time
import gymnasium as gym
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.utils.logger import EpisodeLogger
from src.utils.core import set_seed, _result

def train_baseline(cfg, seed, log_path):
    """Boucle d'exécution pour les agents de référence (Random et Heuristic).
    
    Ces agents ne possèdent pas de phase d'apprentissage. On mesure simplement 
    leurs performances sur le nombre de pas définis pour servir de base de 
    comparaison (baselines) aux algorithmes de RL.
    """
    algo = cfg["algo"]
    os.makedirs(f"videos/train/{algo}_seed{seed}", exist_ok=True)
    env  = gym.make(cfg["env"], render_mode="rgb_array")
    env  = gym.wrappers.RecordVideo(env, f"videos/train/{algo}_seed{seed}")
    
    set_seed(seed)
    obs, _ = env.reset(seed=seed)

    agent  = RandomAgent(env) if algo == "random" else HeuristicAgent()
    logger = EpisodeLogger(log_path)

    episode = 0; ep_return = 0.0; ep_len = 0; ep_returns = []
    t0 = time.time()

    for _ in range(cfg["max_steps"]):
        # Sélection d'action selon des règles fixes ou aléatoires
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = next_obs; ep_return += reward; ep_len += 1

        if done:
            ep_returns.append(ep_return)
            logger.log(episode, ep_return, ep_len, terminated, truncated, info)
            print(f"[{algo}|s{seed}] ep={episode:4d} R={ep_return:8.1f} len={ep_len:4d}")
            
            ep_return = 0.0; ep_len = 0; episode += 1
            obs, _ = env.reset()

    logger.close(); env.close()
    return _result(algo, seed, ep_returns, None, time.time()-t0), agent
