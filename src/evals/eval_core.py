import os
import time
import gymnasium as gym

from src.utils.core import load_config, set_seed
from src.utils.model_utils import load_model
from src.utils.logger import EpisodeLogger
from src.utils.video_utils import concat_episode_videos

from src.agents.dqn_agent import DQNAgent
from src.agents.reinforce_agent import ReinforceAgent
from src.agents.q_learning import QLearningAgent
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent

def create_agent(algo, env, hp):
    if algo.startswith("dqn"):
        return DQNAgent(env.observation_space.shape[0], env.action_space.n, hp)
    elif algo == "reinforce":
        return ReinforceAgent(env.observation_space.shape[0], env.action_space.n, hp)
    elif algo == "q_learning":
        return QLearningAgent(env.action_space.n, hp)
    elif algo == "random":
        return RandomAgent(env)
    elif algo == "heuristic":
        return HeuristicAgent()
    else:
        raise ValueError(f"Unknown algo: {algo}")

def evaluate_model(model_path, config_path, algo, seed, n_episodes=100):
    cfg = load_config(config_path)
    hp = cfg.get("hyperparameters", {})
    
    os.makedirs(f"videos/tests/{algo}_seed{seed}", exist_ok=True)
    env = gym.make(cfg["env"], render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos/tests/{algo}_seed{seed}")
        
    set_seed(seed + 1000)

    agent = create_agent(algo, env, hp)
    
    # Load weights
    if not load_model(agent, algo, model_path):
        return None

    os.makedirs("eval_logs", exist_ok=True)
    log_path = f"eval_logs/{algo}_seed{seed}.csv"
    logger = EpisodeLogger(log_path)
    
    print(f"\n[EVAL] Test de {algo} (seed {seed}) sur {n_episodes} épisodes...")
    
    t0 = time.time()
    for ep in range(n_episodes):
        # We explicitly set a test seed to ensure reproducible evaluation
        # across all models. We use test_seed = 10000 + ep 
        obs, _ = env.reset(seed=10000 + ep)
        ep_return = 0.0
        ep_len = 0
        done = False
        
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            done = terminated or truncated

        logger.log(ep, ep_return, ep_len, terminated, truncated, info)
        
    logger.close()
    env.close()
    
    concat_episode_videos(f"videos/tests/{algo}_seed{seed}")
    
    print(f"[EVAL] Terminé en {time.time()-t0:.1f}s. Logs sauvegardés dans {log_path}")
    return log_path
