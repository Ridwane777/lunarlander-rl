import os
import time
import gymnasium as gym
from src.agents.q_learning import QLearningAgent
from src.utils.logger import EpisodeLogger
from src.utils.core import set_seed, _result

def train_q_learning(cfg, seed, log_path):
    """Boucle d'entraînement pour l'algorithme Q-Learning Tabulaire.
    
    L'agent interagit avec l'environnement et met à jour sa table de valeurs Q 
    à chaque pas de temps (TD-Learning).
    """
    algo = cfg["algo"]
    hp  = cfg["hyperparameters"]
    os.makedirs(f"videos/train/{algo}_seed{seed}", exist_ok=True)
    
    env = gym.make(cfg["env"], render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos/train/{algo}_seed{seed}")
    
    set_seed(seed)
    obs, _ = env.reset(seed=seed)

    agent  = QLearningAgent(env.action_space.n, hp)
    logger = EpisodeLogger(log_path)

    episode = 0; ep_return = 0.0; ep_len = 0
    ep_returns = []; solved_at = None
    t0 = time.time()

    for _ in range(cfg["max_steps"]):
        state  = obs
        # 1. Sélection d'action (epsilon-greedy)
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 2. Mise à jour de la Q-Table
        agent.update(state, action, reward, next_obs, float(done))
        obs = next_obs; ep_return += reward; ep_len += 1

        if done:
            # 3. Fin d'épisode : Logging et statistiques
            ep_returns.append(ep_return)
            logger.log(episode, ep_return, ep_len, terminated, truncated, info,
                       epsilon=agent.epsilon)
            
            # Affichage du nombre d'états découverts dans la table Excel
            print(f"[q_learning|s{seed}] ep={episode:4d} R={ep_return:8.1f} "
                  f"ε={agent.epsilon:.3f} len={ep_len:4d} Q={len(agent.q_table)} states")

            # Vérification du succès
            if episode >= cfg["success_window"] - 1:
                mw = sum(ep_returns[-cfg["success_window"]:]) / cfg["success_window"]
                if mw >= cfg["success_threshold"] and solved_at is None:
                    solved_at = episode
                    print(f"\n[q_learning|s{seed}] RÉSOLU ep={episode} (mean100={mw:.1f})\n")
                    break

            # Reset pour l'épisode suivant
            ep_return = 0.0; ep_len = 0; episode += 1
            obs, _ = env.reset()

    logger.close(); env.close()
    return _result("q_learning", seed, ep_returns, solved_at, time.time()-t0), agent
