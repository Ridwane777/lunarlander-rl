import os
import time
import gymnasium as gym
from src.agents.reinforce_agent import ReinforceAgent
from src.utils.logger import EpisodeLogger
from src.utils.core import set_seed, _result

def train_reinforce(cfg, seed, log_path):
    """Boucle d'entraînement pour l'algorithme REINFORCE (Policy Gradient).
    
    À l'inverse du DQN, cet agent est On-Policy et n'effectue sa mise à jour 
    qu'à la FIN de chaque épisode en utilisant la trajectoire complète.
    """
    algo = cfg["algo"]
    hp  = cfg["hyperparameters"]
    os.makedirs(f"videos/train/{algo}_seed{seed}", exist_ok=True)
    
    env = gym.make(cfg["env"], render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos/train/{algo}_seed{seed}")
    
    set_seed(seed)
    obs, _ = env.reset(seed=seed)

    agent  = ReinforceAgent(env.observation_space.shape[0], env.action_space.n, hp)
    logger = EpisodeLogger(log_path)

    episode = 0; ep_return = 0.0; ep_len = 0
    ep_returns = []; solved_at = None
    t0 = time.time()

    for _ in range(cfg["max_steps"]):
        # 1. Sélection d'action (stochastique)
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 2. Collecte de la trajectoire
        agent.store_reward(reward)
        obs = next_obs; ep_return += reward; ep_len += 1

        if done:
            # 3. Mise à jour de la politique (Théorème du Gradient de Politique)
            loss = agent.update()
            ep_returns.append(ep_return)
            
            logger.log(episode, ep_return, ep_len, terminated, truncated, info, loss=loss)
            print(f"[reinforce|s{seed}] ep={episode:4d} R={ep_return:8.1f} "
                  f"len={ep_len:4d} loss={loss:.4f}")

            # Vérification du succès
            if episode >= cfg["success_window"] - 1:
                mw = sum(ep_returns[-cfg["success_window"]:]) / cfg["success_window"]
                if mw >= cfg["success_threshold"] and solved_at is None:
                    solved_at = episode
                    print(f"\n[reinforce|s{seed}] RÉSOLU ep={episode} (mean100={mw:.1f})\n")
                    break

            # Reset pour le prochain épisode
            ep_return = 0.0; ep_len = 0; episode += 1
            obs, _ = env.reset()

    logger.close(); env.close()
    return _result("reinforce", seed, ep_returns, solved_at, time.time()-t0), agent
