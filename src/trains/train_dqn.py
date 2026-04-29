import os
import time
import gymnasium as gym
from src.agents.dqn_agent import DQNAgent
from src.utils.logger import EpisodeLogger
from src.utils.core import set_seed, _result

def train_dqn(cfg, seed, log_path):
    algo = cfg["algo"]
    hp   = cfg["hyperparameters"]
    os.makedirs(f"videos/train/{algo}_seed{seed}", exist_ok=True)
    env  = gym.make(cfg["env"], render_mode="rgb_array")
    env  = gym.wrappers.RecordVideo(env, f"videos/train/{algo}_seed{seed}")
    set_seed(seed)
    obs, _ = env.reset(seed=seed)

    agent  = DQNAgent(env.observation_space.shape[0], env.action_space.n, hp)
    logger = EpisodeLogger(log_path)

    episode = 0; ep_return = 0.0; ep_len = 0
    ep_returns = []; ep_losses = []; solved_at = None
    t0 = time.time()

    for _ in range(cfg["max_steps"]):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, float(done))
        loss = agent.update()
        if loss is not None:
            ep_losses.append(loss)
        obs = next_obs; ep_return += reward; ep_len += 1

        if done:
            ep_returns.append(ep_return)
            mean_loss = sum(ep_losses) / len(ep_losses) if ep_losses else None
            logger.log(episode, ep_return, ep_len, terminated, truncated, info,
                       epsilon=agent.epsilon, loss=mean_loss)
            tag = f"loss={mean_loss:.4f}" if mean_loss is not None else "filling buffer..."
            print(f"[{algo}|s{seed}] ep={episode:4d} R={ep_return:8.1f} "
                  f"ε={agent.epsilon:.3f} len={ep_len:4d} {tag}")

            if episode >= cfg["success_window"] - 1:
                mw = sum(ep_returns[-cfg["success_window"]:]) / cfg["success_window"]
                if mw >= cfg["success_threshold"] and solved_at is None:
                    solved_at = episode
                    print(f"\n [{algo}|s{seed}] RÉSOLU ep={episode} (mean100={mw:.1f})\n")
                    break

            ep_return = 0.0; ep_len = 0; ep_losses = []; episode += 1
            obs, _ = env.reset()

    logger.close(); env.close()
    return _result(algo, seed, ep_returns, solved_at, time.time()-t0), agent
