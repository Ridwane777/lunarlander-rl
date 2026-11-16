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
from src.baselines.random_agent import RandomAgent
from src.baselines.heuristic import HeuristicAgent

ENV_ID = "LunarLander-v3"
EVAL_SEED = 10000   # every agent sees the same episodes: seeds 10000 to 10000 + n - 1


def create_agent(algo, env, hp):
    """Build a learning agent for the chosen algorithm."""
    if algo.startswith("dqn"):
        return DQNAgent(env.observation_space.shape[0], env.action_space.n, hp)
    if algo.startswith("reinforce"):
        return ReinforceAgent(env.observation_space.shape[0], env.action_space.n, hp)
    if algo.startswith("q_learning"):
        return QLearningAgent(env.action_space.n, hp)
    raise ValueError(f"Unknown algo: {algo}")


def run_episodes(agent, env, logger, n_episodes):
    """Play n_episodes on the shared evaluation seeds and log each one."""
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED + ep)
        ep_return, ep_len, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(agent.select_action(obs))
            ep_return += reward
            ep_len += 1
            done = terminated or truncated
        logger.log(ep, ep_return, ep_len, terminated, truncated, info)


def _evaluate(agent_factory, name, n_episodes):
    video_dir = f"videos/evals/{name}"
    os.makedirs(video_dir, exist_ok=True)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_dir)
    agent = agent_factory(env)
    if agent is None:
        env.close()
        return None

    log_path = f"logs/eval/{name}.csv"
    logger = EpisodeLogger(log_path)
    print(f"\n[EVAL] {name}: {n_episodes} episodes...")
    t0 = time.time()
    run_episodes(agent, env, logger, n_episodes)
    logger.close()
    env.close()
    concat_episode_videos(video_dir)
    print(f"[EVAL] Done in {time.time()-t0:.1f}s. Logs saved to {log_path}")
    return log_path


def evaluate_model(model_path, config_path, algo, seed, n_episodes=100):
    """Evaluate a trained model greedily (DQN, Q-Learning) or by sampling (REINFORCE)."""
    hp = load_config(config_path).get("hyperparameters", {})

    def factory(env):
        set_seed(seed + 1000)
        agent = create_agent(algo, env, hp)
        return agent if load_model(agent, algo, model_path) else None

    return _evaluate(factory, f"{algo}_seed{seed}", n_episodes)


def evaluate_heuristic(n_episodes=100):
    """The rule-based controller is deterministic: one run covers it."""
    return _evaluate(lambda env: HeuristicAgent(), "heuristic", n_episodes)


def evaluate_random(seed, n_episodes=100):
    def factory(env):
        env.action_space.seed(seed)
        return RandomAgent(env)

    return _evaluate(factory, f"random_seed{seed}", n_episodes)
