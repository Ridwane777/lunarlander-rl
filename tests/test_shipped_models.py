"""End-to-end checks: the shipped weights and the heuristic still land.

Same evaluation episodes as eval.py (seeds 10000 onward), shortened to 20 to keep CI fast.
"""
import os

import gymnasium as gym
import numpy as np
import pytest

from src.baselines.heuristic import HeuristicAgent
from src.evals.eval_core import create_agent
from src.utils.core import load_config
from src.utils.model_utils import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_EPISODES = 20


def mean_score(agent):
    env = gym.make("LunarLander-v3")
    scores = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=10000 + ep)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.select_action(obs))
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    return float(np.mean(scores))


# Best seed of each variant on these 20 episodes: 243.7 and 240.4 (logs/eval/).
@pytest.mark.parametrize("variant, seed", [("dqn_hard", 4), ("dqn_soft", 0)])
def test_shipped_dqn_solves(variant, seed):
    env = gym.make("LunarLander-v3")
    hp = load_config(os.path.join(ROOT, "configs", f"{variant}.yaml"))["hyperparameters"]
    agent = create_agent(variant, env, hp)
    assert load_model(agent, variant, os.path.join(ROOT, "models", f"{variant}_seed{seed}.pth"))
    assert mean_score(agent) >= 200


def test_heuristic_solves():
    assert mean_score(HeuristicAgent()) >= 200
