"""
play.py - watch or record the trained agent land, no training needed
====================================================================
Loads shipped DQN weights (models/*.pth) and plays greedy episodes.

Usage:
    python play.py                                  # opens a window, 3 landings
    python play.py --episodes 5 --seed 42
    python play.py --record landing.gif             # saves a GIF (or .mp4) instead
    python play.py --model models/dqn_soft_seed0.pth
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.agents.dqn_agent import DQNAgent
from src.utils.core import load_config
from src.utils.model_utils import load_model

DEFAULT_MODEL = os.path.join(ROOT, "models", "dqn_hard_seed4.pth")   # best seed: 249 mean, 96% success


def save_frames(frames, path, fps=30):
    from moviepy import ImageSequenceClip   # imported here: only needed to record
    clip = ImageSequenceClip(frames, fps=fps)
    if path.endswith(".gif"):
        clip.write_gif(path, fps=fps, logger=None)
    else:
        clip.write_videofile(path, fps=fps, logger=None)


def main():
    parser = argparse.ArgumentParser(description="Watch or record the trained DQN agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DQN weights (.pth)")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "dqn_hard.yaml"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=26, help="first episode seed (26: three clean landings)")
    parser.add_argument("--record", help="save to this .gif or .mp4 instead of opening a window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = gym.make(cfg["env"], render_mode="rgb_array" if args.record else "human")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg["hyperparameters"])
    algo = "dqn_soft" if "soft" in os.path.basename(args.model) else "dqn_hard"
    if not load_model(agent, algo, args.model):
        sys.exit(1)

    frames, scores = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.select_action(obs))
            total += reward
            done = terminated or truncated
            if args.record:
                frames.append(env.render())
        scores.append(total)
        print(f"episode {ep + 1}: score {total:7.1f}  {'landed' if total >= 200 else 'not solved'}")
    env.close()

    print(f"mean score over {args.episodes} episodes: {np.mean(scores):.1f} (solved at 200)")
    if args.record:
        save_frames(frames, args.record)
        print(f"saved {args.record}")


if __name__ == "__main__":
    main()
