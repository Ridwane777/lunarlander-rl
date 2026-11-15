"""
train.py - LunarLander-v3
=================================================
Trains every agent defined in configs/ over several seeds.

Usage:
    python train.py                                          # all agents
    python train.py --config configs/dqn_soft.yaml           # a single agent
    python train.py --config configs/dqn_soft.yaml --seed 0  # a specific seed
    python train.py --list                                   # list configs

Outputs:
    logs/<algo>_seed<N>.csv    - per-episode CSV
    plots/<algo>/              - reward curves + multi-seed aggregate
    models/<algo>_seed<N>.pth  - trained model
"""

import argparse
import os
import sys
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils.core import load_config, print_summary, make_log_path, make_plot_dir
from src.utils.plots import _plot_seed, _plot_aggregate
from src.utils.model_utils import save_model
from src.utils.video_utils import concat_episode_videos

from src.trains.train_dqn import train_dqn
from src.trains.train_reinforce import train_reinforce
from src.trains.train_q_learning import train_q_learning
from src.trains.train_baseline import train_baseline

TRAINERS = {
    "dqn":        train_dqn,
    "dqn_soft":   train_dqn,
    "dqn_hard":   train_dqn,
    "reinforce":     train_reinforce,
    "reinforce_v2":  train_reinforce,
    "q_learning":    train_q_learning,
    "q_learning_v2": train_q_learning,
    "random":        train_baseline,
    "heuristic":     train_baseline,
}


def run_config(config_path, seed_override=None):
    cfg   = load_config(config_path)
    algo  = cfg["algo"]
    seeds = [seed_override] if seed_override is not None else cfg["seeds"]

    if algo not in TRAINERS:
        print(f"[ERROR] unknown algo '{algo}'. Available: {list(TRAINERS.keys())}")
        return []

    trainer  = TRAINERS[algo]
    plot_dir = make_plot_dir(algo)
    results  = []
    log_paths = []

    for seed in seeds:
        log_path = make_log_path(algo, seed)
        log_paths.append(log_path)
        print(f"\n{'='*60}")
        print(f"  {algo.upper()} | seed={seed} | config={os.path.basename(config_path)}")
        print(f"{'='*60}")

        result, agent = trainer(cfg, seed, log_path)
        results.append(result)
        
        # Save plot, model and merge episode videos
        _plot_seed(result, plot_dir)
        save_model(agent, algo, seed)
        concat_episode_videos(f"videos/train/{algo}_seed{seed}")

        print(f"\n→ {result['episodes']} ep | mean={result['mean_reward']:.1f} | "
              f"max={result['max_reward']:.1f} | solved={result['solved_at']} | "
              f"{result['elapsed_s']:.0f}s")

    if len(results) > 1:
        _plot_aggregate(results, plot_dir, algo)
        from src.utils.plots import _plot_training_stats
        _plot_training_stats(algo, log_paths, plot_dir)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train LunarLander agents from YAML configs."
    )
    parser.add_argument("--config", "-c",
        help="A specific YAML file. Default: every file in configs/.")
    parser.add_argument("--seed", "-s", type=int, default=None,
        help="Force a single seed.")
    parser.add_argument("--list", "-l", action="store_true",
        help="List available configs and exit.")
    parser.add_argument("--plot-only", action="store_true",
        help="Skip training; only regenerate plots from logs/train/")
    args = parser.parse_args()

    configs_dir = os.path.join(ROOT, "configs")

    if args.list:
        for y in sorted(glob.glob(os.path.join(configs_dir, "*.yaml"))):
            cfg = load_config(y)
            print(f"  {os.path.basename(y):<30} algo={cfg['algo']:<12} seeds={cfg['seeds']}")
        return

    config_files = ([args.config] if args.config
                    else sorted(glob.glob(os.path.join(configs_dir, "*.yaml"))))

    if not config_files:
        print(f"No YAML found in {configs_dir}/"); sys.exit(1)

    if args.plot_only:
        print("\n[PLOT ONLY] Regenerating training plots...")
        import pandas as pd
        import re
        from src.utils.plots import _plot_aggregate, _plot_training_stats
        
        for cfg_path in config_files:
            cfg = load_config(cfg_path)
            algo = cfg["algo"]
            plot_dir = make_plot_dir(algo)
            log_pattern = os.path.join(ROOT, "logs", "train", f"{algo}_seed*.csv")
            log_paths = sorted(glob.glob(log_pattern))
            
            if not log_paths:
                continue
                
            print(f"  → {algo} ({len(log_paths)} seeds)")
            fake_results = []
            for lp in log_paths:
                df = pd.read_csv(lp)
                m = re.search(r"seed(\d+)", lp)
                s = int(m.group(1)) if m else 0
                fake_results.append({"algo": algo, "seed": s, "returns": df["total_reward"].tolist()})
            
            if fake_results:
                if len(fake_results) > 1:
                    _plot_aggregate(fake_results, plot_dir, algo)
                _plot_training_stats(algo, log_paths, plot_dir)
        return

    all_results = []
    for cfg_path in config_files:
        all_results.extend(run_config(cfg_path, seed_override=args.seed))

    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    main()
