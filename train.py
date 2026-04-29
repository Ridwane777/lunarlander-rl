"""
train.py — LunarLander RLLunarLander-v3
=================================================
Entraîne tous les agents définis dans configs/ sur plusieurs seeds.

Usage:
    python train.py                                          # tous les agents
    python train.py --config configs/dqn_soft.yaml           # un seul agent
    python train.py --config configs/dqn_soft.yaml --seed 0  # seed précise
    python train.py --list                                   # liste les configs

Sorties:
    logs/<algo>_seed<N>.csv    — CSV épisodique
    plots/<algo>/              — courbes reward + agrégé multi-seeds
    models/<algo>_seed<N>.pth  — modèle entraîné
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
    "reinforce":  train_reinforce,
    "q_learning": train_q_learning,
    "random":     train_baseline,
    "heuristic":  train_baseline,
}


def run_config(config_path, seed_override=None):
    cfg   = load_config(config_path)
    algo  = cfg["algo"]
    seeds = [seed_override] if seed_override is not None else cfg["seeds"]

    if algo not in TRAINERS:
        print(f"[ERREUR] algo '{algo}' inconnu. Disponibles : {list(TRAINERS.keys())}")
        return []

    trainer  = TRAINERS[algo]
    plot_dir = make_plot_dir(algo)
    results  = []

    for seed in seeds:
        log_path = make_log_path(algo, seed)
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

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Entraîne les agents LunarLander depuis les configs YAML."
    )
    parser.add_argument("--config", "-c",
        help="Un fichier YAML précis. Absent = tous les fichiers de configs/.")
    parser.add_argument("--seed", "-s", type=int, default=None,
        help="Force une seed unique.")
    parser.add_argument("--list", "-l", action="store_true",
        help="Liste les configs disponibles et quitte.")
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
        print(f"Aucun YAML trouvé dans {configs_dir}/"); sys.exit(1)

    all_results = []
    for cfg_path in config_files:
        all_results.extend(run_config(cfg_path, seed_override=args.seed))

    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    main()
