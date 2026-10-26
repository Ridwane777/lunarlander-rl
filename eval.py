"""
eval.py — LunarLander-v3
================================================
Evaluates trained .pth models in a test environment
and builds a comparative benchmark from those evaluations.

Usage:
    python eval.py                      # all models available in models/
    python eval.py --models models/dqn_soft_seed*.pth
    python eval.py --out plots/benchmark
"""

import argparse
import glob
import os
import sys
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.evals.eval_core import evaluate_model
from src.evals.eval_metrics import algo_from_path, load_all, compute_metrics
from src.evals.eval_plots import (
    plot_mean_ci,
    plot_distribution,
    plot_termination,
    plot_success_rate,
    plot_episode_length
)
from src.evals.eval_utils import save_summary_csv, print_verdict

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate .pth models and build the benchmark."
    )
    parser.add_argument("--models", nargs="*",
        help=".pth paths to evaluate. Default: all models/*.pth")
    parser.add_argument("--n_episodes", type=int, default=100,
        help="Number of test episodes per model.")
    parser.add_argument("--out", default="plots/benchmark",
        help="Output directory for plots (default: plots/benchmark)")
    parser.add_argument("--plot-only", action="store_true",
        help="Skip evaluation; only regenerate plots from logs/eval/")
    args = parser.parse_args()

    eval_logs_paths = []

    if args.plot_only:
        print("\n[PLOT ONLY] Plot-only mode enabled.")
        eval_logs_paths = sorted(glob.glob(os.path.join(ROOT, "logs", "eval", "*.csv")))
        if not eval_logs_paths:
            print("No evaluation logs found in logs/eval/. Run an evaluation first.")
            sys.exit(1)
        print(f"Found {len(eval_logs_paths)} evaluation log files.")
    else:
        if args.models:
            model_paths = args.models
        else:
            model_paths = sorted(glob.glob(os.path.join(ROOT, "models", "*.pth")))

        if not model_paths:
            print("No model (.pth) found. Run training first (python train.py).")
            sys.exit(1)

        print(f"\nModels to evaluate ({len(model_paths)}) :")
        for p in model_paths:
            print(f"  {p}")

        # Evaluation loop
        for path in model_paths:
            algo = algo_from_path(path)
            m = re.search(r"seed(\d+)", os.path.basename(path))
            seed = int(m.group(1)) if m else 0
            
            config_path = os.path.join(ROOT, "configs", f"{algo}.yaml")
            if not os.path.exists(config_path):
                if algo == "dqn":
                    config_path = os.path.join(ROOT, "configs", "dqn_hard.yaml")
                if not os.path.exists(config_path):
                    print(f"[WARNING] Config not found for {path}, skipping.")
                    continue
                
            log_path = evaluate_model(path, config_path, algo, seed, n_episodes=args.n_episodes)
            if log_path:
                eval_logs_paths.append(log_path)

    if not eval_logs_paths:
        print("No evaluation succeeded.")
        sys.exit(1)

    # Analysis and plots
    data    = load_all(eval_logs_paths)
    metrics = compute_metrics(data)
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGenerating plots in {out_dir}/")
    plot_mean_ci(metrics, out_dir)
    plot_distribution(data, out_dir)
    plot_termination(metrics, out_dir)
    plot_success_rate(metrics, out_dir)
    plot_episode_length(data, out_dir)
    save_summary_csv(metrics, out_dir)

    print_verdict(metrics)

if __name__ == "__main__":
    main()
