import os
import yaml
import math
import random
import torch

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def make_log_path(algo, seed):
    os.makedirs("logs/train", exist_ok=True)
    return f"logs/train/{algo}_seed{seed}.csv"

def make_plot_dir(algo):
    d = f"plots/{algo}"
    os.makedirs(d, exist_ok=True)
    return d

def _result(algo, seed, ep_returns, solved_at, elapsed):
    return {
        "algo": algo, "seed": seed,
        "episodes":    len(ep_returns),
        "mean_reward": sum(ep_returns) / max(len(ep_returns), 1),
        "max_reward":  max(ep_returns) if ep_returns else float("-inf"),
        "solved_at":   solved_at,
        "elapsed_s":   elapsed,
        "returns":     ep_returns,
    }

def print_summary(all_results):
    print("\n" + "="*72)
    print(f"{'AGENT':<18} {'N':>4} {'MEAN_R':>9} {'±CI95':>8} "
          f"{'MAX_R':>9} {'SOLVED':>10}")
    print("="*72)

    by_algo = {}
    for r in all_results:
        by_algo.setdefault(r["algo"], []).append(r)

    for algo, runs in sorted(by_algo.items()):
        means = [r["mean_reward"] for r in runs]
        n     = len(means)
        mu    = sum(means)/n
        std   = math.sqrt(sum((m-mu)**2 for m in means)/max(n-1,1))
        ci    = 1.96*std/math.sqrt(n) if n > 1 else float("nan")
        mx    = max(r["max_reward"] for r in runs)
        sol   = sum(1 for r in runs if r["solved_at"] is not None)
        print(f"{algo:<18} {n:>4} {mu:>9.1f} {ci:>8.1f} {mx:>9.1f} {sol:>3}/{n} seeds")

    print("="*72 + "\n")
