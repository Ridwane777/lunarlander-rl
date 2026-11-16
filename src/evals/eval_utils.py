import os
import csv
import math

BASELINES = ("random", "heuristic")   # no training: reference points, not competitors

def save_summary_csv(metrics, out_dir):
    path = os.path.join(out_dir, "benchmark_summary.csv")
    fieldnames = ["algo", "n_seeds", "n_episodes", "mean_reward", "ci95",
                  "max_reward", "success_rate_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for algo in sorted(metrics.keys()):
            m = metrics[algo]
            w.writerow({
                "algo":             algo,
                "n_seeds":          m["n_seeds"],
                "n_episodes":       m["n_episodes"],
                "mean_reward":      round(m["mean_reward"], 2),
                "ci95":             round(m["ci95"], 2) if not math.isnan(m["ci95"]) else "N/A",
                "max_reward":       round(m["max_reward"], 2),
                "success_rate_pct": round(m["success_rate"], 2),
            })
    print(f"  → {path}")
    return path

def print_verdict(metrics):
    print("\n" + "="*80)
    print(f"{'AGENT':<18} {'SEEDS':>5} {'MEAN_R':>9} {'±CI95':>8} {'SUCCESS%':>9}")
    print("="*80)

    for algo in sorted(metrics.keys()):
        m = metrics[algo]
        ci = f"{m['ci95']:.1f}" if not math.isnan(m["ci95"]) else " N/A"
        print(f"{algo:<18} {m['n_seeds']:>5} {m['mean_reward']:>9.1f} {ci:>8} {m['success_rate']:>8.1f}%")

    print("="*80)
    
    learned = {a: m for a, m in metrics.items() if a not in BASELINES}
    if learned:
        best_algo, best_m = max(learned.items(), key=lambda x: x[1]["mean_reward"])
        print(f"\n BEST LEARNED AGENT: {best_algo.upper()}")
        print(f"   mean_reward  = {best_m['mean_reward']:.1f} ± {best_m['ci95']:.1f} (IC 95%)")
        print(f"   success_rate = {best_m['success_rate']:.1f}%")
    print()
