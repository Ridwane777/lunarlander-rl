import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def _rolling(data, w=100):
    """Compute the moving average over a window of size w."""
    out = []
    for i in range(len(data)):
        s = max(0, i - w + 1)
        out.append(sum(data[s:i+1]) / (i - s + 1))
    return out


def _plot_seed(result, plot_dir):
    """Plot the reward curve for a single seed."""
    r = result["returns"]
    if len(r) < 2:
        return
    eps = list(range(len(r)))
    roll = _rolling(r)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps, r, alpha=0.25, color="steelblue", linewidth=0.8)
    ax.plot(eps, roll, color="steelblue", linewidth=2, label="Moyenne mobile (100)")
    ax.axhline(200, color="green", linestyle="--", linewidth=1.5, label="Objectif 200")
    if result["solved_at"] is not None:
        ax.axvline(result["solved_at"], color="orange", linestyle=":",
                   linewidth=1.5, label=f"Solved ep.{result['solved_at']}")
    ax.set_title(f"{result['algo']} | seed={result['seed']}")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, f"seed{result['seed']}_reward.png")
    fig.savefig(path, dpi=100); plt.close(fig)
    print(f"  → {path}")


def _plot_aggregate(results, plot_dir, algo):
    """Plot the aggregate over all seeds with a 95% confidence interval."""
    all_r = [r["returns"] for r in results]
    n     = len(all_r)
    min_l = min(len(r) for r in all_r)
    if min_l < 2:
        return

    trimmed = [r[:min_l] for r in all_r]
    means = [sum(col)/n for col in zip(*trimmed)]
    stds  = []
    for col in zip(*trimmed):
        mu  = sum(col)/n
        var = sum((x-mu)**2 for x in col) / max(n-1, 1)
        stds.append(math.sqrt(var))
    
    # 95% CI: 1.96 * std / sqrt(n)
    ci95 = [1.96*s/math.sqrt(n) for s in stds]
    roll = _rolling(means)
    eps  = list(range(min_l))

    fig, ax = plt.subplots(figsize=(12, 5))
    # Shaded area for the 95% CI
    ax.fill_between(eps, [m-c for m,c in zip(means,ci95)],
                         [m+c for m,c in zip(means,ci95)],
                    alpha=0.2, color="steelblue", label="IC 95% (Robustesse)")
    ax.plot(eps, means, alpha=0.3, color="steelblue", linewidth=1)
    ax.plot(eps, roll, color="steelblue", linewidth=2.5,
            label=f"Aggregated mean ({n} seeds)")
    ax.axhline(200, color="green", linestyle="--", linewidth=1.5, label="Objectif 200")
    ax.set_title(f"{algo} - Aggregated benchmark, {n} seeds")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, "aggregate.png")
    fig.savefig(path, dpi=100); plt.close(fig)
    print(f"  → {path}")

def _plot_training_stats(algo, log_paths, plot_dir):
    """Generate detailed training plots (distribution, success, causes)."""
    all_data = []
    seeds = []
    for path in log_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_data.append(df)
            # Extract seed from path (e.g., logs/train/dqn_seed0.csv)
            seed = path.split("seed")[-1].split(".")[0]
            seeds.append(f"Seed {seed}")

    if not all_data:
        return

    # 1. Reward distribution (boxplot)
    fig, ax = plt.subplots(figsize=(10, 5))
    rewards = [df['total_reward'].tolist() for df in all_data]
    ax.boxplot(rewards, tick_labels=seeds, patch_artist=True,
               medianprops=dict(color="black", linewidth=2))
    ax.axhline(200, color="green", linestyle="--", alpha=0.6)
    ax.set_title(f"Training reward distribution - {algo}")
    ax.set_ylabel("Total Reward")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "training_reward_dist.png"), dpi=110)
    plt.close(fig)

    # 2. Success rate (episodes >= 200)
    fig, ax = plt.subplots(figsize=(10, 4))
    success_rates = [(df['total_reward'] >= 200).mean() * 100 for df in all_data]
    bars = ax.bar(seeds, success_rates, color="skyblue", alpha=0.8, edgecolor="navy")
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 0.5, f"{rate:.1f}%", ha='center', fontsize=9)
    ax.set_title(f"Success rate during training - {algo}")
    ax.set_ylabel("% episodes ≥ 200")
    ax.set_ylim(0, max(success_rates + [10]) * 1.2)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "training_success_rate.png"), dpi=110)
    plt.close(fig)

    # 3. Causes de Terminaison
    fig, ax = plt.subplots(figsize=(10, 5))
    cause_colors = {"landed": "#4CAF50", "crash": "#F44336", "timeout": "#FF9800"}
    
    bottoms = np.zeros(len(seeds))
    for cause, color in cause_colors.items():
        vals = []
        for df in all_data:
            # Look at terminated, truncated, etc., or the 'termination_cause' column if present
            if 'termination_cause' in df.columns:
                count = (df['termination_cause'] == cause).sum()
            else:
                # Fall back to the booleans
                if cause == "landed": count = df['terminated'].sum()
                elif cause == "crash": count = (df['terminated'] == False).sum() # Simpliste
                else: count = 0
            vals.append(count / len(df) * 100)
        
        ax.bar(seeds, vals, bottom=bottoms, label=cause, color=color, alpha=0.8)
        bottoms += np.array(vals)

    ax.set_title(f"Termination breakdown - {algo}")
    ax.set_ylabel("% episodes")
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "training_terminations.png"), dpi=110)
    plt.close(fig)
    
    print(f"  → Detailed training plots written to {plot_dir}")
