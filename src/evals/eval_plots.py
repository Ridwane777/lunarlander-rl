import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "q_learning":    "#9C27B0",
    "q_learning_v2": "#BA68C8",
    "reinforce":     "#FF9800",
    "reinforce_v2":  "#FFB74D",
    "dqn_soft":      "#64B5F6",
    "dqn_hard":      "#1E88E5",
    "random":        "#9E9E9E",
    "heuristic":     "#795548",
}
# Learning agents from simplest to most capable, then the baselines (no training).
ORDER = list(COLORS)
BASELINES = ("random", "heuristic")

DEFAULT_COLOR = "#607D8B"

def color_of(algo):
    return COLORS.get(algo, DEFAULT_COLOR)

def ordered(algos):
    return sorted(algos, key=lambda a: (ORDER.index(a) if a in ORDER else len(ORDER), a))

def plot_mean_ci(metrics, out_dir):
    algos  = ordered(metrics.keys())
    means  = [metrics[a]["mean_reward"] for a in algos]
    ci95s  = [metrics[a]["ci95"]        for a in algos]
    colors = [color_of(a) for a in algos]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(algos, means, yerr=ci95s, capsize=6,
                  color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    for bar, a in zip(bars, algos):
        if a in BASELINES:
            bar.set_hatch("//")
    n_learned = sum(a not in BASELINES for a in algos)
    if 0 < n_learned < len(algos):
        ax.axvline(n_learned - 0.5, color="grey", linestyle=":", linewidth=1)
        ax.text(n_learned - 0.4, ax.get_ylim()[0] + 10, "baselines (no training)",
                fontsize=8, color="grey", va="bottom")
    ax.axhline(200, color="green", linestyle="--", linewidth=1.5, label="Success threshold 200")
    ax.axhline(0,   color="black", linewidth=0.6)

    for bar, m, ci in zip(bars, means, ci95s):
        label = f"{m:.0f}±{ci:.0f}" if not math.isnan(ci) else f"{m:.0f}"
        ax.text(bar.get_x() + bar.get_width()/2, m + (ci if not math.isnan(ci) else 0) + 5,
                label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title("Mean reward ± 95% CI per agent (Evaluation)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean reward"); ax.set_xlabel("Agent")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    p = os.path.join(out_dir, "01_mean_reward_ci95.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  → {p}")

def plot_distribution(data, out_dir):
    algos    = ordered(data.keys())
    all_rew  = []
    labels   = []

    for algo in algos:
        rewards = []
        for rows in data[algo].values():
            rewards.extend(r["total_reward"] for r in rows)
        all_rew.append(rewards)
        labels.append(algo)

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(all_rew, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(color_of(algo))
        patch.set_alpha(0.75)

    ax.axhline(200, color="green", linestyle="--", linewidth=1.5, label="Seuil 200")
    ax.set_title("Reward distribution per agent (Evaluation)", fontsize=13)
    ax.set_ylabel("Total reward"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    p = os.path.join(out_dir, "03_reward_distribution.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  → {p}")

def plot_termination(metrics, out_dir):
    algos  = ordered(metrics.keys())
    causes = set()
    for m in metrics.values():
        causes.update(m["terminations"].keys())
    causes = sorted(causes)

    cause_colors = {
        "landed":  "#4CAF50",
        "crash":   "#F44336",
        "timeout": "#FF9800",
        "Unknown": "#9E9E9E",
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = [0] * len(algos)

    for cause in causes:
        vals = []
        for algo in algos:
            total = sum(metrics[algo]["terminations"].values())
            cnt   = metrics[algo]["terminations"].get(cause, 0)
            vals.append(cnt / max(total, 1) * 100)
        color = cause_colors.get(cause, "#607D8B")
        ax.bar(algos, vals, bottom=bottoms, label=cause, color=color, alpha=0.85)
        bottoms = [b+v for b,v in zip(bottoms, vals)]

    ax.set_title("Termination causes per agent (%)", fontsize=13)
    ax.set_ylabel("% episodes"); ax.set_ylim(0, 105)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    p = os.path.join(out_dir, "04_termination_causes.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  → {p}")

def plot_success_rate(metrics, out_dir):
    algos   = ordered(metrics.keys())
    rates   = [metrics[a]["success_rate"] for a in algos]
    colors  = [color_of(a) for a in algos]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(algos, rates, color=colors, alpha=0.85, edgecolor="white")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, rate + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Success rate - episodes with reward ≥ 200", fontsize=13)
    ax.set_ylabel("% episodes ≥ 200"); ax.set_ylim(0, max(rates or [10]) * 1.2 + 5)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    p = os.path.join(out_dir, "05_success_rate.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  → {p}")


def plot_episode_length(data, out_dir):
    """Boxplot of episode lengths per agent."""
    algos    = ordered(data.keys())
    all_lens = []
    labels   = []

    for algo in algos:
        lengths = []
        for rows in data[algo].values():
            lengths.extend(r["length"] for r in rows)
        all_lens.append(lengths)
        labels.append(algo)

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(all_lens, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(color_of(algo))
        patch.set_alpha(0.75)

    ax.set_title("Episode length distribution per agent", fontsize=13)
    ax.set_ylabel("Longueur (steps)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    p = os.path.join(out_dir, "06_episode_length.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  → {p}")
