import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _rolling(data, w=100):
    out = []
    for i in range(len(data)):
        s = max(0, i - w + 1)
        out.append(sum(data[s:i+1]) / (i - s + 1))
    return out


def _plot_seed(result, plot_dir):
    r = result["returns"]
    if len(r) < 2:
        return
    eps = list(range(len(r)))
    roll = _rolling(r)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eps, r, alpha=0.25, color="steelblue", linewidth=0.8)
    ax.plot(eps, roll, color="steelblue", linewidth=2, label="Rolling mean 100")
    ax.axhline(200, color="green", linestyle="--", linewidth=1.5, label="Seuil 200")
    if result["solved_at"] is not None:
        ax.axvline(result["solved_at"], color="orange", linestyle=":",
                   linewidth=1.5, label=f"Résolu ep.{result['solved_at']}")
    ax.set_title(f"{result['algo']} | seed={result['seed']}")
    ax.set_xlabel("Épisode"); ax.set_ylabel("Total reward")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, f"seed{result['seed']}_reward.png")
    fig.savefig(path, dpi=100); plt.close(fig)
    print(f"  → {path}")


def _plot_aggregate(results, plot_dir, algo):
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
    ci95 = [1.96*s/math.sqrt(n) for s in stds]
    roll = _rolling(means)
    eps  = list(range(min_l))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(eps, [m-c for m,c in zip(means,ci95)],
                         [m+c for m,c in zip(means,ci95)],
                    alpha=0.2, color="steelblue", label="IC 95%")
    ax.plot(eps, means, alpha=0.3, color="steelblue", linewidth=1)
    ax.plot(eps, roll, color="steelblue", linewidth=2.5,
            label=f"Rolling mean 100 ({n} seeds)")
    ax.axhline(200, color="green", linestyle="--", linewidth=1.5, label="Seuil 200")
    ax.set_title(f"{algo} — Agrégé {n} seeds (IC 95%)")
    ax.set_xlabel("Épisode"); ax.set_ylabel("Total reward")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, "aggregate.png")
    fig.savefig(path, dpi=100); plt.close(fig)
    print(f"  → {path}")
