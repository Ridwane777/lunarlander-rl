import csv
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ──────────────────────────────────────────────
# Chargement des CSV
# ──────────────────────────────────────────────

def load_csv(path):
    episodes, rewards = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
    return episodes, rewards


def moving_average(values, window=20):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


def mean_100(rewards, ep):
    if ep < 100:
        return None
    return sum(rewards[ep - 99: ep + 1]) / 100


def std(values):
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def ci95(values):
    return 1.96 * std(values) / math.sqrt(len(values))


# ──────────────────────────────────────────────
# Paths — adapte si besoin
# ──────────────────────────────────────────────

HARD_CSV = "logs/dqn_hard.csv"
SOFT_CSV = "logs/dqn_soft.csv"

eps_hard, rew_hard = load_csv(HARD_CSV)
eps_soft, rew_soft = load_csv(SOFT_CSV)

ma_hard = moving_average(rew_hard, window=20)
ma_soft = moving_average(rew_soft, window=20)

# ──────────────────────────────────────────────
# Statistiques console
# ──────────────────────────────────────────────

def print_stats(name, rewards):
    m = sum(rewards) / len(rewards)
    ci = ci95(rewards)
    print(f"\n── {name} ──")
    print(f"  Episodes       : {len(rewards)}")
    print(f"  Reward moyen   : {m:.1f} ± {ci:.1f} (95% CI)")
    print(f"  Reward max     : {max(rewards):.1f}")
    print(f"  Reward min     : {min(rewards):.1f}")
    print(f"  Ecart-type     : {std(rewards):.1f}")
    landed = sum(1 for r in rewards if r >= 200)
    print(f"  Episodes >= 200: {landed} ({100*landed/len(rewards):.1f}%)")

print_stats("DQN Hard Update", rew_hard)
print_stats("DQN Soft Update", rew_soft)

# ──────────────────────────────────────────────
# Figure 1 — Reward par épisode + moyenne mobile
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("Ablation DQN — Hard Update vs Soft Update", fontsize=14, fontweight="bold")

ALPHA_RAW = 0.25
COLOR_HARD = "#3266AD"
COLOR_SOFT = "#D85A30"

for ax, eps, rews, ma, color, label, solved_ep in [
    (axes[0], eps_hard, rew_hard, ma_hard, COLOR_HARD, "Hard Update", 301),
    (axes[1], eps_soft, rew_soft, ma_soft, COLOR_SOFT, "Soft Update", 186),
]:
    ax.plot(eps, rews, color=color, alpha=ALPHA_RAW, linewidth=0.8, label="Reward brut")
    ax.plot(eps, ma, color=color, linewidth=2.0, label="Moyenne mobile (20 ep)")
    ax.axhline(200, color="#444", linewidth=1.0, linestyle="--", label="Critère (200)")
    ax.axvline(solved_ep, color="#888", linewidth=1.0, linestyle=":", label=f"Résolu ep. {solved_ep}")
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Reward total")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/ablation_reward_per_episode.png", dpi=150, bbox_inches="tight")
print("\nSauvegardé : plots/ablation_reward_per_episode.png")

# ──────────────────────────────────────────────
# Figure 2 — Moyenne glissante 100 épisodes (critère officiel)
# ──────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.set_title("Moyenne glissante sur 100 épisodes (critère officiel)", fontsize=13)

for eps, rews, color, label in [
    (eps_hard, rew_hard, COLOR_HARD, "Hard Update"),
    (eps_soft, rew_soft, COLOR_SOFT, "Soft Update"),
]:
    mean100 = [mean_100(rews, i) for i in range(len(rews))]
    valid_eps = [e for e, m in zip(eps, mean100) if m is not None]
    valid_m   = [m for m in mean100 if m is not None]
    ax2.plot(valid_eps, valid_m, color=color, linewidth=2.0, label=label)

ax2.axhline(200, color="#444", linewidth=1.0, linestyle="--", label="Critère (200)")
ax2.set_xlabel("Épisode")
ax2.set_ylabel("Mean reward (100 derniers épisodes)")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/ablation_mean100.png", dpi=150, bbox_inches="tight")
print("Sauvegardé : plots/ablation_mean100.png")

# ──────────────────────────────────────────────
# Figure 3 — Barplot comparatif par tranches
# ──────────────────────────────────────────────

def tranche_mean(rewards, start, end):
    sub = rewards[start:min(end, len(rewards))]
    if not sub:
        return 0
    return sum(sub) / len(sub)

tranches = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150)]
labels_t  = ["0-24", "25-49", "50-74", "75-99", "100-124", "125-149"]

means_hard = [tranche_mean(rew_hard, s, e) for s, e in tranches]
means_soft = [tranche_mean(rew_soft, s, e) for s, e in tranches]

x = range(len(labels_t))
width = 0.35

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.set_title("Reward moyen par tranche de 25 épisodes", fontsize=13)
bars_hard = ax3.bar([i - width/2 for i in x], means_hard, width, label="Hard Update", color=COLOR_HARD, alpha=0.85)
bars_soft = ax3.bar([i + width/2 for i in x], means_soft, width, label="Soft Update", color=COLOR_SOFT, alpha=0.85)
ax3.axhline(0,   color="#888", linewidth=0.8, linestyle="-")
ax3.axhline(200, color="#444", linewidth=1.0, linestyle="--", label="Critère (200)")
ax3.set_xticks(list(x))
ax3.set_xticklabels(labels_t)
ax3.set_xlabel("Tranche d'épisodes")
ax3.set_ylabel("Reward moyen")
ax3.legend(fontsize=10)
ax3.grid(True, axis="y", alpha=0.3)

for bar in bars_hard:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
             f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8, color=COLOR_HARD)
for bar in bars_soft:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
             f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8, color=COLOR_SOFT)

plt.tight_layout()
plt.savefig("plots/ablation_tranches.png", dpi=150, bbox_inches="tight")
print("Sauvegardé : plots/ablation_tranches.png")