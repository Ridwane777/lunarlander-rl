import argparse
import csv
import math
import os
from statistics import mean, median

import matplotlib.pyplot as plt


def load_log(path):
    episodes, rewards, lengths, causes = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
            lengths.append(int(row["length"]))
            causes.append(row["termination_cause"])
    return {
        "episodes": episodes,
        "rewards": rewards,
        "lengths": lengths,
        "causes": causes,
    }


def moving_average(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def rolling_std(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        m = sum(chunk) / len(chunk)
        var = sum((v - m) ** 2 for v in chunk) / len(chunk)
        out.append(math.sqrt(var))
    return out


def mean_100_curve(rewards):
    vals = []
    for i in range(len(rewards)):
        if i < 99:
            vals.append(None)
        else:
            vals.append(sum(rewards[i - 99 : i + 1]) / 100)
    return vals


def first_solved_episode(rewards, threshold=200):
    m100 = mean_100_curve(rewards)
    for i, v in enumerate(m100):
        if v is not None and v >= threshold:
            return i
    return None


def cause_distribution(causes):
    dist = {}
    for c in causes:
        dist[c] = dist.get(c, 0) + 1
    return dist


def build_summary(method_name, data):
    rewards = data["rewards"]
    lengths = data["lengths"]
    solved_ep = first_solved_episode(rewards)
    tail = rewards[-100:] if len(rewards) >= 100 else rewards
    causes = cause_distribution(data["causes"])
    landed = causes.get("landed", 0)
    crash = causes.get("crash", 0)
    timeout = causes.get("timeout", 0)

    return {
        "method": method_name,
        "episodes": len(rewards),
        "mean_reward": mean(rewards),
        "median_reward": median(rewards),
        "best_reward": max(rewards),
        "worst_reward": min(rewards),
        "std_reward": rolling_std(rewards, len(rewards))[-1],
        "mean_last_100": mean(tail),
        "solved_episode": solved_ep if solved_ep is not None else "not_solved",
        "mean_length": mean(lengths),
        "landed_rate": landed / len(rewards),
        "crash_rate": crash / len(rewards),
        "timeout_rate": timeout / len(rewards),
    }


def save_summary_csv(path, rows):
    headers = [
        "method",
        "episodes",
        "mean_reward",
        "median_reward",
        "best_reward",
        "worst_reward",
        "std_reward",
        "mean_last_100",
        "solved_episode",
        "mean_length",
        "landed_rate",
        "crash_rate",
        "timeout_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_method(arg):
    name, path = arg.split("=", 1)
    return name.strip(), path.strip()


def plot_rewards_with_trend(output_dir, dataset, window=20):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in dataset.items():
        eps = data["episodes"]
        rews = data["rewards"]
        ax.plot(eps, rews, alpha=0.18, linewidth=0.9, label=f"{name} (brut)")
        ax.plot(eps, moving_average(rews, window), linewidth=2.2, label=f"{name} (MA{window})")
    ax.axhline(200, color="#444", linestyle="--", linewidth=1.0, label="Seuil de résolution (200)")
    ax.set_title("Reward par épisode + tendance")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward total")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_rewards_trend.png"), dpi=160)
    plt.close(fig)


def plot_mean100(output_dir, dataset):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in dataset.items():
        m100 = mean_100_curve(data["rewards"])
        valid = [(e, v) for e, v in zip(data["episodes"], m100) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, linewidth=2.2, label=name)
    ax.axhline(200, color="#444", linestyle="--", linewidth=1.0, label="Seuil de résolution (200)")
    ax.set_title("Moyenne glissante sur 100 épisodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean reward (100 derniers épisodes)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_mean100.png"), dpi=160)
    plt.close(fig)


def plot_simple_method_comparison(output_dir, dataset):
    methods = list(dataset.keys())
    mean_last_100 = []
    landed_rate = []

    for method in methods:
        rewards = dataset[method]["rewards"]
        tail = rewards[-100:] if len(rewards) >= 100 else rewards
        mean_last_100.append(sum(tail) / len(tail))
        dist = cause_distribution(dataset[method]["causes"])
        landed_rate.append(100.0 * dist.get("landed", 0) / len(dataset[method]["causes"]))

    x = list(range(len(methods)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width / 2 for i in x], mean_last_100, width=width, label="Mean last 100")
    bars2 = ax.bar([i + width / 2 for i in x], landed_rate, width=width, label="Landed rate (%)")

    ax.axhline(200, color="#444", linestyle="--", linewidth=1.0, label="Seuil solve (200)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Valeur")
    ax.set_title("Comparaison simple des methodes")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_rewards_boxplot.png"), dpi=160)
    plt.close(fig)


def plot_termination_causes(output_dir, dataset):
    methods = list(dataset.keys())
    labels = ["landed", "crash", "timeout"]
    x = range(len(methods))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, cause in enumerate(labels):
        vals = []
        for method in methods:
            dist = cause_distribution(dataset[method]["causes"])
            total = len(dataset[method]["causes"])
            vals.append(100.0 * dist.get(cause, 0) / total)
        shift = (idx - 1) * width
        ax.bar([i + shift for i in x], vals, width=width, label=cause)
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Pourcentage d'episodes (%)")
    ax.set_title("Causes de fin d'episode")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_termination_causes.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare plusieurs methodes d'apprentissage a partir des logs CSV.")
    parser.add_argument(
        "--method",
        action="append",
        required=False,
        help='Format: "Nom=logs/fichier.csv". Peut etre repete 3+ fois.',
    )
    parser.add_argument("--output-dir", default="plots/method_comparison", help="Dossier de sortie des figures.")
    args = parser.parse_args()

    default_methods = [
        "Q-Learning=logs/q_learning.csv",
        "REINFORCE=logs/reinforce.csv",
        "DQN (soft)=logs/dqn_soft.csv",
    ]
    methods = args.method if args.method else default_methods

    dataset = {}
    for m in methods:
        name, path = parse_method(m)
        dataset[name] = load_log(path)

    os.makedirs(args.output_dir, exist_ok=True)

    summaries = [build_summary(name, data) for name, data in dataset.items()]
    summaries.sort(key=lambda x: x["mean_last_100"], reverse=True)

    plot_rewards_with_trend(args.output_dir, dataset, window=20)
    plot_mean100(args.output_dir, dataset)
    plot_simple_method_comparison(args.output_dir, dataset)
    plot_termination_causes(args.output_dir, dataset)
    os.makedirs("logs", exist_ok=True)
    summary_path = os.path.join("logs", "summary_metrics.csv")
    save_summary_csv(summary_path, summaries)

    print("\n=== Classement (mean_last_100) ===")
    for i, row in enumerate(summaries, start=1):
        print(
            f"{i}. {row['method']:<14} | mean_last_100={row['mean_last_100']:.2f} | "
            f"solved_episode={row['solved_episode']} | landed_rate={100*row['landed_rate']:.1f}%"
        )
    print(f"\nFigures sauvegardees dans: {args.output_dir}")
    print(f"Tableau de metriques sauvegarde dans: {summary_path}")


if __name__ == "__main__":
    main()
