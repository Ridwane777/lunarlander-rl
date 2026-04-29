import csv
import os
import re
import math

def parse_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eps  = row.get("epsilon",  "") or None
            loss = row.get("loss_mean","") or None
            rows.append({
                "episode":           int(row["episode"]),
                "total_reward":      float(row["total_reward"]),
                "length":            int(row["length"]),
                "termination_cause": row["termination_cause"].strip(),
                "epsilon":           float(eps)  if eps  else None,
                "loss_mean":         float(loss) if loss else None,
            })
    return rows

def algo_from_path(path):
    name = os.path.basename(path)
    name = re.sub(r"_seed\d+\.(csv|pth)$", "", name)
    name = re.sub(r"\.(csv|pth)$", "", name)
    return name

def load_all(log_paths):
    data = {}
    for path in log_paths:
        if not os.path.exists(path):
            continue
        algo = algo_from_path(path)
        m = re.search(r"seed(\d+)", os.path.basename(path))
        seed = int(m.group(1)) if m else 0
        rows = parse_csv(path)
        data.setdefault(algo, {})[seed] = rows
    return data

def compute_metrics(data):
    metrics = {}
    for algo, seeds in data.items():
        all_rewards = []
        terminations = {}

        for seed, rows in seeds.items():
            rewards = [r["total_reward"] for r in rows]
            all_rewards.extend(rewards)

            for r in rows:
                c = r["termination_cause"]
                terminations[c] = terminations.get(c, 0) + 1

        n = len(all_rewards)
        mu = sum(all_rewards) / max(n, 1)
        std = math.sqrt(sum((x-mu)**2 for x in all_rewards) / max(n-1, 1))
        ci95 = 1.96 * std / math.sqrt(n) if n > 1 else float("nan")

        metrics[algo] = {
            "mean_reward":   mu,
            "std":           std,
            "ci95":          ci95,
            "max_reward":    max(all_rewards) if all_rewards else float("-inf"),
            "success_rate":  sum(1 for r in all_rewards if r >= 200) / max(n, 1) * 100,
            "n_episodes":    n,
            "n_seeds":       len(seeds),
            "terminations":  terminations,
        }
    return metrics
