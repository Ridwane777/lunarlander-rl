import csv
import math

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "episode": int(row["episode"]),
                "total_reward": float(row["total_reward"]),
                "length": int(row["length"]),
                "termination_cause": row["termination_cause"]
            })
    return rows

def mean(values):
    return sum(values) / len(values)

def std(values):
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))

def ci95(values):
    return 1.96 * std(values) / math.sqrt(len(values))

def summarize(name, data):
    rewards = [r["total_reward"] for r in data]
    lengths = [r["length"] for r in data]
    causes = {}
    for r in data:
        c = r["termination_cause"]
        causes[c] = causes.get(c, 0) + 1

    above_200 = [r for r in rewards if r >= 200]

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Episodes total     : {len(data)}")
    print(f"  Reward moyen       : {mean(rewards):.1f} +/- {ci95(rewards):.1f} (95% CI)")
    print(f"  Reward max         : {max(rewards):.1f}")
    print(f"  Reward min         : {min(rewards):.1f}")
    print(f"  Std deviation      : {std(rewards):.1f}")
    print(f"  Episodes >= 200    : {len(above_200)} ({100*len(above_200)/len(data):.1f}%)")
    print(f"  Longueur moyenne   : {mean(lengths):.0f} steps")
    print(f"  Causes de fin      :")
    for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
        print(f"    {cause:<15} : {count} ({100*count/len(data):.1f}%)")

    # Progression par tranches de 25 episodes
    print(f"\n  Progression (moyenne par tranche de 25 episodes) :")
    tranche = 25
    for i in range(0, len(rewards), tranche):
        chunk = rewards[i:i+tranche]
        print(f"    Ep {i:>3}-{min(i+tranche-1, len(rewards)-1):<3} : {mean(chunk):>8.1f}")

hard = load_csv("logs/dqn_hard.csv")
soft = load_csv("logs/dqn_soft.csv")

summarize("DQN — Hard Update", hard)
summarize("DQN — Soft Update", soft)

# Comparaison directe
print(f"\n{'='*50}")
print(f"  COMPARAISON DIRECTE")
print(f"{'='*50}")
hard_rewards = [r["total_reward"] for r in hard]
soft_rewards = [r["total_reward"] for r in soft]
print(f"  Reward moyen Hard  : {mean(hard_rewards):.1f}")
print(f"  Reward moyen Soft  : {mean(soft_rewards):.1f}")
print(f"  Meilleur           : {'Soft Update' if mean(soft_rewards) > mean(hard_rewards) else 'Hard Update'}")
print(f"  Episodes >= 200 Hard : {sum(1 for r in hard_rewards if r >= 200)}")
print(f"  Episodes >= 200 Soft : {sum(1 for r in soft_rewards if r >= 200)}")