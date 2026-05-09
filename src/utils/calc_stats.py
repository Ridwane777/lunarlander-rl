import pandas as pd
import glob
import os

logs = sorted(glob.glob("logs/train/dqn_soft_seed*.csv"))
for log in logs:
    df = pd.read_csv(log)
    mean_r = df['total_reward'].mean()
    max_r = df['total_reward'].max()
    episodes = len(df)
    seed = os.path.basename(log).split("seed")[1].split(".")[0]
    print(f"Seed {seed}: {episodes} eps, Mean {mean_r:.1f}, Max {max_r:.1f}")
