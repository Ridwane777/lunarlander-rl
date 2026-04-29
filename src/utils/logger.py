import csv
import os

class EpisodeLogger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file = open(log_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["episode", "total_reward", "length", "termination_cause", "epsilon", "loss_mean"]
        )

    def log(self, episode, reward, length, terminated, truncated, info,
            epsilon=None, loss=None):
        if truncated:
            cause = "timeout"
        elif terminated:
            cause = "crash" if reward < 0 else "landed"
        else:
            cause = "Unknown"

        eps_val  = round(epsilon, 4) if epsilon  is not None else ""
        loss_val = round(loss,    4) if loss     is not None else ""

        self.writer.writerow([episode, round(reward, 2), length, cause, eps_val, loss_val])
        self.file.flush()

    def close(self):
        self.file.close()