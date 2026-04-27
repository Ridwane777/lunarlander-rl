import random
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states,      dtype=torch.float32).to(self.device),
            torch.tensor(actions,     dtype=torch.int64).to(self.device),
            torch.tensor(rewards,     dtype=torch.float32).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(dones,       dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)