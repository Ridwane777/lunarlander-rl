import random
import torch
import numpy as np
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
            torch.tensor(np.array(states),      dtype=torch.float32).to(self.device),
            torch.tensor(np.array(actions),     dtype=torch.int64).to(self.device),
            torch.tensor(np.array(rewards),     dtype=torch.float32).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(dones),       dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)