import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class ReinforceAgent:
    def __init__(self, obs_dim, n_actions, config):
        self.gamma = config.get("gamma", 0.99)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.get("lr", 0.001))

        # Stockage pour un épisode
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.policy(obs_t)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        # Calcul des retours cumulés
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalisation pour stabiliser l'apprentissage
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calcul de la loss
        loss = []
        for log_prob, G in zip(self.log_probs, returns):
            loss.append(-log_prob * G)
        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset pour le prochain épisode
        self.log_probs = []
        self.rewards = []

        return loss.item()