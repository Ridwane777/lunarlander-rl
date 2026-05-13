import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """Réseau de neurones représentant la politique stochastique π_θ(a|s)."""
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
        # Utilisation de Softmax pour obtenir une distribution de probabilité sur les actions
        return torch.softmax(self.net(x), dim=-1)


class ReinforceAgent:
    """Agent implémentant l'algorithme REINFORCE (Vanilla Policy Gradient).
    Méthode On-Policy basée sur des trajectoires complètes (Monte Carlo).
    """
    def __init__(self, obs_dim, n_actions, config):
        self.gamma = config.get("gamma", 0.99)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(obs_dim, n_actions, hidden=config.get("hidden", 128)).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.get("lr", 0.001))

        # Stockage temporaire pour collecter une trajectoire (un épisode complet)
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        """Échantillonnage d'une action à partir de la distribution apprise par le réseau."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.policy(obs_t)
        
        # On crée une distribution catégorielle pour pouvoir échantillonner et calculer le log_prob
        dist = Categorical(probs)
        action = dist.sample()
        
        # Stockage du log_prob pour le calcul du gradient à la fin de l'épisode
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        """Stockage des récompenses reçues à chaque step."""
        self.rewards.append(reward)

    def update(self):
        """Mise à jour de la politique selon le théorème du Gradient de Politique.
        Cette méthode est appelée uniquement à la FIN d'un épisode.
        """
        # 1. Calcul des retours cumulés (Returns) : G_t = Σ γ^k * r_{t+k}
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # 2. Normalisation des retours (Baseline) pour réduire la variance et stabiliser REINFORCE
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3. Calcul de la fonction de perte (Loss)
        # J(θ) = E[ G_t * log π_θ(a_t|s_t) ]
        loss = []
        for log_prob, G in zip(self.log_probs, returns):
            loss.append(-log_prob * G) # Signe moins car PyTorch minimise, et on veut maximiser le gain
        
        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Nettoyage des mémoires pour le prochain épisode
        self.log_probs = []
        self.rewards = []

        return loss.item()