import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """Neural network for the stochastic policy π_θ(a|s)."""
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
        # Softmax yields a probability distribution over actions
        return torch.softmax(self.net(x), dim=-1)


class ReinforceAgent:
    """REINFORCE agent (vanilla policy gradient).
    On-policy method based on full trajectories (Monte Carlo).
    """
    def __init__(self, obs_dim, n_actions, config):
        self.gamma = config.get("gamma", 0.99)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(obs_dim, n_actions, hidden=config.get("hidden", 128)).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.get("lr", 0.001))

        # Temporary storage for one trajectory (a full episode)
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        """Sample an action from the distribution learned by the network."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.policy(obs_t)
        
        # Build a categorical distribution to sample and compute log_prob
        dist = Categorical(probs)
        action = dist.sample()
        
        # Store log_prob for the gradient at the end of the episode
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        """Store the reward received at each step."""
        self.rewards.append(reward)

    def update(self):
        """Update the policy with the policy gradient theorem.
        Called only at the END of an episode.
        """
        # 1. Compute discounted returns : G_t = Σ γ^k * r_{t+k}
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # 2. Normalize returns (baseline) to reduce variance and stabilize REINFORCE
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3. Compute the loss
        # J(θ) = E[ G_t * log π_θ(a_t|s_t) ]
        loss = []
        for log_prob, G in zip(self.log_probs, returns):
            loss.append(-log_prob * G) # Negative sign: PyTorch minimizes, we want to maximize return
        
        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Clear buffers for the next episode
        self.log_probs = []
        self.rewards = []

        return loss.item()