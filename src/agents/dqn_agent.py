import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import copy
from src.utils.replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
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
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, n_actions, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]
        self.soft_update = config.get("soft_update", False)
        self.tau = config.get("tau", 0.005)

        # Epsilon-greedy
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]

        # Les deux réseaux
        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config["lr"])
        self.buffer = ReplayBuffer(config["buffer_size"])
        self.step_count = 0

    def select_action(self, obs):
        # Epsilon-greedy : exploration ou exploitation
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q-valeurs prédites par le réseau principal
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Cibles calculées par le target network
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Loss et backpropagation
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Décroissance epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Mise à jour du target network
        self.step_count += 1
        if self.soft_update:
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        else:
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()