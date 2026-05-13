import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
    """Architecture Multi-Layer Perceptron (MLP) pour l'approximation de la Q-fonction."""
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
    """Agent implémentant l'algorithme Deep Q-Network avec support du Double DQN et du Soft/Hard Update."""
    def __init__(self, obs_dim, n_actions, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = config["gamma"]  # Facteur de remise (discount factor)
        self.batch_size = config["batch_size"]
        self.target_update_freq = config["target_update_freq"]
        self.update_every = config.get("update_every", 1)
        self.grad_clip = config.get("grad_clip", 10.0)
        self.soft_update = config.get("soft_update", False)
        self.tau = config.get("tau", 0.02)

        # Stratégie Epsilon-greedy pour gérer le dilemme Exploration/Exploitation
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]

        # Initialisation des deux réseaux : Policy (online) et Target (stable)
        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Le target network ne s'entraîne jamais directement

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config["lr"])
        self.buffer = ReplayBuffer(config["buffer_size"], self.device)
        self.step_count = 0
        self.t_step = 0

    def select_action(self, obs):
        """Sélection d'action via politique epsilon-greedy."""
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(self.n_actions, (1,)).item()
        
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        """Mise en mémoire de la transition dans le Replay Buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        """Mise à jour des poids du réseau via échantillonnage du buffer."""
        self.t_step += 1
        if self.t_step % self.update_every != 0:
            return None

        if len(self.buffer) < self.batch_size:
            return None

        # Échantillonnage aléatoire d'un batch de transitions
        states_t, actions_t, rewards_t, next_states_t, dones_t = self.buffer.sample(self.batch_size)

        # Calcul de Q(s,a) actuel
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN : découplage de la sélection et de l'évaluation pour éviter la surestimation
        # 1. Sélection de la meilleure action via le réseau principal
        # 2. Évaluation de cette action via le réseau cible (Target Net)
        with torch.no_grad():
            best_actions = self.q_net(next_states_t).argmax(1)
            next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Calcul de l'erreur quadratique moyenne (MSE Loss)
        loss = nn.MSELoss()(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Écrêtage du gradient pour la stabilité numérique
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Mise à jour synchronisée du Target Network
        self.step_count += 1
        if self.soft_update:
            # Polyak averaging : θ_target = τ*θ_local + (1-τ)*θ_target
            for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        else:
            # Hard update : copie intégrale tous les N pas
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Réduction progressive de l'exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)