import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    """Tabular Q-Learning agent.
    Requires discretizing the continuous state space.
    """
    def __init__(self, n_actions, config):
        self.n_actions = n_actions
        self.gamma = config["gamma"]  # discount factor
        self.lr = config["lr"]        # learning rate

        # Epsilon-greedy parameters
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]

        # Q-table: dict mapping a discretized state to an action-value vector
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def discretize(self, obs):
        """Convert the continuous observation vector (dim 8) into an integer tuple (discrete state).
        This is key for tabular Q-Learning to keep the table size bounded.
        """
        # Bin size for each observation dimension
        bins = np.array([0.5, 0.5, 0.2, 0.2, 0.1, 0.1, 1.0, 1.0])
        return tuple((obs / bins).astype(int))

    def select_action(self, obs):
        """Epsilon-greedy action selection on the discretized state."""
        state = self.discretize(obs)

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # Exploitation (greedy)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Update the Q-table with the temporal-difference Bellman equation.
        Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        """
        s = self.discretize(state)
        s_next = self.discretize(next_state)

        # Estimate the maximum future value
        best_next = np.max(self.q_table[s_next])

        # Update target
        target = reward + self.gamma * best_next * (1 - done)

        # Incremental update
        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])

        # Exploration decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)