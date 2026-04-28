import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions, config):
        self.n_actions = n_actions
        self.gamma = config["gamma"]
        self.lr = config["lr"]

        # epsilon-greedy
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]

        # Q-table (dictionnaire)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def discretize(self, obs):
        bins = np.array([0.5, 0.5, 0.2, 0.2, 0.1, 0.1, 1.0, 1.0])
        return tuple((obs / bins).astype(int))

    def select_action(self, obs):
        state = self.discretize(obs)

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        s = self.discretize(state)
        s_next = self.discretize(next_state)

        best_next = np.max(self.q_table[s_next])

        target = reward + self.gamma * best_next * (1 - done)

        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])

        # epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)