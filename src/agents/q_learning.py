import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    """Agent implémentant l'algorithme Q-Learning tabulaire.
    Nécessite une discrétisation de l'espace des états continu.
    """
    def __init__(self, n_actions, config):
        self.n_actions = n_actions
        self.gamma = config["gamma"]  # Facteur de remise
        self.lr = config["lr"]        # Taux d'apprentissage

        # Paramètres de la stratégie epsilon-greedy
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]

        # Q-table : dictionnaire associant un état discrétisé à un vecteur d'actions
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def discretize(self, obs):
        """Transforme le vecteur d'observation continu (dim 8) en un tuple d'entiers (état discret).
        Cette étape est cruciale pour le Q-Learning tabulaire afin de limiter la taille de la table.
        """
        # Taille des "cases" (bins) pour chaque dimension de l'observation
        bins = np.array([0.5, 0.5, 0.2, 0.2, 0.1, 0.1, 1.0, 1.0])
        return tuple((obs / bins).astype(int))

    def select_action(self, obs):
        """Sélection d'action epsilon-greedy sur l'état discrétisé."""
        state = self.discretize(obs)

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # Exploitation (greedy)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Mise à jour de la Q-table selon l'équation de Bellman temporelle.
        Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        """
        s = self.discretize(state)
        s_next = self.discretize(next_state)

        # Estimation de la valeur future maximale
        best_next = np.max(self.q_table[s_next])

        # Cible (Target) de mise à jour
        target = reward + self.gamma * best_next * (1 - done)

        # Mise à jour incrémentale
        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])

        # Décroissance de l'exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)