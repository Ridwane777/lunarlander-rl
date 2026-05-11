# LunarLander RL — LunarLander-v3

Ce projet implémente et compare plusieurs algorithmes de Reinforcement Learning sur l'environnement **LunarLander-v3** (Gymnasium / Box2D). L'objectif est d'entraîner un module lunaire à atterrir de manière autonome, avec un score moyen ≥ 200 sur 100 épisodes consécutifs.

---

## 🚀 Installation

### Prérequis
- Python ≥ 3.10
- `pip`

### Mise en place de l'environnement

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd lunarlander-rl

# Créer un environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🤖 Agents Implémentés

| Agent | Catégorie | Mode | Fichier |
|---|---|---|---|
| **Random** | Baseline | N/A | `src/agents/random_agent.py` |
| **Heuristic** | Baseline | N/A | `src/agents/heuristic_agent.py` |
| **Q-Learning** | Tabulaire | **Off-policy** | `src/agents/q_learning.py` |
| **REINFORCE** | Policy Gradient | **On-policy** | `src/agents/reinforce_agent.py` |
| **DQN (Double)** | Value-based | **Off-policy** | `src/agents/dqn_agent.py` |

---

## 📂 Structure du Projet

```
.
├── train.py                    # Point d'entrée — entraînement
├── eval.py                     # Point d'entrée — évaluation & benchmark
│
├── scripts/                    # Scripts utilitaires bash
│   ├── reproduce.sh            # Script one-click de reproduction
│   └── run_eval.sh             # Script de lancement de l'évaluation
│
├── configs/                    # Hyperparamètres par algorithme (YAML)
├── docs/                       # Documentation & Rapports
│   ├── benchmark.md            # Synthèse complète des résultats
│   └── optimisation.md         # Justification des choix techniques
│
├── src/
│   ├── agents/                 # Logique des agents RL
│   ├── evals/                  # Pipeline d'évaluation déterministe
│   ├── trains/                 # Boucles d'entraînement
│   └── utils/                  # Utilitaires (Buffer, Logger, Stats, Vidéo)
│
├── tests/                      # Tests unitaires (pytest)
│   ├── test_agents.py
│   ├── test_configs.py
│   └── test_utils.py
│
├── logs/                       # Données CSV
│   ├── train/                  # Logs d'apprentissage
│   └── eval/                   # Logs d'évaluation (100 épisodes)
│
├── models/                     # Poids des modèles .pth
├── plots/                      # Courbes et graphiques comparatifs
└── videos/
    ├── train/                  # Vidéos d'apprentissage
    └── evals/                  # Vidéos finales de pilotage
```

---

## 🛠️ Utilisation

### 1. Entraînement
```bash
# Entraîner un agent spécifique
python train.py --config configs/dqn_hard.yaml

# Entraîner tous les agents du projet
python train.py

# Régénérer uniquement les graphiques d'entraînement depuis les logs
python train.py --plot-only
```

### 2. Évaluation & Benchmark
L'évaluation charge les modèles `.pth`, désactive l'exploration et génère les graphiques comparatifs.
```bash
# Évaluer tous les modèles
python eval.py

# Régénérer uniquement les graphiques sans refaire les tests
python eval.py --plot-only

# Alternative via script bash
bash scripts/run_eval.sh
```

### 3. Reproduction Complète
Pour régénérer l'intégralité des résultats du projet en une commande :
```bash
bash scripts/reproduce.sh
```

---

## 🧪 Tests Unitaires

Nous utilisons `pytest` pour garantir la stabilité de chaque composant.
```bash
# Lancer tous les tests
pytest tests/
```
Les tests vérifient l'initialisation des agents, la sélection d'actions, la validité des fichiers YAML et le bon fonctionnement du Replay Buffer.

---

## 📊 Résultats du Benchmark (Résumé)

Évaluation sur 100 épisodes déterministes (Mean Reward ± IC 95%) :

| Méthode | Score Moyen | Succès | Verdict |
|---|---|---|---|
| **DQN (Hard Update)** | **218.5 ± 5.4** | **75.6%** | ✅ Vainqueur |
| **DQN (Soft Update)** | **210.0 ± 5.4** | **70.6%** | ✅ Stable |
| REINFORCE (C2) | 77.6 ± 12.5 | 12.0% | ⚠️ Amélioré mais instable |
| Q-Learning | -41.8 ± 16.1 | 18.2% | ❌ Limité (Tabulaire) |
| Heuristic | -219.8 ± 14.9 | 0.0% | ❌ Échec des règles |

> **Conclusion** : La supériorité du **Double DQN** avec **Hard Target Update** est démontrée. Il surclasse les méthodes tabulaires et les politiques de gradient simples grâce à sa mémoire (Replay Buffer) et sa capacité de généralisation sur les états continus.

---
