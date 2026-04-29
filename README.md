# LunarLander RL : LunarLander-v3

## 🎯 Objectif

Ce projet implémente et compare plusieurs algorithmes de Reinforcement Learning sur l'environnement **LunarLander-v3** (Gymnasium / Box2D). L'objectif est d'entraîner un module lunaire à atterrir de manière autonome, avec un score moyen ≥ 200 sur 100 épisodes consécutifs.

---

## ⚙️ Installation

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
pip install "gymnasium[box2d]"
```

### Dépendances
| Librairie | Usage |
|---|---|
| `gymnasium` | Environnement LunarLander-v3 |
| `torch` | Réseaux de neurones (DQN, REINFORCE) |
| `matplotlib` | Génération des graphiques |
| `numpy` | Calculs numériques |
| `pyyaml` | Lecture des configs YAML |

---

## Agents Implémentés

### Baselines
| Agent | Description | Fichier |
|---|---|---|
| **Random** | Politique aléatoire uniforme | `src/agents/random_agent.py` |
| **Heuristic** | Règles manuelles (correction angulaire + freinage) | `src/agents/heuristic_agent.py` |

### Algorithmes RL
| Agent | Description | Fichier |
|---|---|---|
| **Q-Learning** | Méthode tabulaire avec discrétisation de l'espace d'états | `src/agents/q_learning.py` |
| **REINFORCE** | Policy gradient Monte-Carlo | `src/agents/reinforce_agent.py` |
| **DQN (hard update)** | Deep Q-Network avec mise à jour périodique du target network | `src/agents/dqn_agent.py` |
| **DQN (soft update)** | DQN avec mise à jour polyak (τ = 0.005) — **ablation** | `src/agents/dqn_agent.py` |

Le DQN utilise **Double DQN** : le réseau principal choisit l'action, le target network évalue la Q-valeur. Il inclut également un **gradient clipping** à 10.0.

---

## 🗂️ Structure du Projet

```
.
├── train.py                    # Point d'entrée — entraînement
├── eval.py                     # Point d'entrée — évaluation & benchmark
├── reproduce.sh                # Script one-click de reproduction
├── requirements.txt
│
├── configs/                    # Hyperparamètres par algorithme (YAML)
│   ├── dqn_soft.yaml
│   ├── dqn_hard.yaml
│   ├── reinforce.yaml
│   ├── q_learning.yaml
│   ├── random.yaml
│   └── heuristic.yaml
│
├── src/
│   ├── agents/                 # Définition des agents
│   │   ├── dqn_agent.py
│   │   ├── reinforce_agent.py
│   │   ├── q_learning.py
│   │   ├── random_agent.py
│   │   └── heuristic_agent.py
│   │
│   ├── trains/                 # Boucles d'entraînement
│   │   ├── train_dqn.py
│   │   ├── train_reinforce.py
│   │   ├── train_q_learning.py
│   │   └── train_baseline.py
│   │
│   ├── evals/                  # Pipeline d'évaluation
│   │   ├── eval_core.py        # Boucle de test greedy sur modèles .pth
│   │   ├── eval_metrics.py     # Calcul des métriques (IC 95%, succès...)
│   │   ├── eval_plots.py       # Génération des graphiques
│   │   └── eval_utils.py       # Résumé console + CSV
│   │
│   └── utils/                  # Utilitaires partagés
│       ├── core.py             # Seeds, configs YAML, résumé
│       ├── logger.py           # Logging CSV par épisode
│       ├── model_utils.py      # Sauvegarde / chargement des .pth
│       ├── plots.py            # Courbes d'entraînement
│       ├── replay_buffer.py    # Experience replay (DQN)
│       └── ablation_plot.py    # Comparaisons d'ablations
│
├── tests/                      # Tests unitaires par agent
│   ├── test_dqn.py
│   ├── test_reinforce.py
│   ├── test_q_learning.py
│   ├── test_ramdom.py
│   └── test_heuristic.py
│
├── docs/                       # Rapports et notes
│   └── ...
│
├── logs/          # CSV d'apprentissage (générés)
├── eval_logs/     # CSV d'évaluation déterministe (générés)
├── models/        # Poids des modèles .pth (générés)
├── plots/         # Graphiques d'entraînement (générés)
└── videos/
    ├── train/     # Vidéos d'entraînement (générés)
    └── tests/     # Vidéos d'évaluation finale (générés)
```

---

## Entraînement

### Entraîner un seul agent

```bash
# DQN soft update (toutes les seeds)
python train.py --config configs/dqn_soft.yaml

# DQN hard update, seed 0 uniquement
python train.py --config configs/dqn_hard.yaml --seed 0

# REINFORCE
python train.py --config configs/reinforce.yaml

# Q-Learning
python train.py --config configs/q_learning.yaml

# Baselines
python train.py --config configs/random.yaml
python train.py --config configs/heuristic.yaml
```

### Entraîner tous les agents (toutes les seeds)

```bash
python train.py
```

Chaque entraînement génère automatiquement :
- `logs/<algo>_seed<N>.csv` — métriques épisodiques (reward, length, epsilon, loss)
- `plots/<algo>/` — courbes de convergence
- `models/<algo>_seed<N>.pth` — poids du modèle sauvegardé
- `videos/train/<algo>_seed<N>/` — vidéos d'entraînement (épisodes sélectionnés)

### Lister les configs disponibles

```bash
python train.py --list
```

---

## 📊 Évaluation

L'évaluation charge les modèles `.pth` depuis `models/`, désactive l'exploration (`ε = 0`), et teste chaque agent sur **100 épisodes déterministes** avec des seeds de test standardisées.

### Évaluer tous les modèles disponibles

```bash
python eval.py
```

### Évaluer un modèle précis

```bash
python eval.py --models models/dqn_soft_seed0.pth models/dqn_soft_seed1.pth
```

### Options

```bash
python eval.py --n_episodes 200           # 200 épisodes de test
python eval.py --out plots/my_benchmark   # Dossier de sortie personnalisé
```

Chaque évaluation génère automatiquement dans `plots/benchmark/` :
| Fichier | Description |
|---|---|
| `01_mean_reward_ci95.png` | Reward moyen ± IC 95% par agent |
| `03_reward_distribution.png` | Boxplot de la distribution des rewards |
| `04_termination_causes.png` | Répartition crash / atterrissage / timeout |
| `05_success_rate.png` | Taux de succès (épisodes ≥ 200) |
| `06_episode_length.png` | Distribution de la longueur des épisodes |
| `benchmark_summary.csv` | Tableau récapitulatif de toutes les métriques |

Et dans `videos/tests/` : les vidéos des comportements finaux de chaque agent.

---

## Reproduction Complète

Pour régénérer **tous** les résultats (entraînement + évaluation + graphiques + vidéos) :

```bash
bash reproduce.sh
```

### Options du script

```bash
bash reproduce.sh --eval-only              # Évaluation uniquement (models/ déjà présents)
bash reproduce.sh --agent dqn_soft         # Un seul agent
bash reproduce.sh --seed 0                 # Une seule seed
```

> **Reproductibilité garantie à ±5%** avec les seeds 0 à 4.

---

## Tests Unitaires

Des scripts de test sont disponibles dans `tests/` pour valider le comportement de chaque agent de manière isolée :

```bash
# Tester le DQN (entraînement court)
python tests/test_dqn.py

# Tester REINFORCE
python tests/test_reinforce.py

# Tester Q-Learning
python tests/test_q_learning.py

# Tester la politique aléatoire
python tests/test_ramdom.py

# Tester l'heuristique
python tests/test_heuristic.py
```

---

## 📈 Métriques et Reproductibilité

### Ce qui est tracé par épisode (CSV)
- `total_reward` — retour total de l'épisode
- `length` — nombre de steps
- `termination_cause` — `landed` / `crash` / `timeout`
- `epsilon` — valeur de l'exploration ε au moment de l'épisode (DQN, Q-Learning)
- `loss_mean` — loss moyenne sur l'épisode (DQN, REINFORCE)

### Critère de succès
Un agent est considéré comme **résolu** quand la moyenne glissante sur 100 épisodes consécutifs dépasse **200 points**.

### Reproductibilité
- **5 seeds** (0, 1, 2, 3, 4) par algorithme
- **Intervalle de confiance à 95%** calculé sur l'ensemble des seeds
- Seeds de test standardisées (10000 + ep) pour les évaluations

---

## Ablation

L'ablation principale compare deux stratégies de mise à jour du **target network** dans DQN :

| Stratégie | Paramètre | Config |
|---|---|---|
| **Hard update** | Copie complète tous les N pas | `configs/dqn_hard.yaml` |
| **Soft update** | Mise à jour Polyak (τ = 0.005) | `configs/dqn_soft.yaml` |

Pour lancer uniquement l'ablation :

```bash
python train.py --config configs/dqn_hard.yaml
python train.py --config configs/dqn_soft.yaml
python eval.py --models models/dqn_hard_seed*.pth models/dqn_soft_seed*.pth
```

---

##  Résultats Observés

| Méthode | Mean Reward | Résolution |
|---|---|---|
| DQN (soft update) | ~202 | ✅ Résolu (~180 épisodes) |
| DQN (hard update) | ~180 | ⚠️ Proche du seuil |
| REINFORCE | ~65 | ❌ Non résolu |
| Q-Learning | ~7 | ❌ Non résolu |
| Heuristic | ~50 | ❌ Non résolu |
| Random | ~-200 | ❌ Non résolu |

> Résultats obtenus sur seed 0 — seed 4. Le DQN soft update est la seule méthode à franchir systématiquement le seuil de 200.
