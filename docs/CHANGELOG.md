# CHANGELOG TECHNIQUE — LunarLander RL

> Document à destination des binômes du projet.  
> Résumé de tous les choix d'architecture et changements appliqués au repo depuis l'état initial.

---

## 1. Refactorisation du `train.py` monolithique

### Problème
Le fichier `train.py` original faisait plus de 400 lignes et gérait seul : la logique d'entraînement de chaque algorithme, les logs, les graphiques, la configuration. Impossible à maintenir ou à étendre.

### Ce qui a été fait
Tout a été découpé en modules spécialisés :

**`src/trains/`** — une boucle d'entraînement par algorithme :
- `train_dqn.py` — DQN (hard ou soft selon le paramètre YAML)
- `train_reinforce.py` — REINFORCE (policy gradient Monte-Carlo)
- `train_q_learning.py` — Q-Learning tabulaire
- `train_baseline.py` — Random et Heuristic

**`src/utils/`** — utilitaires partagés :
- `core.py` — chargement YAML, seeds, chemins de logs, résumé terminal
- `logger.py` — logging CSV par épisode (`EpisodeLogger`)
- `model_utils.py` — sauvegarde et chargement des poids `.pth`
- `plots.py` — courbes d'entraînement par seed et agrégées
- `replay_buffer.py` — experience replay pour DQN

**`train.py`** est désormais réduit à ~100 lignes. C'est un orchestrateur : il lit les YAML de `configs/`, dispatche vers le bon trainer, sauvegarde les graphiques et les modèles.

---

## 2. Sauvegarde automatique des modèles

### Ce qui a été fait
À la fin de chaque entraînement, le modèle est automatiquement sauvegardé dans `models/` :

- DQN → `q_net.state_dict()` → `models/dqn_soft_seed0.pth`
- REINFORCE → `policy.state_dict()` → `models/reinforce_seed0.pth`
- Q-Learning → `dict(agent.q_table)` → `models/q_learning_seed0.pth`

La fonction `save_model()` / `load_model()` est dans `src/utils/model_utils.py`.

---

## 3. Correction du bug d'ablation DQN

### Problème
Les deux configs `dqn_soft.yaml` et `dqn_hard.yaml` avaient toutes les deux `algo: dqn`. Du coup, les deux entraînements écrivaient dans les mêmes fichiers `logs/dqn_seed0.csv` et `models/dqn_seed0.pth`, s'écrasant mutuellement.

### Fix appliqué
- `configs/dqn_soft.yaml` → `algo: dqn_soft`
- `configs/dqn_hard.yaml` → `algo: dqn_hard`
- `train.py` → `TRAINERS` dispatch mis à jour pour reconnaître `dqn_soft` et `dqn_hard`

Désormais chaque variante a ses propres fichiers isolés.

---

## 4. Refactorisation du `eval.py`

### Problème
L'ancien `eval.py` lisait les logs d'entraînement (`logs/*.csv`) pour produire les graphiques. Ce n'est pas une vraie évaluation : les données incluaient l'exploration aléatoire et les épisodes d'apprentissage.

### Ce qui a été fait
`eval.py` charge maintenant les modèles `.pth`, les teste en mode **greedy** (ε = 0) sur **100 épisodes déterministes** avec des seeds de test standardisées (10000 + ep). C'est une évaluation propre des performances finales.

Le code a été découpé en modules dans **`src/evals/`** :
- `eval_core.py` — boucle de test greedy pour un modèle donné
- `eval_metrics.py` — parsing CSV, calcul des métriques (IC 95%, success rate...)
- `eval_plots.py` — génération de tous les graphiques (6 au total)
- `eval_utils.py` — résumé console + `benchmark_summary.csv`

Les logs de test vont dans `eval_logs/` (séparé de `logs/`).

---

## 5. Enregistrement vidéo automatique

### Ce qui a été fait
`gym.wrappers.RecordVideo` a été ajouté dans tous les trainers et dans `eval_core.py` :

- **Entraînement** → `videos/train/<algo>_seed<N>/`
- **Évaluation** → `videos/tests/<algo>_seed<N>/`

Le wrapper n'enregistre pas tous les épisodes (ça serait trop lourd), mais les épisodes 1, 8, 27, 64... (suite cubique par défaut de Gymnasium). Cela permet de voir la progression de l'agent sans saturer le disque.

---

## 6. Amélioration du logger CSV

### Problème
Le `EpisodeLogger` original ne loguait que `episode`, `total_reward`, `length`, `termination_cause`. Le sujet demande explicitement de tracker **epsilon** et **loss**.

### Fix appliqué
Deux nouvelles colonnes dans chaque CSV :
- `epsilon` — valeur de ε au moment de l'épisode (DQN, Q-Learning uniquement ; vide sinon)
- `loss_mean` — moyenne des losses sur les steps de l'épisode (DQN, REINFORCE uniquement ; vide sinon)

Pour DQN : la loss est accumulée step par step dans `ep_losses[]` et la moyenne est calculée à la fin de chaque épisode.

---

## 7. Nouveau graphique : longueur des épisodes

Le sujet demandait un graphique de la longueur des épisodes. Ajout de `plot_episode_length()` dans `eval_plots.py`. Il génère `06_episode_length.png` — un boxplot par agent montrant la distribution du nombre de steps par épisode lors des évaluations.

---

## 8. Ablation hard vs soft update

### Ce qui a été testé
La seule différence entre les deux configs DQN est la stratégie de mise à jour du **target network** :

| Variante | Paramètre | Comportement |
|---|---|---|
| `dqn_hard` | `soft_update: false`, `target_update_freq: 500` | Copie complète du réseau tous les 500 updates |
| `dqn_soft` | `soft_update: true`, `tau: 0.005` | Mise à jour Polyak à chaque step : `θ_t ← τ·θ + (1−τ)·θ_t` |

### Justification théorique
- Le **hard update** (Mnih et al. 2015) gèle complètement la cible pendant `C` steps → stabilité maximale mais convergence potentiellement plus lente.
- Le **soft update** (Lillicrap et al. 2016, DDPG) fait évoluer la cible graduellement → convergence plus fluide mais cible jamais vraiment figée.

### Comment visualiser les résultats
```bash
python eval.py --models models/dqn_soft_seed*.pth models/dqn_hard_seed*.pth \
               --out plots/ablation_dqn
```

---

## 9. Nettoyage du repo

### Fichiers supprimés (redondants)
- `src/utils/ablation_plot.py` — vieux script monolithique remplacé par `eval_plots.py`
- `src/utils/compare_learning_methods.py` — vieux script remplacé par le pipeline `eval.py`
- `tests/test_ramdom.py` — typo + script de training déguisé en test

---

## 10. Script de reproduction

`reproduce.sh` permet de tout relancer en une commande :

```bash
bash reproduce.sh              # entraînement complet + évaluation + graphiques
bash reproduce.sh --eval-only  # évaluation uniquement (models/ déjà présents)
bash reproduce.sh --agent dqn_soft --seed 0  # un seul agent, une seule seed
```

Le script vérifie la version Python (≥ 3.10), installe les dépendances, crée tous les dossiers nécessaires, lance `train.py` puis `eval.py`.

---

## Structure finale du repo

```
train.py          ← point d'entrée entraînement
eval.py           ← point d'entrée évaluation + benchmark
reproduce.sh      ← one-click reproduction
configs/          ← 1 YAML par algorithme (hyperparamètres + seeds)
src/
  agents/         ← définition des agents RL
  trains/         ← boucles d'entraînement
  evals/          ← pipeline d'évaluation
  utils/          ← utilitaires partagés
logs/             ← CSV d'apprentissage (généré)
eval_logs/        ← CSV d'évaluation greedy (généré)
models/           ← poids .pth (généré)
plots/            ← graphiques d'entraînement (généré)
videos/train/     ← vidéos d'entraînement (généré)
videos/tests/     ← vidéos d'évaluation (généré)
```
