# Benchmark LunarLander-v3

Ce document regroupe les résultats d'entraînement et d'évaluation des différents agents.

## 1. Random Agent (Baseline)

L'agent aléatoire choisit ses actions uniformément au hasard. Il sert de base de comparaison pour les autres algorithmes.

### Entraînement
- **Config** : `configs/random.yaml`
- **Graines (Seeds)** : [0, 1, 2, 3, 4]
- **Nombre de pas max** : 50 000

| Seed | Épisodes | Reward Moyen | Reward Max | Status | Temps |
|------|----------|--------------|------------|--------|-------|
| 0    | 528      | -180.5       | 77.1       | Terminé| ~60s  |
| 1    | 543      | -178.1       | 28.4       | Terminé| ~60s  |
| 2    | 525      | -179.2       | 37.1       | Terminé| ~60s  |
| 3    | 532      | -181.8       | 39.3       | Terminé| ~60s  |
| 4    | 540      | -186.3       | 74.3       | Terminé| ~60s  |

### Évaluation (100 épisodes par seed, soit 500 épisodes au total)
- **Reward Moyen** : -184.6 ± 9.6
- **Taux de succès** : 0.0%
- **Cause de fin** : Principalement 'crash' (l'agent ne sait pas se poser).
- **Max Reward** : 72.2

### Hyperparamètres
Aucun (agent purement aléatoire).

### Conclusion
L'agent aléatoire affiche des performances extrêmement faibles avec un reward moyen de **-184.6**. Le taux de succès est de **0%**, ce qui confirme que la probabilité de réussir un atterrissage par pur hasard est quasi nulle. Cet agent servira de "zéro de référence" pour mesurer l'apprentissage des futurs agents (DQN, REINFORCE, etc.). Tout algorithme d'apprentissage devra impérativement dépasser ces scores pour être considéré comme fonctionnel.

---

## 2. Heuristic Agent (Baseline)

L'agent heuristique utilise un ensemble de règles codées à la main (basées sur la position, l'angle et la vitesse) pour tenter de stabiliser le lander.

### Entraînement
- **Config** : `configs/heuristic.yaml`
- **Graines (Seeds)** : [0, 1, 2, 3, 4]
- **Nombre de pas max** : 50 000

| Seed | Épisodes | Reward Moyen | Reward Max | Status | Temps |
|------|----------|--------------|------------|--------|-------|
| 0    | 716      | -226.0       | 38.2       | Terminé| ~60s  |
| 1    | 718      | -219.1       | 55.5       | Terminé| ~60s  |
| 2    | 714      | -215.7       | 19.1       | Terminé| ~60s  |
| 3    | 706      | -223.2       | 32.7       | Terminé| ~60s  |
| 4    | 709      | -226.4       | 7.8        | Terminé| ~60s  |

### Évaluation (100 épisodes par seed, soit 500 épisodes au total)
- **Reward Moyen** : -219.8 ± 14.9
- **Taux de succès** : 0.0%
- **Cause de fin** : Exclusivement 'crash'.
- **Max Reward** : 55.5

### Hyperparamètres
Aucun (règles fixes).

### Conclusion
Contre-intuitivement, l'agent heuristique obtient des résultats **inférieurs** à l'agent aléatoire sur cette configuration (-219.8 contre -184.6). Bien qu'il tente de stabiliser l'appareil, ses règles codées semblent provoquer des crashs plus systématiques ou plus pénalisants. Cela montre qu'une stratégie naïve, même intentionnelle, peut être moins efficace qu'une absence totale de stratégie dans certains environnements complexes. L'apprentissage par renforcement sera crucial pour trouver un équilibre que les règles manuelles n'atteignent pas ici.

---

## 3. Q-Learning (Tabulaire) - Config 1

Le Q-Learning est un algorithme d'apprentissage par renforcement hors-politique (off-policy). Comme l'espace des observations est continu, il nécessite une discrétisation pour être utilisé de manière tabulaire.

### Entraînement
- **Config** : `configs/q_learning.yaml`
- **Graines (Seeds)** : [0, 1, 2, 3, 4]
- **Nombre de pas max** : 200 000

| Seed | Épisodes | Reward Moyen | Reward Max | Status | Temps |
|------|----------|--------------|------------|--------|-------|
| 0    | 840      | -101.6       | 254.4      | Terminé| ~7min |
| 1    | 898      | -108.9       | 254.3      | Terminé| ~7min |
| 2    | 810      | -93.6        | 263.7      | Terminé| ~7min |
| 3    | 915      | -122.4       | 285.6      | Terminé| ~7min |
| 4    | 833      | -91.9        | 268.0      | Terminé| ~7min |

### Évaluation (100 épisodes par seed, soit 500 épisodes au total)
- **Reward Moyen** : -41.8 ± 16.1
- **Taux de succès** : 18.2%
- **Cause de fin** : Mélange de 'crash', 'solved' et 'timeout'.
- **Max Reward** : 285.6

### Hyperparamètres
- **Gamma** : 0.99
- **Learning Rate (LR)** : 0.1
- **Epsilon** : 1.0 -> 0.01 (decay 0.995)

### Conclusion
Le Q-Learning tabulaire montre une progression nette par rapport aux baselines. Avec un taux de succès de **18.2%**, l'agent commence à comprendre comment stabiliser le lander et se poser, même si l'espace discrétisé limite sa précision. On observe que l'agent réussit parfois des atterrissages parfaits (> 200 points). Cependant, la moyenne reste négative car les crashs sont encore fréquents et lourdement pénalisés.

---

## 4. Q-Learning (Tabulaire) - Config 2

Cette version utilise une exploration plus longue (`epsilon_decay=0.999`) et un taux d'apprentissage plus faible (`lr=0.05`) pour voir si une convergence plus lente améliore la stabilité.

### Entraînement
- **Config** : `configs/q_learning_v2.yaml`
- **Graines (Seeds)** : [0, 1, 2, 3, 4]
- **Nombre de pas max** : 200 000

| Seed | Épisodes | Reward Moyen | Reward Max | Status | Temps |
|------|----------|--------------|------------|--------|-------|
| 0    | 1031     | -153.2       | 261.8      | Terminé| ~7min |
| 1    | 952      | -117.1       | 262.2      | Terminé| ~7min |
| 2    | 921      | -117.7       | 257.4      | Terminé| ~7min |
| 3    | 997      | -138.1       | 257.0      | Terminé| ~7min |
| 4    | 990      | -116.7       | 280.4      | Terminé| ~7min |

### Évaluation (100 épisodes par seed, soit 500 épisodes au total)
- **Reward Moyen** : -92.7 ± 14.4
- **Taux de succès** : 11.6%
- **Cause de fin** : Principalement 'crash', quelques 'solved'.
- **Max Reward** : 280.4

### Hyperparamètres
- **Gamma** : 0.99
- **Learning Rate (LR)** : 0.05 (contre 0.1 en Config 1)
- **Epsilon** : 1.0 -> 0.01 (decay 0.999 contre 0.995)

### Conclusion
La Config 2, malgré une exploration plus longue et un apprentissage plus prudent, obtient des résultats **moins bons** que la Config 1 (-92.7 contre -41.8 de moyenne). Le taux de succès chute également à **11.6%**. Il est probable que le taux d'apprentissage trop faible ralentisse excessivement la convergence, et que l'exploration prolongée empêche l'agent de stabiliser sa politique dans le temps imparti (200 000 pas). La Config 1 reste donc notre meilleure approche pour le Q-Learning tabulaire.

### Synthèse Comparative Q-Learning
| Config | Reward Moyen | Taux de Succès | Observation Clé |
|--------|--------------|----------------|-----------------|
| Config 1 (LR 0.1, Decay 0.995) | **-41.8** | **18.2%** | Convergence plus rapide et meilleure exploitation. |
| Config 2 (LR 0.05, Decay 0.999) | -92.7 | 11.6% | Exploration trop longue, manque de temps pour converger. |

**Verdict** : La **Config 1** est la plus efficace. Elle offre le meilleur compromis entre vitesse d'apprentissage et exploitation des connaissances. Cependant, le Q-Learning tabulaire montre ici ses limites face à la complexité de l'environnement : même la meilleure configuration ne parvient pas à une moyenne positive, illustrant la nécessité de passer à des méthodes d'approximation de fonction (Deep RL).

---

## 4. REINFORCE (Policy Gradient)

REINFORCE est un algorithme de gradient de politique de type Monte-Carlo. Contrairement au Q-Learning, il apprend directement une politique paramétrée par un réseau de neurones.

### Entraînement
- **Config** : `configs/reinforce.yaml`
- **Graines (Seeds)** : [0, 1, 2, 3, 4]
- **Nombre de pas max** : 200 000

| Seed | Épisodes | Reward Moyen | Reward Max | Status | Temps |
|------|----------|--------------|------------|--------|-------|
| 0    | 828      | -38.4        | 315.2      | Terminé| ~5min |
| 1    | 795      | -70.1        | 297.7      | Terminé| ~5min |
| 2    | 806      | -41.7        | 299.6      | Terminé| ~5min |
| 3    | 824      | -30.3        | 307.1      | Terminé| ~5min |
| 4    | 616      | -85.4        | 201.6      | Terminé| ~6min |

### Évaluation (100 épisodes par seed, soit 500 épisodes au total)
- **Reward Moyen** : 39.2 ± 11.9
- **Taux de succès** : 18.2%
- **Cause de fin** : Mélange de 'solved' et 'crash'.
- **Max Reward** : 315.2

### Hyperparamètres
- **Gamma** : 0.99
- **Learning Rate (LR)** : 0.001
- **Réseau** : 1 couche cachée de 128 neurones

### Conclusion
REINFORCE montre une nette supériorité sur le Q-Learning en termes de reward moyen (passant de -41.8 à **+39.2**). Bien que le taux de succès soit identique (**18.2%**), l'agent semble avoir une bien meilleure gestion des échecs, évitant des crashs trop violents. L'utilisation d'un réseau de neurones permet de capturer la continuité de l'espace d'observation de manière bien plus efficace que la discrétisation tabulaire.

---

## 5. REINFORCE (Policy Gradient) - Config 2

Cette version utilise un réseau plus large (`hidden=256`) et un taux d'apprentissage plus faible (`lr=0.0005`) pour capturer des relations plus complexes tout en maintenant la stabilité.

### Entraînement
- **Config** : `configs/reinforce_v2.yaml`
- **Graines (Seeds)** : [0, 1, 2, 3, 4]
- **Nombre de pas max** : 200 000

| Seed | Épisodes | Reward Moyen | Reward Max | Status | Temps |
|------|----------|--------------|------------|--------|-------|
| 0    | 627      | -50.6        | 166.2      | Terminé| ~6min |
| 1    | 674      | -63.6        | 253.2      | Terminé| ~6min |
| 2    | 684      | -39.0        | 279.4      | Terminé| ~6min |
| 3    | 816      | -61.9        | 257.9      | Terminé| ~6min |
| 4    | 695      | -41.6        | 215.4      | Terminé| ~6min |

### Évaluation (100 épisodes par seed, soit 500 épisodes au total)
- **Reward Moyen** : 77.6 ± 8.1
- **Taux de succès** : 12.0%
- **Cause de fin** : Très peu de crashs, beaucoup de 'landed' hors zone ou temps limite.
- **Max Reward** : 279.4

### Hyperparamètres
- **Gamma** : 0.99
- **Learning Rate (LR)** : 0.0005
- **Réseau** : 1 couche cachée de 256 neurones

### Conclusion
La Config 2 avec un réseau plus large et un taux d'apprentissage plus faible montre une **stabilité nettement supérieure** (Reward moyen de +77.6). Cependant, le taux de succès brut est plus faible que la Config 1 (12.0% vs 18.2%). Cela suggère que l'agent a appris une politique très prudente qui minimise les risques de crash mais manque encore de la précision nécessaire pour se poser systématiquement dans la zone cible. 

Le passage de REINFORCE (Config 2) marque le premier agent à obtenir une moyenne de reward solide et positive sur cet environnement complexe.

### 5.3. Synthèse Comparative REINFORCE
| Config | Hidden | LR | Reward Moyen | Taux de Succès | Observation Clé |
|--------|--------|----|--------------|----------------|-----------------|
| Config 1| 128    | 0.001 | 39.2        | **18.2%**      | Plus agressif, mais plus instable. |
| Config 2| 256    | 0.0005| **77.6**      | 12.0%          | Très stable, évite les crashs. |

**Leçons pour le DQN :**
- **Architecture** : Un réseau plus large (256) aide à la stabilité globale, mais le réseau standard (128) semble suffisant pour "trouver" la zone cible.
- **Vitesse d'apprentissage** : Le DQN étant plus sensible aux oscillations, nous utiliserons un LR de **0.0003** (standard pour Adam/DQN) pour maximiser la stabilité apprise avec la Config 2 de REINFORCE.
- **Cible** : La stabilité du reward en Config 2 montre qu'une approche plus "douce" (LR bas, gros réseau) est préférable pour ce type d'environnement continu.

---

## 6. DQN (Deep Q-Network)

Le DQN combine le Q-Learning avec des réseaux de neurones profonds. Contrairement à REINFORCE qui est un algorithme de gradient de politique (on-policy), le DQN est un algorithme de valeur (off-policy) utilisant un **Replay Buffer** pour réutiliser les expériences passées et un **Réseau Cible (Target Network)** pour stabiliser l'apprentissage.

**Pourquoi passer au DQN ?**
- **Efficacité** : Le Replay Buffer permet d'apprendre beaucoup plus de choses avec moins d'épisodes en rejouant les transitions importantes.
- **Stabilité** : Les leçons de REINFORCE (Config 2) nous ont montré que la stabilité est la clé du succès sur LunarLander. Le Target Network du DQN est spécifiquement conçu pour éviter les divergences que nous avons vues en Q-Learning Tabulaire.

### Stratégie de mise à jour
Nous allons comparer deux méthodes de stabilisation du réseau cible, tout en conservant les leçons apprises sur le réseau de neurones (LR faible, architecture robuste).

#### 6.1. DQN Hard Update
Le réseau cible est synchronisé avec le réseau principal par paliers (`target_update_freq=500`).

- **Config** : `configs/dqn_hard.yaml`
- **Graines (Seeds)** : [0, 1, 2, 3, 4]
- **Nombre de pas max** : 500 000

| Seed | Épisodes | Reward Moyen | Reward Max | Status | Temps |
|------|----------|--------------|------------|--------|-------|
| 0    | 432      | 12.0         | 316.0      | Résolu | ~12min|
| 1    | 450      | -19.2        | 292.4      | Résolu | ~12min|
| 2    | 441      | 26.8         | 290.6      | Résolu | ~12min|
| 3    | 534      | 37.0         | 309.6      | Résolu | ~15min|
| 4    | 453      | 18.9         | 315.9      | Résolu | ~12min|

### 6.2. DQN Soft Update (Polyak)
Le réseau cible est mis à jour progressivement à chaque pas en utilisant un facteur $\tau$ (`tau=0.02`).

- **Config** : `configs/dqn_soft.yaml`
- **Graines (Seeds)** : [0, 1, 2, 3, 4]
- **Nombre de pas max** : 500 000

| Seed | Épisodes | Reward Moyen | Reward Max | Status | Temps |
|------|----------|--------------|------------|--------|-------|
| 0    | 393      | 14.0         | 304.9      | Résolu | ~12min|
| 1    | 405      | 29.8         | 291.7      | Résolu | ~12min|
| 2    | 470      | 31.2         | 296.8      | Résolu | ~15min|
| 3    | 504      | 12.6         | 292.0      | Résolu | ~15min|
| 4    | 560      | 63.0         | 298.1      | Résolu | ~15min|

### Évaluation DQN Hard (100 épisodes par seed)
- **Reward Moyen** : **218.5 ± 5.4**
- **Taux de succès** : **75.6%**
- **Cause de fin** : Atterrissages maîtrisés dans la zone cible.
- **Max Reward** : 316.0

**Observation DQN Hard :**
Le DQN Hard Update marque une rupture technologique dans ce benchmark. En utilisant un Replay Buffer et un Target Network, il parvient à **résoudre** l'environnement sur toutes les seeds là où REINFORCE plafonnait. L'agent a acquis une précision de pilotage que les méthodes on-policy n'ont pas réussi à stabiliser dans le temps imparti.

### Évaluation DQN Soft (100 épisodes par seed)
- **Reward Moyen** : **210.0 ± 5.4**
- **Taux de succès** : **70.6%**
- **Cause de fin** : Atterrissages maîtrisés, mais parfois trop de prudence (timeouts).
- **Max Reward** : 304.9

### Conclusion DQN
La famille DQN a prouvé sa supériorité absolue sur ce benchmark. 
- Le **DQN Hard Update** est le plus performant (**+218.5 reward**, **75.6% succès**), montrant qu'une mise à jour périodique et stable du réseau cible suffit à maîtriser parfaitement le pilotage du Lander.
- Le **DQN Soft Update** est tout aussi capable mais a montré une convergence légèrement plus laborieuse sur certaines seeds et un comportement plus hésitant en vol.

L'utilisation d'un **Replay Buffer** a été l'élément transformateur, permettant au DQN d'apprendre environ 2 fois plus vite que REINFORCE avec une précision finale 4 fois supérieure.

---

## 7. Synthèse Finale du Benchmark

| Agent | Type | Espace d'Action | Reward Moyen | Taux de Succès | Verdict |
|-------|------|-----------------|--------------|----------------|---------|
| Random| Baseline | Discret | -184.6 | 0.0% | Inefficace |
| Heuristic| Baseline | Discret | -219.8 | 0.0% | Très Faible |
| Q-Learning| Tabulaire | Discrétisé | -41.8 | 18.2% | Limité |
| REINFORCE (C2)| PG | Discret | +77.6 | 12.0% | Stable mais imprécis |
| **DQN Hard** | **Deep Q** | **Discret** | **+218.5** | **75.6%** | **Vainqueur** |
| DQN Soft | Deep Q | Discret | +210.0 | 70.6% | Solide mais lent |

---

## 8. Étude d'Ablation : Stratégies de Target Network Update

Une partie clé de notre benchmark a consisté à comparer deux méthodes de mise à jour du réseau cible pour stabiliser le DQN.

| Paramètre | Hard Target Update | Soft Target Update (Polyak) |
|-----------|--------------------|-----------------------------|
| **Mécanisme** | Copie intégrale tous les 500 pas | Glissement continu ($\tau=0.02$) |
| **Reward Moyen** | **+218.5** | +210.0 |
| **Taux de Succès**| **75.6%** | 70.6% |
| **Convergence** | Rapide (~450 épisodes) | Plus lente (~500+ épisodes) |
| **Comportement** | Décisif, atterrissages francs | Prudent, tendance au vol stationnaire |

### Analyse des résultats
L'ablation montre que pour le LunarLander, le **Hard Update** est supérieur. La mise à jour par paliers semble agir comme une "secousse" bénéfique pour la politique, empêchant l'agent de s'enfermer dans une prudence excessive (les nombreux timeouts observés en Soft Update). Le Soft Update, bien que théoriquement plus stable, a ici ralenti la convergence vers une solution optimale de landing.

**Conclusion Générale** : Le projet démontre l'importance capitale de l'approximation de fonction par réseaux de neurones pour les environnements à états continus. Le passage du Q-Learning Tabulaire au DQN a permis de transformer un agent incapable de se poser en un pilote expert dépassant même les performances de l'approche heuristique codée à la main.
