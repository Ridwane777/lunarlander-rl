# Rapport de Projet — LunarLander-v3 
**Auteur : Ridwane777**

## 1. Introduction et Définition du Problème

Le projet **LunarLander-v3** consiste à entraîner un agent intelligent capable de piloter un module lunaire pour le faire atterrir en douceur sur une zone cible définie par deux drapeaux jaunes. 

### 1.1 Objectif Mesurable
L'objectif principal est d'atteindre un score moyen de **200 points ou plus sur 100 épisodes consécutifs**. Cet objectif est mesurable via notre pipeline d'évaluation déterministe qui calcule le reward moyen ainsi que l'intervalle de confiance à 95%.

### 1.2 Enjeux Techniques
Le pilotage d'un module lunaire pose deux défis majeurs :
1. **Dimensionnalité** : L'espace des états est continu (8 dimensions), ce qui rend les approches tabulaires classiques inefficaces.
2. **Stabilité** : L'agent doit apprendre à équilibrer son carburant (économie) et sa vitesse (sécurité) tout en gérant une cinématique complexe.

## 2. Cadre Mathématique

Le problème est formalisé comme un **Processus de Décision de Markov (MDP)** défini par le quintuplet $(S, A, P, R, \gamma)$.

### 2.1 Espace des États ($S$)
L'état est un vecteur de dimension 8 comprenant :
- Coordonnées $(x, y)$ du module.
- Vitesses linéaires $(v_x, v_y)$.
- Angle d'inclinaison $\theta$ et sa vitesse angulaire $\omega$.
- Deux variables booléennes indiquant si chaque pied touche le sol.

### 2.2 Espace des Actions ($A$)
Nous utilisons l'espace d'actions **discret** (4 actions) :
- 0 : Ne rien faire.
- 1 : Allumer le moteur gauche.
- 2 : Allumer le moteur principal (poussée vers le haut).
- 3 : Allumer le moteur droit.

### 2.3 Fonction de Récompense ($R$)
La récompense est conçue pour guider l'agent :
- Atterrissage réussi au centre : +100 à +140 points.
- Crash : -100 points.
- Chaque contact de jambe au sol : +10 points.
- Utilisation du moteur principal : -0.3 point par step (incitation à l'économie).

### 2.4 Probabilités de Transition ($P$)
Le noyau de transition $P(s' | s, a)$ définit la probabilité de passer à l'état $s'$ sachant que l'agent a pris l'action $a$ dans l'état $s$. Dans LunarLander, ces transitions sont déterminées par le moteur physique Box2D (gravité, poussée, collisions).

### 2.5 Facteur de Remise ($\gamma$)
Le paramètre $\gamma \in [0, 1]$ (fixé à 0.99) définit l'importance accordée aux récompenses futures. Un $\gamma$ proche de 1 incite l'agent à adopter une stratégie de long terme (anticipation de l'atterrissage final).

### 2.6 Algorithmes d'Apprentissage
Nous avons comparé deux familles d'algorithmes :
1. **Value-Based (Off-policy)** : Le **DQN (Deep Q-Network)** qui utilise un réseau de neurones pour approximer la fonction $Q(s, a)$ en minimisant l'erreur de Bellman :
   $$L(\theta) = \mathbb{E} \left[ (R + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 \right]$$
2. **Policy-Gradient (On-policy)** : **REINFORCE** qui optimise directement la politique $\pi_\theta(a|s)$ via le théorème du gradient de politique :
   $$\nabla_\theta J(\theta) = \mathbb{E} \left[ G_t \nabla_\theta \ln \pi_\theta(a_t|s_t) \right]$$

## 3. Expériences et Résultats

Nous avons mené nos expériences sur **5 graines (seeds)** différentes pour garantir la robustesse statistique.

### 3.1 Benchmark Comparatif
Les résultats ci-dessous sont calculés sur 100 épisodes d'évaluation finale :

| Algorithme | Reward Moyen | IC 95% | Succès (%) |
|------------|--------------|--------|------------|
| **DQN (Hard Update)** | **218.5** | **± 5.4** | **75.6%** |
| **DQN (Soft Update)** | 210.0 | ± 5.4 | 70.6% |
| REINFORCE | 77.6 | ± 12.5 | 12.0% |
| Q-Learning (Tabulaire) | -41.8 | ± 16.1 | 18.2% |
| Heuristic (Baseline) | -219.8 | ± 14.9 | 0.0% |

### 3.2 Analyse des Courbes de Convergence
(Note : Voir les figures dans `plots/benchmark/` pour le détail visuel). 
Le **Double DQN** montre une stabilité supérieure dès le 400ème épisode, tandis que le Q-Learning tabulaire stagne à cause de la perte d'information liée à la discrétisation de l'espace continu.

## 4. Analyse et Discussion

### 4.1 Ce qui a fonctionné
- **Double DQN** : L'utilisation de deux réseaux (Principal et Target) a permis de réduire considérablement la surestimation des Q-valeurs, un problème fréquent dans le DQN classique.
- **Hard vs Soft Update** : Nos tests ont montré que la mise à jour "Hard" (copie intégrale du réseau tous les 500 pas) était plus performante sur cet environnement. La mise à jour "Soft" (Polyak averaging) a eu tendance à rendre l'agent trop prudent, menant souvent à des timeouts (vol stationnaire prolongé).
- **Gradient Clipping** : L'écrêtage des gradients à 10.0 a été crucial pour empêcher l'explosion des poids lors des premières phases d'apprentissage où l'erreur est maximale.

### 4.2 Ce qui n'a pas fonctionné
- **Q-Learning Tabulaire** : La discrétisation de l'espace 8D a créé une explosion combinatoire ("malédiction de la dimensionnalité"). Même avec un grain fin, l'agent ne parvenait pas à généraliser les trajectoires.
- **REINFORCE** : Bien que l'agent apprenne à se stabiliser, la variance élevée du gradient rend la convergence vers un atterrissage parfait très aléatoire. L'absence de "Replay Buffer" (car l'algorithme est on-policy) freine l'efficacité de l'échantillonnage.

## 5. Limites et Plan d'Amélioration

### 5.1 Limites Identifiées
- **Parcimonie des récompenses** : Dans les premières phases, l'agent peut errer longtemps avant de toucher le sol, retardant le premier signal positif.
- **Sensibilité aux Hyperparamètres** : Le taux d'apprentissage ($lr$) et le facteur d'exploration ($\epsilon$) ont un impact drastique. Un $lr$ légèrement trop élevé peut faire diverger le réseau en quelques épisodes.

### 5.2 Plan d'Amélioration
Pour dépasser les performances actuelles, plusieurs pistes sont identifiées :
1. **Prioritized Experience Replay (PER)** : Au lieu d'échantillonner le buffer uniformément, donner la priorité aux transitions où l'erreur de Bellman est la plus forte (là où l'agent apprend le plus).
2. **Dueling DQN Architecture** : Séparer l'estimation de la valeur de l'état $V(s)$ et de l'avantage de l'action $A(s, a)$ pour mieux comprendre quels états sont intrinsèquement bons, indépendamment de l'action.
3. **PPO (Proximal Policy Optimization)** : Utiliser une méthode Actor-Critic pour bénéficier de la stabilité des méthodes de gradient tout en réutilisant les données via un clipping de la fonction d'objectif.

## 6. Conclusion

Le projet LunarLander RL a permis de démontrer que le **Deep Q-Learning** est une solution robuste pour résoudre le problème de pilotage du LunarLander-v3. En passant d'une approche tabulaire limitée à une approximation par réseau de neurones avec Double DQN, nous avons réussi à transformer un comportement aléatoire en un pilotage expert capable de réussir plus de 75% de ses atterrissages avec une précision chirurgicale.

## 7. Bibliographie

1. **Mnih, V., et al. (2015)**. *Human-level control through deep reinforcement learning*. Nature. (Papier fondateur du DQN).
2. **Van Hasselt, H., Guez, A., & Silver, D. (2016)**. *Deep Reinforcement Learning with Double Q-learning*. AAAI Conference on Artificial Intelligence.
3. **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement Learning: An Introduction*. MIT Press. (La référence absolue pour le cadre mathématique MDP).
4. **Gymnasium Documentation**. *Lunar Lander Environment*. [gymnasium.farama.org].
