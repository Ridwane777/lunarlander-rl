# Optimisations du Deep Q-Network (DQN) sur LunarLander-v3

Ce document résume l'analyse comparative entre l'implémentation du DQN du projet et l'implémentation d'un agent public hautement performant ([reinforcement_learning_lunar_lander_v3](https://github.com/IgorComune/reinforcement_learning_lunar_lander_v3)).

Bien que l'architecture du réseau de neurones (3 couches denses avec 128 neurones cachés) soit quasi-identique, plusieurs différences cruciales dans le "control flow" (gestion de l'apprentissage) expliquent la différence de performance.

## 1. Fréquence de mise à jour (`update_every`) 🏆
**L'approche initiale** : Le réseau subit une passe de rétropropagation (backpropagation) à **chaque itération** de l'environnement (à chaque appel de `env.step()`).
**L'approche optimisée** : Le réseau n'est mis à jour que **tous les 4 steps**.

**Pourquoi c'est mieux :**
- C'est une technique standard issue du papier original de DeepMind.
- Cela permet d'accumuler de nouvelles expériences dans le Replay Buffer entre chaque mise à jour, ce qui brise la corrélation temporelle des échantillons.
- Empêche l'agent de "sur-apprendre" (overfitting) sur la dynamique très récente de son vol.
- Divise par 4 le coût computationnel de l'entraînement.

## 2. Décroissance du taux d'exploration (`epsilon_decay`)
**L'approche initiale** : L'exploration décroît avec une multiplication de `0.995` à **chaque step**. Résultat : l'exploration (`epsilon`) chute de 1.0 à 0.01 en moins de 1000 itérations (soit 1 ou 2 épisodes), ce qui ne laisse pas le temps à l'agent de découvrir l'environnement.
**L'approche optimisée** : Une décroissance de `0.996` **par épisode** (ou `0.9995` par step) permet à l'agent d'explorer activement pendant une cinquantaine d'épisodes avant de figer sa stratégie. 

## 3. Écrêtage des gradients (Gradient Clipping)
**L'approche initiale** : Clipping fixé à `10.0`.
**L'approche optimisée** : Clipping agressif fixé à `1.0`.

**Pourquoi c'est mieux :** 
Sur LunarLander, un crash inattendu ou l'allumage prolongé d'un propulseur peut générer des erreurs de Différence Temporelle (TD Error) énormes dans le calcul de la *Loss*. Un clipping à `1.0` garantit que les poids du réseau ne font pas de sauts brutaux qui viendraient "casser" l'apprentissage précédent.

## 4. Stratégie de mise à jour du Target Network
L'agent optimisé utilise une **Hard Update** (copie complète des poids) tous les 1000 cycles d'apprentissage, offrant une cible parfaitement immuable pendant de longs instants.
Notre modèle emploie une **Soft Update** (méthode de Polyak, `tau=0.005`) à chaque mise à jour. Bien que la *Soft Update* soit plus moderne (utilisée en algorithmes continus comme SAC), la *Hard Update* s'avère souvent un peu plus stable pour un DQN classique standard.

---

## Conclusion
Les deux optimisations immédiates ayant le plus d'impact pour résoudre LunarLander-v3 plus rapidement sont :
1. Intégrer un `update_every = 4` pour désaturer la mise à jour des poids.
2. Baisser le `grad_clip` à `1.0` pour plus de stabilité face aux récompenses négatives sévères (-100 en cas de crash).
