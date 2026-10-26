# Analyse de l'Optimisation et Plan d'Amélioration
**Projet LunarLander RL**

Ce document détaille l'analyse technique des performances de nos agents et définit une feuille de route stratégique pour dépasser les limites actuelles du benchmark.

---

## 1. Justification de l'Architecture (MLP 128x128)

Le choix d'un réseau Multi-Layer Perceptron (MLP) avec deux couches cachées de 128 neurones chacune repose sur une analyse du compromis biais-variance :

- **Capacité de Représentation** : L'espace d'état continu (8D) nécessite une capacité de généralisation que les architectures plus fines (64 neurones) peinent à atteindre, comme observé lors de nos tests préliminaires sur REINFORCE (variance excessive).
- **Stabilité Numérique** : 128 neurones constituent le "sweet spot" pour le DQN. Au-delà (ex: 256+), le risque d'overfitting sur les trajectoires initiales de crash augmente, et le temps de calcul par step impacte négativement la fréquence d'échantillonnage du Replay Buffer.
- **Activation ReLU** : L'utilisation de `ReLU` permet d'apprendre les non-linéarités de la cinématique (gravité vs poussée) tout en évitant le problème de disparition du gradient.

---

## 2. Analyse des Stratégies de Stabilisation

### 2.1 Le Biais d'Optimisme (Double DQN)
Le DQN classique a tendance à surestimer les Q-valeurs. L'implémentation du **Double DQN** a été l'optimisation la plus impactante, permettant d'atteindre le score de 218.5 en découplant :
1.  Le choix de l'action : $a^* = \arg\max_a Q(s', a; \theta_{policy})$
2.  L'évaluation de l'action : $Target = R + \gamma Q(s', a^*; \theta_{target})$

### 2.2 Duel des Target Updates (Hard vs Soft)
Nos données révèlent un paradoxe intéressant :
- **Soft Update ($\tau=0.02$)** : Produit un apprentissage stable et lisse, mais l'agent devient souvent "trop prudent" (vol stationnaire infini).
- **Hard Update (C=500)** : Bien que plus instable, elle force l'agent à réévaluer brutalement ses priorités tous les 500 pas, ce qui semble bénéfique pour sortir des minima locaux de type "Hovering".

---

## 3. Synthèse des Hyperparamètres Critiques

| Hyperparamètre | Valeur | Impact sur les Données |
|----------------|--------|-------------------------|
| **Learning Rate** | $10^{-3}$ | Garantit la convergence en moins de 600 épisodes. |
| **Epsilon Decay** | $0.995$ | Crucial pour laisser 200 épisodes d'exploration pure. |
| **Buffer Size** | $10^5$ | Permet de conserver des atterrissages réussis sur une longue durée. |
| **Grad Clip** | $10.0$ | Empêche la divergence lors des crashs violents. |

---

## 4. Plan d'Amélioration Basé sur les Données

L'analyse de notre benchmark identifie trois goulots d'étranglement majeurs que ce plan vise à lever.

### 4.1 Efficacité de l'Échantillonnage (PER)
**Problème constaté** : Le DQN échantillonne uniformément, "gaspillant" du temps sur des trajectoires de crash déjà comprises.
- **Solution** : **Prioritized Experience Replay (PER)**.
- **Objectif** : Prioriser les transitions avec une forte erreur TD ($\delta$).
- **Impact attendu** : Réduction du temps d'entraînement de 30% et augmentation de la robustesse sur les seeds difficiles.

### 4.2 Réduction de la Variance (Actor-Critic)
**Problème constaté** : REINFORCE affiche un CI95 de ±11.9, signe d'une instabilité chronique.
- **Solution** : Migration vers **PPO (Proximal Policy Optimization)**.
- **Mécanisme** : Ajout d'une Baseline apprise (Critique) et clipping du ratio de probabilité.
- **Impact attendu** : Convergence stable vers des scores > 250 sans les oscillations actuelles.

### 5.3 Extension au Contrôle Continu
**Problème constaté** : Les actions discrètes (ON/OFF) limitent la finesse de l'atterrissage.
- **Solution** : Utilisation de **SAC (Soft Actor-Critic)** sur LunarLanderContinuous-v3.
- **Justification** : Un contrôle continu de la puissance permettrait d'éliminer les -0.3 pts de pénalité moteur par step en lissant la poussée.

---

## 5. Conclusion de l'Analyse
L'optimisation actuelle a permis de transformer un échec (Q-Learning) en succès (DQN > 200). Cependant, pour atteindre un pilotage "humain" parfait (>280 pts), l'évolution vers des méthodes Actor-Critic et un échantillonnage prioritaire est la suite logique de ce projet.
