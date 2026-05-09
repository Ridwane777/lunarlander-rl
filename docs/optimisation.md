# Optimisation de l'Apprentissage — LunarLander-v3

Ce document détaille les choix techniques et le réglage des hyperparamètres qui ont permis d'atteindre les performances documentées dans le benchmark.

## 1. Architecture des Réseaux de Neurones

### Pourquoi 128 neurones ? (Justification Technique)
Le choix d'une architecture avec des couches cachées de **128 neurones** pour le DQN découle de nos observations sur les agents précédents :
- **Retour d'expérience REINFORCE** : Lors de nos tests sur REINFORCE, nous avons constaté qu'un réseau de 64 neurones était trop limité pour capturer la précision nécessaire à l'atterrissage, entraînant une forte variance. L'augmentation de la capacité a été la clé du succès.
- **Le "Sweet Spot" du DQN** : Pour le DQN, 128 neurones constituent le point d'équilibre idéal. Cela offre une capacité de représentation supérieure à ce que nous avions initialement sur REINFORCE, permettant de modéliser finement la fonction de valeur Q pour les **4 actions discrètes**, sans introduire la complexité inutile d'un réseau trop large.
- **Environnement Déterministe** : Comme nous évoluons dans une version **sans vent** du LunarLander, la physique est stable. 128 neurones suffisent à apprendre les équations de mouvement du lander (cinématique) sans risquer de mémoriser des bruits aléatoires inexistants.

### Améliorations Architecturales
- **Double DQN (mis en œuvre)** : Pour pallier le biais d'optimisme du DQN classique (surestimation des valeurs d'action), nous utilisons le réseau principal pour choisir l'action et le réseau cible pour l'évaluer.

## 2. Stratégies de Stabilisation

### Target Network Update (Ablation)
Nous avons comparé deux approches pour la mise à jour du réseau cible dans le DQN :
- **Hard Update** (Vainqueur) : Copie totale tous les $C$ pas (ex: 500). Provoque des "sauts" de politique qui aident l'agent à explorer de nouvelles stratégies de descente.
- **Soft Update (Polyak)** : Mise à jour lente et continue ($\tau=0.02$). Bien que théoriquement plus stable, elle a montré une convergence plus lente, l'agent restant souvent bloqué dans un comportement de vol stationnaire trop prudent.

### Normalisation (REINFORCE)
Sans normalisation, les gradients dans REINFORCE sont extrêmement instables car ils dépendent de l'amplitude brute des récompenses. Nous avons optimisé l'apprentissage en normalisant les retours (Retours - Moyenne / Écart-type) à chaque fin d'épisode.

## 3. Réglage des Hyperparamètres Critiques

| Paramètre | Choix | Justification |
|-----------|-------|---------------|
| **Learning Rate** | $10^{-3}$ | Équilibre entre vitesse d'apprentissage et stabilité (évite les oscillations divergentes). |
| **Epsilon Decay** | $0.995$ | Garantit une phase d'exploration suffisante (environ 200 épisodes) avant de devenir purement déterministe. |
| **Batch Size** | $64$ | Taille optimale pour stabiliser le gradient stochastique sans saturer la mémoire. |
| **Gamma** | $0.99$ | Facteur de remise privilégiant la récompense finale (atterrissage) sur les récompenses immédiates. |

## 4. Pistes d'Optimisation Futures

Pour dépasser le taux de succès actuel (75%), plusieurs pistes sont envisageables :
1. **Prioritized Experience Replay (PER)** : Pour que le DQN apprenne plus des erreurs importantes (crashs) que des situations banales.
2. **Dueling DQN** : Séparer l'estimation de la valeur de l'état et de l'avantage de l'action.
3. **PPO (Proximal Policy Optimization)** : Pour stabiliser davantage la politique en limitant les changements de paramètres trop brusques.
