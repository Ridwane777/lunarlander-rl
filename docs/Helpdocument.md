# 🛡️ Guide de Préparation à la Soutenance : Projet LunarLander RL (LunarLander)

Ce document est votre "antisèche" ultime. Il contient toutes les explications, vulgarisations et définitions nécessaires pour répondre à n'importe quelle question d'un jury lors de votre défense de projet.

---

## 1. Concepts Fondamentaux du Machine Learning & RL

### Le Reinforcement Learning (RL) vulgarisé
Le RL fonctionne sur le principe de l'apprentissage par **essai-erreur**.
- **L'Agent** (votre IA) observe un **État** (les capteurs du vaisseau : position, vitesse, angle).
- Il prend une **Action** (allumer le moteur gauche, droit, etc.).
- L'**Environnement** (le moteur physique Box2D) réagit à cette action et renvoie un nouvel état et une **Récompense** (Reward).
- Le but de l'agent est de maximiser la somme de ses récompenses futures.

### C'est quoi une "Policy" (Politique $\pi$) ?
La Policy (notée $\pi$) est le "cerveau" ou la "stratégie" de l'agent. Mathématiquement, c'est une fonction qui prend un État en entrée et renvoie l'Action à effectuer. 
- Une **politique déterministe** dira : "Si je suis à 10m d'altitude, j'allume TOUJOURS le moteur à fond".
- Une **politique stochastique** (utilisée en REINFORCE) dira : "Si je suis à 10m d'altitude, j'ai 80% de chances d'allumer le moteur, et 20% de chances de ne rien faire".
Tout le but du Reinforcement Learning est de trouver la **Politique Optimale** ($\pi^*$), c'est-à-dire la stratégie parfaite pour atterrir à chaque fois.

### L'Exploration vs L'Exploitation (Le dilemme)
C'est le concept central du RL.
- **Exploration** : L'agent tente des actions complètement au hasard pour découvrir l'environnement (et potentiellement trouver comment atterrir).
- **Exploitation** : L'agent utilise ce qu'il a déjà appris pour maximiser son score.
- **Pourquoi c'est crucial ?** Si on exploite trop tôt, l'agent trouvera une stratégie nulle mais "sécurisée" (ex: flotter en l'air pour ne pas s'écraser, c'est le fameux *Hovering Local Minimum* qu'on a vu). S'il explore trop, il n'optimisera jamais son atterrissage. On gère ça avec le paramètre **Epsilon ($\epsilon$)** qui diminue progressivement (le *decay*).

### C'est quoi une "Seed" ?
Une seed (graine aléatoire) est un nombre qui initialise les générateurs d'aléatoire de l'ordinateur. 
- En informatique, l'aléatoire n'existe pas, c'est du "pseudo-aléatoire". Si on utilise la même seed, l'agent fera exactement les mêmes actions au hasard et le terrain aura la même forme.
- **Pourquoi 5 seeds ?** En RL, un agent peut avoir "de la chance" lors d'un entraînement. Pour prouver que notre algorithme fonctionne vraiment, on doit l'entraîner 5 fois avec des destins initiaux différents. Si les 5 convergent, l'algorithme est robuste.

### C'est quoi un Réseau de Neurones ?
C'est un modèle mathématique inspiré du cerveau humain.
- En entrée : Les 8 chiffres des capteurs du vaisseau.
- En sortie : 4 chiffres (les probabilités des actions pour REINFORCE, ou les scores "Q-Values" de chaque action pour le DQN).
- À l'intérieur : Des "poids" mathématiques. Au début, ils sont aléatoires (l'agent est bête). À chaque erreur, l'algorithme de "Rétropropagation du gradient" (Backpropagation) ajuste un peu ces poids pour que la prochaine fois, l'agent prenne une meilleure décision.

---

## 2. Décryptage des Modèles Implémentés

### A. Q-Learning (La méthode Tabulaire)
**Le principe** : On crée un immense tableau Excel. En ligne : les états. En colonne : les actions. Dans les cases : la note (Q-value) de cette action dans cet état.
- **Le problème sur LunarLander** : Les états (vitesse, angle) sont des nombres continus (avec des virgules). Il y a une infinité d'états ! 
- **La solution mise en place** : La "Discrétisation". On a découpé l'espace en "boîtes" (bins). Par exemple, un angle de 12.1° et 12.8° tombent dans la même boîte.
- **Pourquoi ça échoue au benchmark ?** On appelle ça la *Malédiction de la dimensionnalité*. Même avec 8 boîtes par capteur, ça fait $8^8$ = 16 millions de cases à remplir. L'agent ne vivra jamais assez longtemps pour remplir sa table Excel par l'expérience.

### B. REINFORCE (Policy Gradient Monte-Carlo)
**Le principe** : Au lieu de noter des actions, le réseau de neurones recrache directement les pourcentages (probabilités) de prendre chaque action.
L'agent joue un épisode complet (jusqu'à s'écraser ou atterrir). À la fin, il regarde son score total.
Si le score est bon, on dit au réseau "augmente la probabilité de toutes les actions que tu as prises". Si c'est nul, "diminue-les".
- **Pourquoi ça échoue ici ?** Le problème de la *Haute Variance*. Un épisode entier donne des dizaines d'actions. L'agent s'écrase (-100) mais certaines actions au milieu étaient peut-être bonnes. Comme on pénalise "tout l'épisode" en bloc, l'apprentissage est très lent et bruité. 

### C. Deep Q-Network / DQN (Le grand gagnant)
**Le principe** : C'est le Q-learning, mais la table Excel est remplacée par un Réseau de Neurones qui "devine" les notes (Q-values) même pour des situations qu'il n'a jamais vues (Généralisation).
**Les trois piliers qui le font marcher :**
1. **L'Experience Replay (Le Buffer)** : Au lieu d'apprendre sur la frame qu'il vient de vivre et l'oublier, il la stocke dans une "mémoire" (Replay Buffer). À chaque pas, il pioche un *Batch* (128 souvenirs mélangés) dans le passé pour s'entraîner. Ça évite que l'agent n'oublie comment atterrir s'il passe 10 minutes à voler.
2. **Le Target Network** : Pour calculer son erreur, le réseau compare sa note à la "vraie" note cible. Mais comme le réseau s'entraîne, cette "vraie" note bouge tout le temps ! C'est comme tirer sur une cible mouvante. On crée donc un 2ème réseau (le Target Network) qui est figé, et on s'en sert comme point de référence stable.
3. **Le Double DQN** : Pour éviter que le réseau ne soit trop optimiste (surestimation), le réseau A choisit la meilleure action, mais c'est le réseau B (Target) qui donne la note de cette action.

---

## 3. Zoom sur le Réseau de Neurones (`PolicyNetwork`)

Si le jury vous demande comment l'agent "réfléchit" concrètement à partir des capteurs, voici le détail du réseau utilisé (type MLP - Multi-Layer Perceptron) :

### Architecture pas à pas :
1.  **Entrée (8 neurones)** : Reçoit les 8 données de `LunarLander-v3` (Coordonnées X/Y, Vitesses VX/VY, Angle, Vitesse Angulaire, Contacts jambes).
2.  **Couches Cachées (ex: 128 neurones)** :
    *   **`nn.Linear`** : C'est une opération de multiplication matricielle ($Y = WX + B$). Chaque neurone de la couche précédente est connecté à chaque neurone de la suivante par un "poids" ($W$) qui définit l'importance de l'information.
    *   **`nn.ReLU` (Activation)** : C'est l'ingrédient magique. Elle bloque les signaux négatifs ($f(x) = \max(0, x)$). Cela permet au réseau d'apprendre des fonctions mathématiques **non-linéaires**. Sans ReLU, votre IA ne pourrait apprendre que des trajectoires en ligne droite !
3.  **Sortie (4 neurones)** : Un pour chaque action possible (Rien, Gauche, Principal, Droit).

### La différence entre DQN et REINFORCE en sortie :
-   **DQN** : Les 4 neurones sortent des "Q-Values" (des notes de -200 à +200). L'agent prend l'action qui a la meilleure note (le `argmax`).
-   **REINFORCE** : On ajoute une couche **Softmax** à la fin. Elle transforme les sorties en **pourcentages** (probabilités). L'agent tire au sort son action selon ces pourcentages. C'est ce qu'on appelle une **politique stochastique**.

---

## 4. L'Étude d'Ablation (Hard vs Soft Update)

**Qu'est-ce qu'une ablation ?** En IA, c'est l'action de modifier ou de retirer un mécanisme précis de l'algorithme pour prouver son utilité.
Ici, nous avons étudié comment mettre à jour le **Target Network** (la cible figée du DQN).
- **Hard Update** : Tous les 500 pas, on prend le réseau principal, et on copie/colle brutalement tous ses poids dans le Target Network. 
  - *Conséquence* : L'apprentissage est parfois un peu saccadé (on change les règles du jeu d'un coup).
- **Soft Update (Polyak)** : À chaque pas, le Target Network "glisse" de 2% ($\tau = 0.02$) vers le réseau principal.
  - *Conséquence* : La cible bouge tout doucement, l'apprentissage est beaucoup plus lisse.
- **Résultat de notre benchmark** : Les deux marchent ! La Soft Update a fait un entraînement plus facile, mais bizarrement, la Hard Update a créé une politique très légèrement plus solide à l'évaluation (score de 218 vs 210).

---

## 5. Dictionnaire des Hyperparamètres
Si le jury vous demande ce que font ces lignes de vos fichiers YAML :
- **`gamma` (0.99)** : C'est la myopie de l'agent. À 0.99, il se soucie énormément de l'avenir (il sait que l'atterrissage +100 arrivera dans longtemps). À 0, il ne pense qu'à la seconde qui suit.
- **`lr` (Learning Rate - 0.0003)** : La vitesse d'apprentissage. S'il est trop grand, le réseau modifie ses connexions trop violemment et oublie tout. Trop petit, l'entraînement prendrait des mois.
- **`batch_size` (128)** : Le nombre de souvenirs qu'il remémore en même temps pour faire ses maths.
- **`epsilon_decay` (0.995 par épisode)** : À la fin de chaque partie, on multiplie l'exploration par 0.995. Au bout de 100 parties, l'agent n'explorera plus qu'à 60%, au bout de 500, presque plus du tout (1%).
- **`update_every` (4)** : L'agent interagit 4 fois avec l'environnement avant de faire des mathématiques pour apprendre. Cela permet au buffer de se remplir de choses nouvelles et soulage énormément le processeur.

---

## 6. Concepts Avancés : On-policy vs Off-policy

C'est une distinction majeure dans l'apprentissage par renforcement qui définit comment l'agent utilise son expérience pour apprendre.

### On-policy (Apprentissage "en direct")
L'agent apprend à améliorer sa stratégie en utilisant uniquement les données collectées par sa politique **actuelle**. Une fois que les données ont servi à une mise à jour, elles sont jetées car elles ne correspondent plus à la nouvelle politique.
- **Agent concerné** : **REINFORCE**.
- **Avantage** : Plus stable mathématiquement, car les données sont toujours "fraîches".
- **Inconvénient** : Très inefficace en termes de données (on jette tout après chaque mise à jour).

### Off-policy (Apprentissage "par l'expérience")
L'agent peut apprendre à partir de n'importe quelle donnée, qu'elle vienne de sa version actuelle, d'une version très ancienne de lui-même, ou même d'un autre agent. Ces algorithmes utilisent généralement un **Replay Buffer** pour stocker et réutiliser les expériences.
- **Agents concernés** : **Q-Learning** et **DQN**.
- **Avantage** : Très efficace car chaque expérience peut être réutilisée plusieurs fois pour l'entraînement.
- **Inconvénient** : Plus complexe à stabiliser (nécessite un Target Network pour éviter les divergences).

---

## 7. Synthèse de nos agents

| Algorithme | Type | Mémoire | Mode |
|------------|------|---------|------|
| **Random** | Baseline | Non | N/A |
| **Heuristic** | Baseline | Non | N/A |
| **Q-Learning** | Tabulaire | Non* | **Off-policy** |
| **REINFORCE** | Policy Gradient | Non | **On-policy** |
| **DQN** | Value-based | **Oui** (Replay Buffer) | **Off-policy** |

*\*Le Q-Learning tabulaire est théoriquement Off-policy car il apprend la Q-valeur optimale indépendamment de l'action choisie, mais dans notre implémentation simple, il n'utilise pas de Replay Buffer.*

---

## 8. Comment lire les Métriques du Benchmark ?

- **`MEAN_R` (Score Moyen)** : La moyenne de vos 100 épisodes d'évaluation. Pour valider le projet Ridwane777, il fallait que cette valeur dépasse 200.
- **`±CI95` (Intervalle de Confiance à 95%)** : C'est la marge d'erreur. Si vous avez 210 ± 5, ça veut dire que statistiquement, vous êtes sûrs à 95% que votre algorithme donne un agent dont le vrai niveau se situe entre 205 et 215. Plus le CI95 est bas, plus votre algorithme est fiable et vos 5 seeds ont donné la même chose.
- **`SUCCESS%`** : Le pourcentage d'épisodes qui se sont terminés par un atterrissage parfait (score pur $\ge$ 200). 75%, c'est un excellent résultat sur ce problème !

---

