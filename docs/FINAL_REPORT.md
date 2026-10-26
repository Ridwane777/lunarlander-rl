# Project Report — LunarLander-v3 
**Author: Ridwane777**

## 1. Introduction and Problem Definition

The **LunarLander-v3** project trains an intelligent agent to pilot a lunar module and land it softly on a target zone marked by two yellow flags. 

### 1.1 Measurable Objective
The main objective is a mean score of **200 points or more over 100 consecutive episodes**. It is measured by our deterministic evaluation pipeline, which computes the mean reward and its 95% confidence interval.

### 1.2 Technical Challenges
Piloting a lunar module raises two major challenges:
1. **Dimensionality**: The state space is continuous (8 dimensions), which makes classic tabular approaches ineffective.
2. **Stability**: The agent must balance fuel (economy) against speed (safety) while handling complex kinematics.

## 2. Mathematical Framework

The problem is formalized as a **Markov Decision Process (MDP)** defined by the tuple $(S, A, P, R, \gamma)$.

### 2.1 State Space ($S$)
The state is an 8-dimensional vector containing:
- The module's $(x, y)$ coordinates.
- Linear velocities $(v_x, v_y)$.
- Tilt angle $\theta$ and its angular velocity $\omega$.
- Two booleans indicating whether each leg touches the ground.

### 2.2 Action Space ($A$)
We use the **discrete** action space (4 actions):
- 0: Do nothing.
- 1: Fire the left engine.
- 2: Fire the main engine (upward thrust).
- 3: Fire the right engine.

### 2.3 Reward Function ($R$)
The reward is designed to guide the agent:
- Successful landing at the center: +100 to +140 points.
- Crash: -100 points.
- Each leg-ground contact: +10 points.
- Main engine use: -0.3 point per step (encourages fuel economy).

### 2.4 Transition Probabilities ($P$)
The transition kernel $P(s' | s, a)$ gives the probability of reaching state $s'$ when the agent takes action $a$ in state $s$. In LunarLander, these transitions are determined by the Box2D physics engine (gravity, thrust, collisions).

### 2.5 Discount Factor ($\gamma$)
The parameter $\gamma \in [0, 1]$ (set to 0.99) sets the weight given to future rewards. A $\gamma$ close to 1 pushes the agent toward a long-term strategy (anticipating the final landing).

### 2.6 Learning Algorithms
We compared two algorithm families:
1. **Value-Based (Off-policy)**: **DQN (Deep Q-Network)**, which uses a neural network to approximate $Q(s, a)$ by minimizing the Bellman error:
   $$L(\theta) = \mathbb{E} \left[ (R + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 \right]$$
2. **Policy-Gradient (On-policy)**: **REINFORCE**, which directly optimizes the policy $\pi_\theta(a|s)$ via the policy gradient theorem:
   $$\nabla_\theta J(\theta) = \mathbb{E} \left[ G_t \nabla_\theta \ln \pi_\theta(a_t|s_t) \right]$$

## 3. Experiments and Results

All experiments were run on **5 different seeds** for statistical robustness.

### 3.1 Comparative Benchmark
The results below are computed over 100 final evaluation episodes:

| Algorithm | Mean Reward | 95% CI | Success (%) |
|------------|--------------|--------|------------|
| **DQN (Hard Update)** | **218.5** | **± 5.4** | **75.6%** |
| **DQN (Soft Update)** | 210.0 | ± 5.4 | 70.6% |
| REINFORCE | 77.6 | ± 12.5 | 12.0% |
| Q-Learning (Tabular) | -41.8 | ± 16.1 | 18.2% |
| Heuristic (Baseline) | -219.8 | ± 14.9 | 0.0% |

### 3.2 Convergence Curve Analysis
(Note: see the figures in `plots/benchmark/` for visual detail). 
**Double DQN** is markedly more stable from episode 400 onward, while tabular Q-Learning stalls because discretizing the continuous space loses information.

## 4. Analysis and Discussion

### 4.1 What worked
- **Double DQN**: Using two networks (Policy and Target) greatly reduced Q-value overestimation, a common issue in vanilla DQN.
- **Hard vs Soft Update**: Our tests showed that the "Hard" update (full network copy every 500 steps) performed better on this environment. The "Soft" update (Polyak averaging) tended to make the agent overly cautious, often leading to timeouts (prolonged hovering).
- **Gradient Clipping**: Clipping gradients at 10.0 was crucial to prevent weight explosion in the early learning phases, when the error is largest.

### 4.2 What did not work
- **Tabular Q-Learning**: Discretizing the 8D space caused a combinatorial explosion (the "curse of dimensionality"). Even with fine bins, the agent could not generalize across trajectories.
- **REINFORCE**: Although the agent learns to stabilize, the high gradient variance makes convergence to a perfect landing very erratic. The lack of a Replay Buffer (the algorithm is on-policy) hurts sample efficiency.

## 5. Limitations and Improvement Plan

### 5.1 Identified Limitations
- **Sparse rewards**: Early on, the agent can wander for a long time before touching the ground, delaying the first positive signal.
- **Hyperparameter sensitivity**: The learning rate ($lr$) and exploration factor ($\epsilon$) have a drastic impact. A slightly too high $lr$ can make the network diverge within a few episodes.

### 5.2 Improvement Plan
Several directions could push performance further:
1. **Prioritized Experience Replay (PER)**: Instead of sampling the buffer uniformly, prioritize transitions with the largest Bellman error (where the agent learns the most).
2. **Dueling DQN Architecture**: Separate the state value $V(s)$ from the action advantage $A(s, a)$ to better identify intrinsically good states, regardless of the action.
3. **PPO (Proximal Policy Optimization)**: Use an Actor-Critic method to get the stability of gradient methods while reusing data through a clipped objective.

## 6. Conclusion

The LunarLander RL project shows that **Deep Q-Learning** is a robust solution to the LunarLander-v3 piloting problem. Moving from a limited tabular approach to neural network approximation with Double DQN turned random behavior into expert piloting that succeeds in over 75% of landings with surgical precision.

## 7. Bibliography

1. **Mnih, V., et al. (2015)**. *Human-level control through deep reinforcement learning*. Nature. (The founding DQN paper).
2. **Van Hasselt, H., Guez, A., & Silver, D. (2016)**. *Deep Reinforcement Learning with Double Q-learning*. AAAI Conference on Artificial Intelligence.
3. **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement Learning: An Introduction*. MIT Press. (The definitive reference for the MDP framework).
4. **Gymnasium Documentation**. *Lunar Lander Environment*. [gymnasium.farama.org].
