# Optimization Analysis and Improvement Plan
**LunarLander RL Project**

This document details the technical analysis of our agents' performance and sets a strategic roadmap to move past the current benchmark limits.

---

## 1. Architecture Rationale (MLP 128x128)

The choice of a Multi-Layer Perceptron (MLP) with two hidden layers of 128 neurons each rests on a bias-variance trade-off analysis:

- **Representational Capacity**: The continuous (8D) state space requires a generalization capacity that thinner architectures (64 neurons) struggle to reach, as seen in our preliminary REINFORCE tests (excessive variance).
- **Numerical Stability**: 128 neurons is the "sweet spot" for DQN. Beyond that (e.g. 256+), the risk of overfitting on early crash trajectories grows, and per-step compute time slows down Replay Buffer sampling.
- **ReLU Activation**: `ReLU` lets the network learn the non-linear kinematics (gravity vs thrust) while avoiding vanishing gradients.

---

## 2. Stabilization Strategy Analysis

### 2.1 Optimism Bias (Double DQN)
Vanilla DQN tends to overestimate Q-values. Implementing **Double DQN** was the most impactful optimization, reaching the 218.5 score by decoupling:
1.  Action selection: $a^* = \arg\max_a Q(s', a; \theta_{policy})$
2.  Action evaluation: $Target = R + \gamma Q(s', a^*; \theta_{target})$

### 2.2 Target Update Showdown (Hard vs Soft)
Our data reveals an interesting paradox:
- **Soft Update ($\tau=0.02$)**: Produces smooth, stable learning, but the agent often becomes "too cautious" (endless hovering).
- **Hard Update (C=500)**: Although less stable, it forces the agent to abruptly re-evaluate its priorities every 500 steps, which appears to help it escape "Hovering"-type local minima.

---

## 3. Critical Hyperparameters

| Hyperparameter | Value | Impact on the Data |
|----------------|--------|-------------------------|
| **Learning Rate** | $10^{-3}$ | Ensures convergence in under 600 episodes. |
| **Epsilon Decay** | $0.995$ | Crucial to allow 200 episodes of pure exploration. |
| **Buffer Size** | $10^5$ | Keeps successful landings in memory for a long time. |
| **Grad Clip** | $10.0$ | Prevents divergence during violent crashes. |

---

## 4. Data-Driven Improvement Plan

Our benchmark analysis identifies three major bottlenecks that this plan aims to remove.

### 4.1 Sample Efficiency (PER)
**Observed problem**: DQN samples uniformly, "wasting" time on crash trajectories it already understands.
- **Solution**: **Prioritized Experience Replay (PER)**.
- **Goal**: Prioritize transitions with a high TD error ($\delta$).
- **Expected impact**: 30% shorter training time and more robustness on difficult seeds.

### 4.2 Variance Reduction (Actor-Critic)
**Observed problem**: REINFORCE shows a CI95 of ±11.9, a sign of chronic instability.
- **Solution**: Move to **PPO (Proximal Policy Optimization)**.
- **Mechanism**: Add a learned baseline (Critic) and clip the probability ratio.
- **Expected impact**: Stable convergence toward scores > 250 without the current oscillations.

### 5.3 Extension to Continuous Control
**Observed problem**: Discrete (ON/OFF) actions limit landing finesse.
- **Solution**: Use **SAC (Soft Actor-Critic)** on LunarLanderContinuous-v3.
- **Rationale**: Continuous power control would remove the -0.3 pts engine penalty per step by smoothing thrust.

---

## 5. Analysis Conclusion
The current optimization turned a failure (Q-Learning) into a success (DQN > 200). However, reaching perfect "human-like" piloting (>280 pts) calls for Actor-Critic methods and prioritized sampling as the logical next step for this project.
