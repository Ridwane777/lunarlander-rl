# LunarLander-v3 Benchmark

This document gathers the training and evaluation results of the different agents.

## 1. Random Agent (Baseline)

The random agent picks its actions uniformly at random. It serves as the comparison baseline for the other algorithms.

### Training
- **Config**: `configs/random.yaml`
- **Seeds**: [0, 1, 2, 3, 4]
- **Max steps**: 50 000

| Seed | Episodes | Mean Reward | Max Reward | Status | Time |
|------|----------|--------------|------------|--------|-------|
| 0    | 528      | -180.5       | 77.1       | Done   | ~60s  |
| 1    | 543      | -178.1       | 28.4       | Done   | ~60s  |
| 2    | 525      | -179.2       | 37.1       | Done   | ~60s  |
| 3    | 532      | -181.8       | 39.3       | Done   | ~60s  |
| 4    | 540      | -186.3       | 74.3       | Done   | ~60s  |

### Evaluation (100 episodes per seed, 500 episodes in total)
- **Mean Reward**: -184.6 ± 9.6
- **Success rate**: 0.0%
- **End cause**: Mostly 'crash' (the agent cannot land).
- **Max Reward**: 72.2

### Hyperparameters
None (purely random agent).

### Conclusion
The random agent performs extremely poorly, with a mean reward of **-184.6**. The success rate is **0%**, confirming that the probability of landing by pure chance is close to zero. This agent serves as the "reference zero" to measure the learning of later agents (DQN, REINFORCE, etc.). Any learning algorithm must beat these scores to be considered functional.

---

## 2. Heuristic Agent (Baseline)

A hand-written proportional-derivative controller (`src/agents/heuristic_agent.py`), adapted
from the heuristic shipped with Gymnasium's LunarLander (MIT license). It steers two targets:
an angle that leans toward the pad, and a hover height that grows with the horizontal
distance to the pad. No learning, no hyperparameters.

### Evaluation (same 100 episodes as every agent, seeds 10000 to 10099)
- **Mean Reward**: 255.4
- **Success rate** (score >= 200): 95.0%
- **Max Reward**: 314.0

The controller is deterministic, so the 5 "seeds" of the benchmark produce identical runs: the
summary table shows ± 6.5 because it treats them as 500 independent episodes, but only 100 are
distinct. Measured on 2026-11-16, Python 3.14, Gymnasium 1.4.0.

### History
The first version only reacted to large tilts and fast falls. It scored -219.8 ± 14.9 with
0% success, worse than random actions (-184.6), so it was not a meaningful reference and was
replaced by the controller above.

### Conclusion
A well-tuned controller that knows the physics beats every learned agent of this benchmark
(255.4 vs 218.5 for the best DQN configuration). The DQN reaches its score without any
knowledge of the dynamics, from the reward alone; that is what the comparison measures, not
which method lands best on this particular, fully known task.

---

## 3. Q-Learning (Tabular) - Config 1

Q-Learning is an off-policy reinforcement learning algorithm. Since the observation space is continuous, it must be discretized to be used in tabular form.

### Training
- **Config**: `configs/q_learning.yaml`
- **Seeds**: [0, 1, 2, 3, 4]
- **Max steps**: 200 000

| Seed | Episodes | Mean Reward | Max Reward | Status | Time |
|------|----------|--------------|------------|--------|-------|
| 0    | 840      | -101.6       | 254.4      | Done   | ~7min |
| 1    | 898      | -108.9       | 254.3      | Done   | ~7min |
| 2    | 810      | -93.6        | 263.7      | Done   | ~7min |
| 3    | 915      | -122.4       | 285.6      | Done   | ~7min |
| 4    | 833      | -91.9        | 268.0      | Done   | ~7min |

### Evaluation (100 episodes per seed, 500 episodes in total)
- **Mean Reward**: -41.8 ± 16.1
- **Success rate**: 18.2%
- **End cause**: Mix of 'crash', 'solved' and 'timeout'.
- **Max Reward**: 285.6

### Hyperparameters
- **Gamma**: 0.99
- **Learning Rate (LR)**: 0.1
- **Epsilon**: 1.0 -> 0.01 (decay 0.995)

### Conclusion
Tabular Q-Learning clearly improves on the baselines. With an **18.2%** success rate, the agent starts to understand how to stabilize and land, even though the discretized space limits its precision. The agent sometimes achieves perfect landings (> 200 points). However, the mean stays negative because crashes are still frequent and heavily penalized.

---

## 4. Q-Learning (Tabular) - Config 2

This version uses longer exploration (`epsilon_decay=0.999`) and a lower learning rate (`lr=0.05`) to see whether slower convergence improves stability.

### Training
- **Config**: `configs/q_learning_v2.yaml`
- **Seeds**: [0, 1, 2, 3, 4]
- **Max steps**: 200 000

| Seed | Episodes | Mean Reward | Max Reward | Status | Time |
|------|----------|--------------|------------|--------|-------|
| 0    | 1031     | -153.2       | 261.8      | Done   | ~7min |
| 1    | 952      | -117.1       | 262.2      | Done   | ~7min |
| 2    | 921      | -117.7       | 257.4      | Done   | ~7min |
| 3    | 997      | -138.1       | 257.0      | Done   | ~7min |
| 4    | 990      | -116.7       | 280.4      | Done   | ~7min |

### Evaluation (100 episodes per seed, 500 episodes in total)
- **Mean Reward**: -92.7 ± 14.4
- **Success rate**: 11.6%
- **End cause**: Mostly 'crash', a few 'solved'.
- **Max Reward**: 280.4

### Hyperparameters
- **Gamma**: 0.99
- **Learning Rate (LR)**: 0.05 (vs 0.1 in Config 1)
- **Epsilon**: 1.0 -> 0.01 (decay 0.999 vs 0.995)

### Conclusion
Despite longer exploration and more cautious learning, Config 2 performs **worse** than Config 1 (-92.7 vs -41.8 mean). The success rate also drops to **11.6%**. The too-low learning rate likely slows convergence excessively, and the extended exploration prevents the agent from stabilizing its policy within the allotted budget (200 000 steps). Config 1 therefore remains our best tabular Q-Learning approach.

### Q-Learning Comparison
| Config | Mean Reward | Success Rate | Key Observation |
|--------|--------------|----------------|-----------------|
| Config 1 (LR 0.1, Decay 0.995) | **-41.8** | **18.2%** | Faster convergence and better exploitation. |
| Config 2 (LR 0.05, Decay 0.999) | -92.7 | 11.6% | Exploration too long, not enough time to converge. |

**Verdict**: **Config 1** is the most effective. It offers the best trade-off between learning speed and exploitation. However, tabular Q-Learning shows its limits against the environment's complexity: even the best configuration fails to reach a positive mean, which shows the need to move to function approximation methods (Deep RL).

---

## 4. REINFORCE (Policy Gradient)

REINFORCE is a Monte-Carlo policy gradient algorithm. Unlike Q-Learning, it directly learns a policy parameterized by a neural network.

### Training
- **Config**: `configs/reinforce.yaml`
- **Seeds**: [0, 1, 2, 3, 4]
- **Max steps**: 200 000

| Seed | Episodes | Mean Reward | Max Reward | Status | Time |
|------|----------|--------------|------------|--------|-------|
| 0    | 828      | -38.4        | 315.2      | Done   | ~5min |
| 1    | 795      | -70.1        | 297.7      | Done   | ~5min |
| 2    | 806      | -41.7        | 299.6      | Done   | ~5min |
| 3    | 824      | -30.3        | 307.1      | Done   | ~5min |
| 4    | 616      | -85.4        | 201.6      | Done   | ~6min |

### Evaluation (100 episodes per seed, 500 episodes in total)
- **Mean Reward**: 39.2 ± 11.9
- **Success rate**: 18.2%
- **End cause**: Mix of 'solved' and 'crash'.
- **Max Reward**: 315.2

### Hyperparameters
- **Gamma**: 0.99
- **Learning Rate (LR)**: 0.001
- **Network**: 1 hidden layer of 128 neurons

### Conclusion
REINFORCE clearly beats Q-Learning on mean reward (from -41.8 to **+39.2**). Although the success rate is identical (**18.2%**), the agent appears to handle failures much better, avoiding overly violent crashes. A neural network captures the continuity of the observation space far more effectively than tabular discretization.

---

## 5. REINFORCE (Policy Gradient) - Config 2

This version uses a wider network (`hidden=256`) and a lower learning rate (`lr=0.0005`) to capture more complex relationships while staying stable.

### Training
- **Config**: `configs/reinforce_v2.yaml`
- **Seeds**: [0, 1, 2, 3, 4]
- **Max steps**: 200 000

| Seed | Episodes | Mean Reward | Max Reward | Status | Time |
|------|----------|--------------|------------|--------|-------|
| 0    | 627      | -50.6        | 166.2      | Done   | ~6min |
| 1    | 674      | -63.6        | 253.2      | Done   | ~6min |
| 2    | 684      | -39.0        | 279.4      | Done   | ~6min |
| 3    | 816      | -61.9        | 257.9      | Done   | ~6min |
| 4    | 695      | -41.6        | 215.4      | Done   | ~6min |

### Evaluation (100 episodes per seed, 500 episodes in total)
- **Mean Reward**: 77.6 ± 8.1
- **Success rate**: 12.0%
- **End cause**: Very few crashes, many 'landed' outside the zone or time limit.
- **Max Reward**: 279.4

### Hyperparameters
- **Gamma**: 0.99
- **Learning Rate (LR)**: 0.0005
- **Network**: 1 hidden layer of 256 neurons

### Conclusion
Config 2, with a wider network and a lower learning rate, is **much more stable** (mean reward of +77.6). However, its raw success rate is lower than Config 1 (12.0% vs 18.2%). This suggests the agent learned a very cautious policy that minimizes crash risk but still lacks the precision to land consistently in the target zone. 

REINFORCE (Config 2) is the first agent to reach a solid, positive mean reward on this complex environment.

### 5.3. REINFORCE Comparison
| Config | Hidden | LR | Mean Reward | Success Rate | Key Observation |
|--------|--------|----|--------------|----------------|-----------------|
| Config 1| 128    | 0.001 | 39.2        | **18.2%**      | More aggressive, but less stable. |
| Config 2| 256    | 0.0005| **77.6**      | 12.0%          | Very stable, avoids crashes. |

**Lessons for DQN:**
- **Architecture**: A wider network (256) helps overall stability, but the standard network (128) seems sufficient to "find" the target zone.
- **Learning speed**: Since DQN is more sensitive to oscillations, we will use an LR of **0.0003** (standard for Adam/DQN) to maximize the stability learned with REINFORCE Config 2.
- **Target**: The reward stability of Config 2 shows that a "gentler" approach (low LR, large network) is preferable for this kind of continuous environment.

---

## 6. DQN (Deep Q-Network)

DQN combines Q-Learning with deep neural networks. Unlike REINFORCE, a policy gradient (on-policy) algorithm, DQN is a value-based (off-policy) algorithm that uses a **Replay Buffer** to reuse past experiences and a **Target Network** to stabilize learning.

**Why move to DQN?**
- **Efficiency**: The Replay Buffer learns much more from fewer episodes by replaying important transitions.
- **Stability**: The REINFORCE lessons (Config 2) showed that stability is the key to success on LunarLander. DQN's Target Network is specifically designed to avoid the divergence we saw with tabular Q-Learning.

### Update strategy
We compare two methods for stabilizing the target network, while keeping the lessons learned about the neural network (low LR, robust architecture).

#### 6.1. DQN Hard Update
The target network is synchronized with the main network in steps (`target_update_freq=500`).

- **Config**: `configs/dqn_hard.yaml`
- **Seeds**: [0, 1, 2, 3, 4]
- **Max steps**: 500 000

| Seed | Episodes | Mean Reward | Max Reward | Status | Time |
|------|----------|--------------|------------|--------|-------|
| 0    | 432      | 12.0         | 316.0      | Solved | ~12min|
| 1    | 450      | -19.2        | 292.4      | Solved | ~12min|
| 2    | 441      | 26.8         | 290.6      | Solved | ~12min|
| 3    | 534      | 37.0         | 309.6      | Solved | ~15min|
| 4    | 453      | 18.9         | 315.9      | Solved | ~12min|

### 6.2. DQN Soft Update (Polyak)
The target network is updated gradually at each step using a factor $\tau$ (`tau=0.02`).

- **Config**: `configs/dqn_soft.yaml`
- **Seeds**: [0, 1, 2, 3, 4]
- **Max steps**: 500 000

| Seed | Episodes | Mean Reward | Max Reward | Status | Time |
|------|----------|--------------|------------|--------|-------|
| 0    | 393      | 14.0         | 304.9      | Solved | ~12min|
| 1    | 405      | 29.8         | 291.7      | Solved | ~12min|
| 2    | 470      | 31.2         | 296.8      | Solved | ~15min|
| 3    | 504      | 12.6         | 292.0      | Solved | ~15min|
| 4    | 560      | 63.0         | 298.1      | Solved | ~15min|

### DQN Hard Evaluation (100 episodes per seed)
- **Mean Reward**: **218.5 ± 5.4**
- **Success rate**: **75.6%**
- **End cause**: Controlled landings in the target zone.
- **Max Reward**: 316.0

**DQN Hard observation:**
DQN Hard Update is a breakthrough in this benchmark. Using a Replay Buffer and a Target Network, it **solves** the environment on every seed where REINFORCE plateaued. The agent acquired a piloting precision that the on-policy methods could not stabilize within the allotted time.

### DQN Soft Evaluation (100 episodes per seed)
- **Mean Reward**: **210.0 ± 5.4**
- **Success rate**: **70.6%**
- **End cause**: Controlled landings, but sometimes too cautious (timeouts).
- **Max Reward**: 304.9

### DQN Conclusion
The DQN family proved clearly superior on this benchmark. 
- **DQN Hard Update** performs best (**+218.5 reward**, **75.6% success**), showing that a periodic, stable target network update is enough to fully master piloting the Lander.
- **DQN Soft Update** is just as capable but converged with slightly more difficulty on some seeds and showed more hesitant flight behavior.

The **Replay Buffer** was the transformative element, letting DQN learn about 2 times faster than REINFORCE with 4 times higher final precision.

---

## 7. Final Benchmark Summary

| Agent | Type | Action Space | Mean Reward | Success Rate | Verdict |
|-------|------|-----------------|--------------|----------------|---------|
| Random| Baseline | Discrete | -184.6 | 0.0% | Ineffective |
| Heuristic| Baseline (PD controller) | Discrete | +255.4 | 95.0% | Best score, no learning |
| Q-Learning| Tabular | Discretized | -41.8 | 18.2% | Limited |
| REINFORCE (C2)| PG | Discrete | +77.6 | 12.0% | Stable but imprecise |
| **DQN Hard** | **Deep Q** | **Discrete** | **+218.5** | **75.6%** | **Best learned agent** |
| DQN Soft | Deep Q | Discrete | +210.0 | 70.6% | Solid but slow |

---

## 8. Ablation Study: Target Network Update Strategies

A key part of our benchmark compared two target network update methods for stabilizing DQN.

| Parameter | Hard Target Update | Soft Target Update (Polyak) |
|-----------|--------------------|-----------------------------|
| **Mechanism** | Full copy every 500 steps | Continuous blending ($\tau=0.02$) |
| **Mean Reward** | **+218.5** | +210.0 |
| **Success Rate**| **75.6%** | 70.6% |
| **Convergence** | Fast (~450 episodes) | Slower (~500+ episodes) |
| **Behavior** | Decisive, clean landings | Cautious, tends to hover |

### Results analysis
The ablation shows that for LunarLander, the **Hard Update** is superior. The step-wise update seems to act as a beneficial "jolt" for the policy, keeping the agent from locking into excessive caution (the many timeouts observed with Soft Update). The Soft Update, although theoretically more stable, slowed convergence toward an optimal landing solution here.

**Overall Conclusion**: The project demonstrates how essential neural network function approximation is for continuous-state environments. Moving from tabular Q-Learning to DQN turned an agent unable to land into a pilot that solves the task, although a hand-tuned controller using knowledge of the physics still scores higher (255.4).
