# 5. Deep Q-Network (Double DQN)

[Docs index](../README.md) · previous: [REINFORCE](04-reinforce.md) · next: [Rule-based control](06-rule-based-control.md)

**Papers:** Mnih et al. (2013, 2015) for DQN; Lin (1992) for experience replay;
van Hasselt (2010) and van Hasselt, Guez and Silver (2016) for Double DQN; Lillicrap et al.
(2016) for soft target updates.
**Code:** [`src/agents/dqn_agent.py`](../../src/agents/dqn_agent.py),
[`src/utils/replay_buffer.py`](../../src/utils/replay_buffer.py),
[`src/trains/train_dqn.py`](../../src/trains/train_dqn.py).
**Configs:** [`configs/dqn_hard.yaml`](../../configs/dqn_hard.yaml),
[`configs/dqn_soft.yaml`](../../configs/dqn_soft.yaml).

## Idea

Keep the Q-Learning update, but replace the table with a neural network $Q_\theta(s, a)$
that takes the 8 continuous observations and outputs one value per action. Similar states
now share what they learn. The difficulty is that naively training a network on the
Q-Learning target diverges; DQN is the set of techniques that makes it stable.

## The loss

For a transition $(s, a, r, s', \text{done})$ the network is trained to reduce

$$L(\theta) = \left( y - Q_\theta(s, a) \right)^2, \qquad y = r + \gamma (1 - \text{done}) \, Q_{\theta^-}\!\left(s', \arg\max_{a'} Q_\theta(s', a')\right)$$

averaged over a batch. $\theta^-$ are the weights of a separate **target network**. Each
ingredient below fixes one cause of instability.

## Ingredient 1: experience replay

Consecutive frames are almost identical, and training a network on strongly correlated
samples makes it overfit to the last few seconds of flight. Every transition is stored in a
**replay buffer** (100 000 transitions, oldest dropped first) and each update uses a batch
of 128 transitions **sampled uniformly at random** from it (Lin, 1992; Mnih et al., 2013).
This breaks the correlation and lets every experience be reused many times, which is the
main reason DQN needs far fewer episodes than REINFORCE.

## Ingredient 2: target network

The target $y$ depends on the network being trained: every update moves the target it is
chasing. A second network with frozen weights $\theta^-$ computes the targets (Mnih et al.,
2015). Two ways to refresh it are compared, which is the ablation of this project:

| Variant | Rule | Config |
|---|---|---|
| **Hard update** | copy $\theta^- \leftarrow \theta$ every 500 gradient updates | `dqn_hard` |
| **Soft update** (Polyak averaging) | $\theta^- \leftarrow \tau \theta + (1 - \tau)\theta^-$ after every update, $\tau = 0.02$ | `dqn_soft` |

## Ingredient 3: Double DQN

Taking $\max_{a'} Q(s', a')$ over noisy estimates systematically picks the luckiest
overestimate, and the bias compounds through bootstrapping. Double DQN **selects** the next
action with the online network and **evaluates** it with the target network, as in the
formula above (van Hasselt et al., 2016). Selection noise and evaluation noise are then
independent, which removes most of the upward bias.

## Other details

| Parameter | Value | Why |
|---|---|---|
| Network | 8 → 128 → 128 → 4, ReLU | enough capacity for 8 inputs, fast on a CPU |
| Optimiser | Adam (Kingma and Ba, 2015), lr 0.0003 | standard for DQN on small control tasks |
| Update frequency | one gradient step every 4 environment steps | as in Mnih et al. (2015); cheaper, and the buffer changes between updates |
| Gradient clipping | global norm ≤ 1.0 | a crash produces a large TD error; clipping keeps one bad batch from wrecking the weights (Pascanu et al., 2013) |
| Exploration | ε from 1.0 to 0.01, × 0.995 **per episode** | about 450 episodes to reach 0.1 |
| Discount | γ = 0.99 | |
| Stop rule | mean of the last 100 training episodes ≥ 200, or 500 000 steps | |

Learning starts as soon as the buffer holds one batch (128 transitions).

## Results

| Variant | Eval mean (100 episodes × 5 seeds) | Success (≥ 200) | Per-seed eval means |
|---|---|---|---|
| `dqn_hard` | **218.5 ± 5.4** | **75.6%** | 202.0, 212.2, 203.7, 225.2, 249.2 |
| `dqn_soft` | 210.0 ± 5.4 | 70.6% | 238.4, 234.0, 186.0, 189.8, 201.9 |

Every run of both variants **solved the task during training**, after 393 to 560 episodes
(210 000 to 283 000 environment steps). In evaluation, 8 of the 10 trained models keep a
mean above 200; the best one (`dqn_hard_seed4`) lands 96 times out of 100 and is the model
shipped as the default of `play.py`.

**Hard versus soft.** The hard update scores slightly higher on average, but the two
confidence intervals overlap and the spread between seeds of the same variant (186 to 238
for soft) is larger than the gap between variants. With 5 seeds, the honest conclusion is
that both variants work equally well on this task. In both, the weaker seeds lose points
mainly to timeouts (hovering until the 1000-frame limit, up to 16 episodes out of 100),
rarely to crashes (at most 5 out of 100).

## Why it works where the others plateau

1. The network **generalises** across continuous states (unlike the Q table).
2. **TD updates** give a low-variance signal at every step (unlike REINFORCE's returns).
3. The **replay buffer** reuses each transition many times (unlike both).
4. Target network, Double DQN and gradient clipping keep this combination **stable**.
