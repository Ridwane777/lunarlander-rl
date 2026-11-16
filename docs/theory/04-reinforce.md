# 4. REINFORCE (policy gradient)

[Docs index](../README.md) · previous: [Q-Learning](03-q-learning.md) · next: [DQN](05-dqn.md)

**Papers:** Williams (1992); policy gradient theorem: Sutton, McAllester, Singh and Mansour (2000).
**Code:** [`src/agents/reinforce_agent.py`](../../src/agents/reinforce_agent.py),
[`src/trains/train_reinforce.py`](../../src/trains/train_reinforce.py).
**Configs:** [`configs/reinforce.yaml`](../../configs/reinforce.yaml),
[`configs/reinforce_v2.yaml`](../../configs/reinforce_v2.yaml).

## Idea

Instead of learning how good each action is, learn **the pilot itself**: a neural network
$\pi_\theta(a \mid s)$ that outputs a probability for each of the 4 actions. Fly a full
episode by sampling from it, then make the actions that preceded high returns more likely
and the others less likely.

## The policy gradient

The objective is the expected return $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$. The policy
gradient theorem gives its gradient without any model of the physics:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t G_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

Read it as: for every step, push $\log \pi_\theta(a_t \mid s_t)$ up in proportion to the
return $G_t$ that followed. REINFORCE estimates this expectation with a single episode
(Monte Carlo) and takes one gradient step. In code, the loss minimised by Adam is
$-\sum_t G_t \log \pi_\theta(a_t \mid s_t)$.

## Implementation details

- **Network:** 8 inputs → 128 → 128 → 4, ReLU, softmax output (256 hidden units in `reinforce_v2`).
- **Returns:** computed backward at the end of the episode, $G_t = r_{t+1} + \gamma G_{t+1}$.
- **Normalisation:** the returns of the episode are standardised (mean 0, standard
  deviation 1) before the update. This acts as a simple baseline: actions better than the
  episode average are reinforced, the others discouraged. It reduces variance but is not
  the learned baseline (critic) of actor-critic methods.
- **One update per episode**, data discarded afterward (on-policy).
- **Evaluation samples from the policy** rather than taking the most likely action, so
  REINFORCE results include its own randomness.
- Configurations: `reinforce` (lr 0.001, 128 units) and `reinforce_v2` (lr 0.0005, 256
  units); γ = 0.99, budget 200 000 steps, 5 seeds.

## Results

| Config | Eval mean (100 episodes × 5 seeds) | Success (≥ 200) |
|---|---|---|
| `reinforce` | 39.2 ± 11.9 | 18.2% |
| `reinforce_v2` | 77.6 ± 8.1 | 12.0% |

A clear step above Q-Learning: the network generalises across continuous states. But no run
solved the task within the budget, and seeds disagree strongly: the per-seed evaluation
means of `reinforce` range from -26.9 to 127.0.

`reinforce_v2` has the higher mean but the lower success rate. Its episodes often end by
**timeout** (up to 72 of 100 for one seed): the agent learned that hovering avoids the
-100 crash, without learning to finish the landing. A higher mean with fewer real
successes is a reminder to always read the mean together with the success rate.

## Why it is slow and unstable

1. **High variance.** One episode's return depends on hundreds of random action draws.
   The gradient estimate is very noisy, so updates often point in a poor direction.
2. **Sample inefficiency.** Each episode is used for exactly one gradient step, then
   thrown away. DQN, by contrast, reuses each transition many times.
3. **Credit assignment.** A crash at the end lowers the return of every earlier step,
   including the good ones.

The standard remedies (a learned value baseline, actor-critic, PPO) are discussed in
[limitations and next steps](../experiments/limitations-and-next-steps.md).
