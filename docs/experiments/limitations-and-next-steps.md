# Limitations and next steps

[Docs index](../README.md) · see also: [methodology](methodology.md), [results](results.md)

What is known to be weak in the current code and experiments, and what would be tried
next, with the reason and the expected cost.

## Known weaknesses

| Weakness | Effect | Fix | Cost |
|---|---|---|---|
| Q-Learning decays ε **per step** ([chapter 3](../theory/03-q-learning.md#why-it-plateaus)) | exploration ends after about 10 episodes, the table stops improving | decay per episode, as DQN does | one line, then retrain 5 seeds |
| Q-Learning bins round toward zero | the bin around 0 is twice as wide on every axis | `np.floor` instead of `astype(int)`, finer bins on angle and angular velocity | small; table grows |
| Termination cause inferred from the return sign ([methodology](methodology.md#caveats)) | `landed` / `crash` counts are approximate | record the last step's reward (+100 rest, -100 crash) in the logger | small; needs a full re-evaluation, including the Q-Learning and REINFORCE models that are not versioned |
| Unequal training budgets | DQN had up to 500k steps, the others 200k | rerun everything at the same budget | hours of CPU |
| 5 seeds, pooled confidence interval | differences of a few points are not significant | 10+ seeds, report the interquartile mean with bootstrap intervals over runs (Agarwal et al., 2021) | twice the training time |
| REINFORCE evaluated by sampling | its score includes exploration noise | report both sampled and greedy evaluations | small |

## Algorithmic next steps

Ordered by expected gain for this project:

1. **Actor-critic for the policy-gradient family.** REINFORCE's main problem is variance.
   A learned value baseline (A2C) or PPO (Schulman et al., 2017) reuses each batch of data
   for several clipped updates and is the usual way to solve LunarLander with a policy
   method. Worth it when a stochastic policy or continuous actions are needed.
2. **Prioritised experience replay** (Schaul et al., 2016). DQN samples old transitions
   uniformly, including many uneventful hovering frames. Sampling by TD error focuses
   updates on surprising transitions; it typically reduces the number of episodes needed,
   at the cost of a sum-tree and importance-sampling weights.
3. **Dueling architecture** (Wang et al., 2016). Splits the network into a state value and
   per-action advantages; helps when many actions have similar values, as in the hovering
   phase. A few lines in `QNetwork`.
4. **Rainbow-style combination** (Hessel et al., 2018): n-step returns, distributional
   values, noisy networks. Mostly relevant beyond this task's difficulty.
5. **Continuous control.** `LunarLanderContinuous-v3` gives throttle control; SAC
   (Haarnoja et al., 2018) would allow smoother, cheaper landings than on/off engines.

## What the rule-based controller suggests

The controller lands faster (about 250 frames against 370 to 530) and more often (95%
against 76%). Two cheap experiments follow from it:

- **Learning from demonstrations:** pre-fill DQN's replay buffer with controller flights,
  or pre-train the network on its actions, then fine-tune with RL. This usually cuts the
  exploration phase sharply.
- **Residual learning:** let the network learn a correction on top of the controller
  instead of a policy from scratch.

Both would no longer answer the question this project asks (learning from nothing), so
they belong in a separate experiment.
