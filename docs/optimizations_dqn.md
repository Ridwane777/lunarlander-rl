# Deep Q-Network (DQN) Optimizations on LunarLander-v3

## 1. Update frequency (`update_every`)
**Initial approach**: The network runs a backpropagation pass at **every iteration** of the environment (every `env.step()` call).
**Optimized approach**: The network is only updated **every 4 steps**.

**Why it is better:**
- It is a standard technique from DeepMind's original paper.
- New experiences accumulate in the Replay Buffer between updates, which breaks the temporal correlation of samples.
- It keeps the agent from overfitting on the very recent dynamics of its flight.
- It cuts training compute cost by 4.

## 2. Exploration rate decay (`epsilon_decay`)
**Initial approach**: Exploration decays by multiplying by `0.995` at **every step**. As a result, exploration (`epsilon`) drops from 1.0 to 0.01 in under 1000 iterations (1 or 2 episodes), leaving the agent no time to discover the environment.
**Optimized approach**: A decay of `0.996` **per episode** (or `0.9995` per step) lets the agent explore actively for about fifty episodes before settling its strategy. 

## 3. Gradient Clipping
**Initial approach**: Clipping set to `10.0`.
**Optimized approach**: Aggressive clipping set to `1.0`.

**Why it is better:** 
On LunarLander, an unexpected crash or a prolonged thruster burn can produce huge Temporal Difference (TD) errors in the *Loss*. Clipping at `1.0` ensures the network weights do not make abrupt jumps that would "break" earlier learning.

## 4. Target Network update strategy
The optimized agent uses a **Hard Update** (full weight copy) every 1000 learning cycles, giving a perfectly fixed target for long stretches.
Our model uses a **Soft Update** (Polyak method, `tau=0.005`) at every update. Although the *Soft Update* is more modern (used in continuous algorithms such as SAC), the *Hard Update* often proves slightly more stable for a standard vanilla DQN.

---

## Conclusion
The two immediate optimizations with the most impact for solving LunarLander-v3 faster are:
1. Adding `update_every = 4` to ease the weight-update load.
2. Lowering `grad_clip` to `1.0` for more stability against severe negative rewards (-100 on a crash).
