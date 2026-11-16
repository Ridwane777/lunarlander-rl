# 3. Tabular Q-Learning

[Docs index](../README.md) · previous: [RL foundations](02-rl-foundations.md) · next: [REINFORCE](04-reinforce.md)

**Paper:** Watkins (1989), convergence proof in Watkins and Dayan (1992).
**Code:** [`src/agents/q_learning.py`](../../src/agents/q_learning.py),
[`src/trains/train_q_learning.py`](../../src/trains/train_q_learning.py).
**Configs:** [`configs/q_learning.yaml`](../../configs/q_learning.yaml),
[`configs/q_learning_v2.yaml`](../../configs/q_learning_v2.yaml).

## Idea

Keep a table with one row per state and one column per action, holding the estimate
$Q(s, a)$. After every step, nudge the entry of the action just taken toward the Bellman
target:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

$\alpha$ is the learning rate. The $\max$ makes the algorithm **off-policy**: it learns the
value of the greedy policy even while acting ε-greedily. Watkins and Dayan proved that the
table converges to $Q^*$ if every state-action pair is visited infinitely often and $\alpha$
decays appropriately. Neither condition holds here, which is the whole story of this chapter.

## From continuous states to table rows

The table needs a finite number of states, so each of the 8 observations is divided by a
bin width and truncated to an integer:

| Observation | `x` | `y` | `vx` | `vy` | `angle` | `ang. vel.` | legs |
|---|---|---|---|---|---|---|---|
| Bin width | 0.5 | 0.5 | 0.2 | 0.2 | 0.1 | 0.1 | 1 |

`(obs / bins).astype(int)` gives the row key. The table is a dictionary filled on demand,
so only visited bins use memory (a trained table weighs up to 8 MB).

## Implementation details

- ε-greedy action selection; unseen states start at $Q = 0$ for every action.
- Terminal transitions use $r$ alone as the target (no bootstrap past the end).
- Two configurations: `q_learning` ($\alpha = 0.1$, ε decay 0.995) and `q_learning_v2`
  ($\alpha = 0.05$, ε decay 0.999). γ = 0.99, budget 200 000 environment steps, 5 seeds.

## Results

| Config | Eval mean (100 episodes × 5 seeds) | Success (≥ 200) |
|---|---|---|
| `q_learning` | -41.8 ± 16.1 | 18.2% |
| `q_learning_v2` | -92.7 ± 14.4 | 11.6% |

Clearly better than random actions (-193.9), far from solving the task. No run reached the
100-episode mean of 200 within the budget; the best last-100 training mean was -8.0.

## Why it plateaus

1. **Coarse, uneven bins.** One bin of `angle` covers 0.1 rad (6°), and truncation toward
   zero makes the bin around 0 twice as wide as the others. Very different situations,
   such as "slightly tilted, rotating fast" and "upright, still", can share a row and
   therefore share the same action.
2. **No generalisation.** What is learned in one bin says nothing about its neighbour.
   With 8 dimensions the agent keeps meeting bins it has never seen, where all actions
   are worth 0 and the choice is arbitrary.
3. **Exploration collapses almost immediately.** ε is multiplied by the decay factor at
   **every step**, not every episode. With 0.995 it falls from 1 to its floor of 0.01 after
   about 920 steps, roughly the first ten episodes; with 0.999 after about 4 600 steps.
   For the remaining 195 000 steps the agent almost never tries anything new. This is a
   known weakness of the implementation, kept as is so that the published numbers match
   the code; decaying per episode is the first fix to try (see
   [limitations and next steps](../experiments/limitations-and-next-steps.md)).
4. **The second configuration did worse**, not better: a smaller learning rate slows the
   propagation of the landing bonus through the table, and longer exploration (still very
   short, see point 3) does not compensate.

Tabular Q-Learning is the right first step to understand the Bellman update, and its
failure motivates the two neural network methods that follow.
