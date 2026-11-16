# 2. Reinforcement learning foundations

[Docs index](../README.md) · previous: [The problem](01-the-problem.md) · next: [Q-Learning](03-q-learning.md)

The concepts every algorithm of this project relies on, from the intuition to the
equations. The reference for all of it is Sutton and Barto (2018).

## The learning loop

```
        ┌──────────── action a_t ────────────┐
        │                                    ▼
   ┌─────────┐                         ┌─────────────┐
   │  agent  │                         │ environment │
   └─────────┘                         └─────────────┘
        ▲                                    │
        └──── state s_t+1, reward r_t+1 ─────┘
```

At every step the agent sees a state, picks an action, and receives a reward and the next
state. Nobody tells it which action was right: it only learns which actions **tend to lead
to more reward**. That is the difference with supervised learning, where every example comes
with the correct answer.

## Markov decision process (MDP)

The formal model is a tuple $(S, A, P, R, \gamma)$:

| Symbol | Meaning | In LunarLander |
|---|---|---|
| $S$ | set of states | the 8-number observation |
| $A$ | set of actions | 4 engine commands |
| $P(s' \mid s, a)$ | transition dynamics | Box2D physics, unknown to the agent |
| $R(s, a, s')$ | reward | shaping + fuel cost + landing / crash bonus |
| $\gamma \in [0, 1)$ | discount factor | 0.99 in every configuration |

*Markov* means the next state depends only on the current state and action, not on the
past. LunarLander's observation includes positions **and** velocities, so it is (nearly)
Markov: knowing where the lander is and how fast it moves is enough to predict what
happens next.

## Return and discount

The agent maximises the **return**, the discounted sum of future rewards:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

With $\gamma = 0.99$, a reward 100 steps ahead still weighs $0.99^{100} \approx 0.37$: the
agent cares about the landing bonus that arrives seconds later, not only about the next frame.

## Policy and value functions

- A **policy** $\pi(a \mid s)$ is the pilot: the probability of each action in each state.
  It can be deterministic (always the same action) or stochastic.
- The **state value** $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ is the return expected
  from state $s$ when following $\pi$.
- The **action value** $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$ is the return
  expected after taking $a$ in $s$, then following $\pi$.

If we knew the optimal action values $Q^*(s, a)$, flying would be trivial: always pick
$\arg\max_a Q^*(s, a)$. Value-based methods (Q-Learning, DQN) try to learn $Q^*$;
policy-based methods (REINFORCE) learn $\pi$ directly.

## The Bellman equation

Values satisfy a recursive relation (Bellman, 1957): the value of a situation is the
immediate reward plus the discounted value of where you end up next. For the optimal action
values:

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \,\middle|\, s, a\right]$$

Q-Learning and DQN turn this equation into an update rule: compare the current estimate
$Q(s, a)$ with the **target** $r + \gamma \max_{a'} Q(s', a')$ and move it toward the target.
The difference is the **temporal-difference (TD) error**:

$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

## Monte Carlo versus temporal difference

Two ways to estimate how good an action was:

| | Monte Carlo (REINFORCE) | Temporal difference (Q-Learning, DQN) |
|---|---|---|
| Uses | the real return $G_t$, known at the end of the episode | one reward + the current estimate of the next state |
| Updates | once per episode | at every step |
| Bias | none: $G_t$ is a true sample | biased by the estimate it bootstraps on |
| Variance | high: one return depends on hundreds of random events | low |

This trade-off explains a large part of the results: REINFORCE's noisy returns make it
learn slowly and unevenly, while the TD methods learn step by step.

## Exploration versus exploitation

To find good actions the agent must try actions it is unsure about (**explore**), but to
score it must use what it already knows (**exploit**). The value-based agents of this
project use **ε-greedy** exploration: with probability ε pick a random action, otherwise the
best known one. ε starts at 1 (pure exploration) and decays toward 0.01. REINFORCE explores
naturally, by sampling from its stochastic policy.

## On-policy versus off-policy

- **On-policy** (REINFORCE): learns only from data produced by the current policy. After
  each update, old episodes are thrown away.
- **Off-policy** (Q-Learning, DQN): learns about the greedy policy while behaving
  ε-greedily, so it can reuse any past experience. This is what makes DQN's replay buffer
  possible, and it is a major source of its sample efficiency.

## Function approximation

With continuous states, $Q$ or $\pi$ cannot be stored in a table. Two answers appear in
this project:

1. **Discretise** the state into bins and keep a table (Q-Learning). Simple, but similar
   states in different bins share nothing, and the number of bins explodes with the number
   of dimensions (the *curse of dimensionality*).
2. **Approximate** the function with a neural network (REINFORCE, DQN). The network
   generalises: what it learns in one state transfers to nearby states. The price is
   instability, which DQN addresses with the techniques of [chapter 5](05-dqn.md).

## The three learning agents at a glance

| | Q-Learning | REINFORCE | DQN |
|---|---|---|---|
| Learns | $Q$ table | policy $\pi_\theta$ | $Q_\theta$ network |
| State representation | discretised bins | neural network | neural network |
| Estimate | TD | Monte Carlo | TD |
| Data | each step, then discarded | one episode, then discarded | replay buffer of 100 000 steps |
| On / off-policy | off | on | off |
| Chapter | [3](03-q-learning.md) | [4](04-reinforce.md) | [5](05-dqn.md) |
