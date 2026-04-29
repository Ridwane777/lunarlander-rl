# Q-learning for LunarLander-v3

## Overview

This document describes the tabular Q-learning baseline used in this project for the `LunarLander-v3` environment.
The implementation uses a discretized observation space and a Q-table to learn action values for the discrete action set.

## Agent file

- `src/agents/q_learning.py`
- Class: `QLearningAgent`
- Environment: `LunarLander-v3`
- Action space: discrete (4 actions)

## Learning approach

The agent uses standard Q-learning:

- select actions with an epsilon-greedy policy
- update the Q-table using the Bellman equation
- decay epsilon after each update to reduce exploration over time

## State discretization

Because `LunarLander-v3` observations are continuous, the agent first discretizes the observation vector into a tuple key for the Q-table.
The observation vector contains:

1. horizontal position
2. vertical position
3. horizontal velocity
4. vertical velocity
5. lander angle
6. angular velocity
7. left leg contact flag
8. right leg contact flag

The discretization is intentionally coarse to keep the tabular state space manageable.

## Hyperparameters

The training script in `tests/test_q_learning.py` uses the following default configuration:

- `gamma`: 0.99
- `lr`: 0.1
- `epsilon_start`: 1.0
- `epsilon_end`: 0.01
- `epsilon_decay`: 0.995

## Training & logging

Training proceeds by repeatedly taking actions, observing rewards, updating the Q-table, and logging episode statistics.
Episode summaries are written to `logs/q_learning.csv` with the following columns:

- `episode`
- `total_reward`
- `length`
- `termination_cause`

The training run stops early if the mean reward over the last 100 episodes reaches 200.

## Notes

This implementation is a simple tabular baseline and is not tuned to fully solve `LunarLander-v3` in every run.
The discretization strategy and Q-table size are the main limitations in this environment.
