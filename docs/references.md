# References

[Docs index](README.md)

Grouped by topic, with where each one is used in this project.

## Textbook

- **Sutton, R. S., and Barto, A. G. (2018).** *Reinforcement Learning: An Introduction*,
  2nd edition. MIT Press. Free online at http://incompleteideas.net/book/the-book.html.
  The reference for every concept of [chapter 2](theory/02-rl-foundations.md).
- **Bellman, R. (1957).** *Dynamic Programming*. Princeton University Press. Origin of the
  Bellman equation.

## Algorithms implemented here

| Reference | Used for |
|---|---|
| Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, University of Cambridge. | Q-Learning ([chapter 3](theory/03-q-learning.md)) |
| Watkins, C. J. C. H., and Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279-292. | convergence of Q-Learning |
| Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229-256. | REINFORCE ([chapter 4](theory/04-reinforce.md)) |
| Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS 12*. | policy gradient theorem |
| Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning*, 8, 293-321. | experience replay ([chapter 5](theory/05-dqn.md)) |
| Mnih, V., et al. (2013). Playing Atari with deep reinforcement learning. arXiv:1312.5602. | DQN |
| Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533. | DQN with target network |
| van Hasselt, H. (2010). Double Q-learning. *NeurIPS 23*. | overestimation bias |
| van Hasselt, H., Guez, A., and Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*. | Double DQN target |
| Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning. *ICLR*. | soft (Polyak) target update |
| Kingma, D. P., and Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*. | optimiser of every network |
| Pascanu, R., Mikolov, T., and Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*. | gradient norm clipping |

## Environment and control

| Reference | Used for |
|---|---|
| Towers, M., et al. (2024). Gymnasium: a standard interface for reinforcement learning environments. arXiv:2407.17032. | the `LunarLander-v3` environment ([chapter 1](theory/01-the-problem.md)) |
| Brockman, G., et al. (2016). OpenAI Gym. arXiv:1606.01540. | original Gym and LunarLander |
| Ng, A. Y., Harada, D., and Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. *ICML*. | reward shaping |
| Åström, K. J., and Murray, R. M. (2008). *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton University Press. | PD control ([chapter 6](theory/06-rule-based-control.md)) |
| Gymnasium's LunarLander heuristic, `gymnasium/envs/box2d/lunar_lander.py` (MIT license). | source of the rule-based controller |

## Evaluation practice

| Reference | Used for |
|---|---|
| Henderson, P., et al. (2018). Deep reinforcement learning that matters. *AAAI*. | several seeds, reporting variance ([methodology](experiments/methodology.md)) |
| Agarwal, R., et al. (2021). Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS*. | interquartile mean, intervals over runs (next step) |

## Next steps (not implemented)

| Reference | Idea |
|---|---|
| Schaul, T., et al. (2016). Prioritized experience replay. *ICLR*. | sample transitions by TD error |
| Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML*. | value / advantage split |
| Hessel, M., et al. (2018). Rainbow: combining improvements in deep reinforcement learning. *AAAI*. | combined DQN extensions |
| Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347. | stable policy gradient |
| Haarnoja, T., et al. (2018). Soft actor-critic. *ICML*. | continuous control |
