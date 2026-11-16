# LunarLander RL

**Can a program learn to land on the Moon from nothing but trial and error?**

This project trains agents on Gymnasium's
[LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) with no
knowledge of physics: they only see 8 numbers (position, speed, tilt, leg contacts), choose
one of 4 engine commands, and receive a score. Three reinforcement learning algorithms of
increasing power are compared, from a lookup table to a deep Q-network, against two pilots
that need no training at all: random actions and a hand-written controller.

![Three landings by the trained DQN agent](docs/media/landing.gif)

*The trained DQN agent (`python play.py`). Nobody told it what gravity or thrust is.*

## Results

Same 100 evaluation episodes for every pilot, 5 training runs per learning method (best
configuration of each shown). An episode succeeds at 200 points; the task is solved at a
mean of 200.

| Pilot | How it gets its skill | Mean score ± 95% CI | Success |
|---|---|---|---|
| Random actions | none | -193.9 ± 10.3 | 0% |
| Q-Learning | learns a table of action values | -41.8 ± 16.1 | 18% |
| REINFORCE | learns a policy network | 77.6 ± 8.1 | 12% |
| **DQN** (Double DQN, hard target update) | learns a Q-network from replayed experience | **218.5 ± 5.4** | **76%** |
| Rule-based controller | written by hand from the physics | 255.4 ± 14.6 | 95% |

**DQN learns to land reliably from the reward alone**: every training run solved the task,
and the best model lands 96 times out of 100. The hand-written controller still scores
higher, because a human who understands the dynamics wrote it; DQN gets close without that
understanding, which is what the project sets out to show. Details, per-seed scores and
ablations: [results](docs/experiments/results.md).

## Try it

```bash
git clone https://github.com/Ridwane777/lunarlander-rl.git && cd lunarlander-rl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # Box2D needs swig: sudo apt install swig (or pip install swig)

python play.py                      # watch the trained DQN agent (weights included, no training)
python play.py --agent heuristic    # same episodes, flown by the hand-written controller
python play.py --record landing.gif # save a GIF or MP4 instead of opening a window
```

Train and evaluate:

```bash
python train.py --config configs/dqn_hard.yaml   # train one method (5 seeds)
python train.py                                  # train every learning method
python eval.py                                   # benchmark: shipped models + baselines
bash scripts/reproduce.sh                        # everything from scratch (hours on a CPU)
pytest tests/                                    # unit tests + shipped models still land
```

## How it works, in short

The agent and the environment interact in a loop: observe the state, act, receive a reward.
The agent's goal is to maximise the **return**, the discounted sum of future rewards.

- **Q-Learning** (Watkins, 1989) estimates how good each action is in each state with the
  Bellman equation, in a table. The continuous state must be cut into bins, and the table
  cannot generalise between them: it plateaus.
- **REINFORCE** (Williams, 1992) learns the policy directly with a neural network, pushing
  up the probability of actions that preceded high returns. It generalises, but its
  gradient estimate is very noisy.
- **DQN** (Mnih et al., 2015) approximates action values with a neural network and makes it
  stable with an **experience replay** buffer, a **target network** and the **Double DQN**
  target (van Hasselt et al., 2016). It reuses every experience many times and solves the task.
- The **rule-based controller** is a proportional-derivative controller adapted from the
  one shipped with Gymnasium: it leans the lander toward the pad and brakes to a target
  height. No learning; usable as a pilot on its own.

The full explanations, with equations and the reasoning behind each design choice, are in
the documentation:

| | |
|---|---|
| [1. The problem](docs/theory/01-the-problem.md) | observations, actions, reward, what makes it hard |
| [2. RL foundations](docs/theory/02-rl-foundations.md) | MDP, value functions, Bellman, TD vs Monte Carlo, exploration |
| [3. Q-Learning](docs/theory/03-q-learning.md) · [4. REINFORCE](docs/theory/04-reinforce.md) · [5. DQN](docs/theory/05-dqn.md) | each algorithm: idea, equations, implementation, results, limits |
| [6. Rule-based control](docs/theory/06-rule-based-control.md) | PD control, the controller, hand-written versus learned |
| [Methodology](docs/experiments/methodology.md) · [Results](docs/experiments/results.md) · [Next steps](docs/experiments/limitations-and-next-steps.md) | protocol, caveats, benchmark, known weaknesses |
| [References](docs/references.md) | papers and where each one is used |

## Project structure

```
.
├── play.py                watch or record a pilot (DQN or controller), no training needed
├── train.py               train the learning agents from a YAML config
├── eval.py                evaluate models and baselines on the same 100 episodes, draw the benchmark
├── configs/               hyperparameters of each learning configuration (YAML)
├── src/
│   ├── agents/            learning agents: q_learning, reinforce_agent, dqn_agent
│   ├── baselines/         pilots without training: random_agent, heuristic (rule-based controller)
│   ├── trains/            one training loop per algorithm family
│   ├── evals/             evaluation, metrics, benchmark plots
│   └── utils/             replay buffer, logger, model I/O, plots, videos
├── models/                trained DQN weights (76 KB each, 5 seeds × 2 variants)
├── logs/                  per-episode CSV logs of every training and evaluation run
├── plots/                 learning curves per configuration, benchmark charts
├── videos/                training and evaluation videos (MP4)
├── docs/                  theory, experiments, references
├── scripts/reproduce.sh   full retraining and evaluation
└── tests/                 pytest suite, run by CI on every push
```

## Author

[Ridwane777](https://github.com/Ridwane777): architecture, agents, training and evaluation
pipeline, benchmark, CI.

## License

[MIT](LICENSE). The rule-based controller is adapted from Gymnasium's LunarLander
heuristic, also under the MIT license.
