# Methodology

[Docs index](../README.md) · see also: [results](results.md), [limitations and next steps](limitations-and-next-steps.md)

How the numbers of this project are produced, and how far they can be trusted. Reporting
practice follows Henderson et al. (2018) and Agarwal et al. (2021): several seeds,
fixed evaluation episodes, confidence intervals, and caveats stated.

## Training

- **Seeds.** Every configuration is trained 5 times (seeds 0 to 4). The seed fixes the
  environment's initial conditions, PyTorch's weight initialisation and Python's `random`.
- **Budget.** Q-Learning and REINFORCE: 200 000 environment steps. DQN: up to 500 000.
- **Early stop.** A run stops as soon as the mean of its last 100 training episodes
  reaches 200. Only DQN runs reached it (between 210 000 and 283 000 steps).
- **Logs.** One CSV per run in `logs/train/`: episode, return, length, termination cause,
  ε and mean loss when relevant.

## Evaluation

- **Same episodes for everyone.** Every pilot flies the same 100 episodes, created with
  environment seeds 10000 to 10099. Differences between pilots therefore do not come from
  easier or harder starting conditions.
- **No exploration.** DQN and Q-Learning act greedily (ε = 0). REINFORCE samples from its
  policy, as its policy is stochastic by design.
- **Baselines.** Random actions are evaluated with 5 seeds of the action sampler; the
  rule-based controller is deterministic and evaluated once.
- **Logs and videos.** `logs/eval/<agent>.csv`, one MP4 per evaluated model in
  `videos/evals/`.

## Metrics

| Metric | Definition |
|---|---|
| Mean reward | mean return over all evaluation episodes of a method (5 seeds × 100 = 500 episodes for learned agents) |
| 95% CI | $1.96 \cdot s / \sqrt{n}$ over those episodes |
| Success rate | share of episodes with a return ≥ 200 |
| Termination cause | `timeout` if the 1000-frame limit was hit; otherwise `landed` if the episode return is positive, `crash` if negative |

## Caveats

1. **The confidence interval pools episodes, not seeds.** It measures how precisely the
   mean over these 500 episodes is known, not how much a new training run would vary.
   With 5 seeds the spread between runs is larger: DQN soft ranges from 186 to 238 per seed.
   The per-seed means are listed in [results](results.md) for that reason.
2. **Unequal budgets.** DQN was allowed 500 000 steps and needed 210 000 to 283 000; the
   other learners stopped at 200 000 without solving. The comparison at equal budget would
   cut DQN slightly short of its final level. The conclusion does not change (both
   Q-Learning and REINFORCE were far below 200 and still improving slowly), but the gap is
   not measured at strictly equal cost.
3. **Termination labels are approximate.** `landed` and `crash` are inferred from the sign
   of the episode return, not from the physics event. A lander that comes to rest outside
   the pad after a poor flight can be labelled `crash`, and a crash after a long good
   approach can be labelled `landed`. The success rate (return ≥ 200) is exact.
4. **Few seeds.** Five runs per configuration is common practice but small; differences of
   a few points between configurations (for example DQN hard versus soft) are not
   significant.
5. **Library versions.** The Gymnasium version used for training was not recorded.
   Re-evaluating two shipped models (`dqn_hard_seed4`, `dqn_soft_seed0`) with Gymnasium
   1.4.0 and PyTorch 2.15 reproduces every logged episode return to the hundredth.

## Reproducing

```bash
pip install -r requirements.txt
python eval.py                    # re-evaluate the shipped DQN models and the baselines
bash scripts/reproduce.sh         # retrain everything, then evaluate (several hours on a CPU)
pytest tests/                     # unit tests + checks that shipped models still land
```
