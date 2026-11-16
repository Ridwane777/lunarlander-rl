# 6. Rule-based control: flying without training

[Docs index](../README.md) · previous: [DQN](05-dqn.md)

**Reference:** the heuristic shipped with Gymnasium's LunarLander (MIT license), from which
this controller is adapted; PD control: Åström and Murray (2008), chapter 10.
**Code:** [`src/baselines/heuristic.py`](../../src/baselines/heuristic.py) (about 30 lines).

```bash
python play.py --agent heuristic            # watch it fly, no weights, no training
```

## Why it is in the project

The learning agents start from zero knowledge. This controller is the opposite: an engineer
who understands the physics writes the rules directly. It answers two questions:

- **How good can a pilot be on this task?** It gives a reference score that does not depend
  on any training run.
- **What does learning buy?** DQN gets close to it without being told anything about
  gravity, thrust or angles. That is the point of the project, not beating the controller.

It is a usable pilot in its own right: deterministic, instant to run, no data needed.

## Proportional-derivative (PD) control in one minute

A PD controller compares where the system **is** with where it **should be**, and acts on:

- the **error** (proportional term, P): the further from the target, the stronger the push;
- the **rate of change** of the error (derivative term, D): if already moving fast toward
  the target, ease off to avoid overshooting.

$$u = K_p \, (\text{target} - \text{value}) - K_d \, \text{velocity}$$

Without the D term the lander would swing past the target and oscillate; the D term is a
damper.

## The controller

Two targets are recomputed at every frame from the 8 observations:

```python
angle_target = clip(0.5 * x + 1.0 * vx, -0.4, 0.4)   # lean toward the pad, more if far or drifting
hover_target = 0.55 * abs(x)                        # stay higher while far from the pad

angle_todo = 0.5 * (angle_target - angle) - 1.0 * angular_velocity   # PD on the tilt
hover_todo = 0.5 * (hover_target - y)     - 0.5 * vy                  # PD on the height
```

Then a single engine is chosen:

1. if a leg touches the ground: stop rotating (`angle_todo = 0`) and only soften the
   vertical speed (`hover_todo = -0.5 * vy`);
2. if the height correction dominates and is significant (> 0.05): **main engine**;
3. else, if the tilt correction is significant: **right engine** to rotate one way
   (`angle_todo < -0.05`), **left engine** the other way (`> 0.05`);
4. otherwise do nothing.

Leaning toward the pad is how a lander with no lateral thruster moves sideways: tilting
the main engine turns part of its thrust into horizontal force. Hovering higher while far
away leaves time to cross over before descending.

## Results

| Pilot | Eval mean (same 100 episodes) | Success (≥ 200) | Mean episode length |
|---|---|---|---|
| Rule-based controller | **255.4 ± 14.6** | **95%** | 249 frames |
| Best learned agent (`dqn_hard`, 5 seeds) | 218.5 ± 5.4 | 75.6% | 371 to 533 frames |

Measured on 2026-11-16 (Python 3.14, Gymnasium 1.4.0). The controller is deterministic, so
one run of 100 episodes covers it; the interval reflects the spread between episodes.
It also lands faster: its episodes are about half as long, which saves fuel cost.

## A first version that failed

The first hand-written controller scored **-219.8**, worse than random actions:

```python
if angle > 0.2:  return FIRE_LEFT     # wrong side: increases the tilt
if angle < -0.2: return FIRE_RIGHT
if vy < -1.0 and y < 0.5: return FIRE_MAIN
return NOOP
```

1. **Inverted correction.** In Gymnasium, a positive angle is corrected with the right
   engine; this code fired the left one, so every tilt grew until the lander tipped over.
2. **No damping.** It reacted only beyond 0.2 rad (11°) and ignored the rotation speed.
3. **Braking too late.** The main engine fired only when already low *and* falling fast,
   with no room left to stop.
4. **No horizontal guidance.** `x` and `vx` were ignored, so the lander never aimed for the pad.

It shows the other side of hand-written control: it is only as good as the engineer's
understanding, and a single sign error is enough to make it worse than doing nothing.

## Hand-written or learned?

| | Rule-based controller | Learned agent (DQN) |
|---|---|---|
| Needs | an understanding of the dynamics | a simulator and compute (about 250 000 steps) |
| Performance here | best (255) | good (218, best seed 249) |
| Explainable | fully: 6 lines of arithmetic | no: an 18 000-parameter network |
| Adapts to a new task | rewrite and retune by hand | retrain with the same code |
| Fails | when the model of the world is wrong | when training is unstable or data is short |

On a small, fully known problem like LunarLander, a good controller wins. Learning becomes
the better tool when the dynamics are unknown, too complex to model, or change from one
task to the next.
