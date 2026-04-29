# Reinforcement Learning

## < LunarLander RL />

---

# REINFORCEMENT LEARNING

- **binary name:** *.py  
- **language:** Python  
- **compilation:** None  
- **authorized libraries:** gymnasium, pytorch  
- **additional documents needed:** Anything used as a support during your defense  

---

## Start the Trek through the stars

You are part of one of the first teams exploring our solar system.  
Destination: Mars.  

However, to carry humans safely across space, we must first set up a relay base with resources on the Moon.  

While the long-term objective is to be able to launch manned missions, you can’t afford to put a pilot in every lunar module just yet.  
Thus, you will have to train the module to land by itself.  

Your role is to train a module in a simulator to do just that.  

You will program an algorithm that enables the module to correct its own trajectory. It will be a reinforcement-learning agent that learns by trial and error from sparse rewards (soft penalties per step, strong bonus on safe landing).  

You will emphasize:
- the RL loop (state → action → reward)
- exploration–exploitation trade-offs
- basic value/Q estimation before any deep networks  

Success is measured by:
- consistent autonomous landings  
- improved sample-efficiency across iterations  

You will build disciplined engineering habits:
- metrics  
- reproducibility  
- readable code  

Then—before sending a multi-million-dollar prototype into space—you will:
- deliver a detailed report  
- provide evidence of your tests  
- explain your modifications  
- defend your solution  

**Shoot for the moon, and hopefully you’ll land on the moon**

---

# Lunar Lander

## Gymnasium

In this project, we will use Gymnasium, an API standard for reinforcement learning with a diverse collection of reference environments.  

The environment used is **Lunar Lander**:  
- a 2D rocket landing task  
- based on Box2D physics  

It handles:
- physics simulation  
- module representation  
- environment mechanics  

You focus on reinforcement learning.  
Code must be written in Python.

---

## Parameters

### Actions (Discrete)
- do nothing  
- fire left thruster  
- fire main engine  
- fire right thruster  

A continuous-control version also exists.

---

## State

The state is an 8-dimensional vector:

- `x`: horizontal position  
- `y`: vertical position  
- `v_x`: horizontal velocity  
- `v_y`: vertical velocity  
- `theta`: lander angle  
- `theta_dot`: angular velocity  
- `left_leg_contact`: 1 if left leg touches ground, else 0  
- `right_leg_contact`: 1 if right leg touches ground, else 0  

---

## Reward

The reward function encourages:
- being close to the landing pad  
- moving slowly  
- staying level  
- landing with both legs  

Additional rewards:
- per-step penalties when engines fire  
- about **+100** for safe landing  
- about **-100** for crash  

The environment is solved when:
- mean score ≥ 200 (without wind)

---

## Usage (Gymnasium)

```python
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="rgb_array")
obs, info = env.reset(seed=SEED)

while True:
    action = policy(obs)
    obs, r, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
```

For continuous control or wind:

```python
gym.make(
    "LunarLander-v3",
    continuous=True,
    enable_wind=True,
    wind_power=15.0,
    turbulence_power=1.5
)
```

---

# Goals

## Deliverables

- Codebase (clean repo: `train.py`, `eval.py`, README)  
- YAML configs for each run (env, algo, hyperparameters, seeds)  
- Tracking & assets:
  - returns plots  
  - episode length plots  
  - termination stats  
  - videos  
- Short report (4–6 pages):
  - setup  
  - results  
  - ≥1 ablation  
  - limits  
  - next steps  
- One-click reproduction script (figures + CSV)  

---

## Tasks

### 1. Baseline & instrumentation
- Random policy  
- Simple heuristic  
- Log:
  - returns  
  - episode length  
  - termination reason (crash / out-of-view / sleep)  
- Save videos  

---

### 2. Agent (discrete)

- MLP (2–3 layers with ReLU)  
- Replay buffer  
- Target network  
- Epsilon-greedy decay  

Track:
- returns  
- epsilon  
- loss  

---

### 3. Ablation (≥1)

Examples:
- buffer size  
- hard vs soft target updates  
- observation normalization  
- reward clipping  

---

### 4. Reproducibility

- ≥5 seeds (0..4)  
- Report mean ± 95% confidence interval  

---

### 5. Defense

You must:
- justify your choices  
- benchmark techniques  
- explain your reasoning  

---

## Acceptance Criteria

- Mean score ≥ 200 over 100 consecutive episodes  
- Report distribution & variance  
- Properly handle:
  - `terminated` vs `truncated`  
  - termination reason (crash, out-of-view, sleep)  
- Reproducibility:
  - regenerate plots/CSV within ±5%  

---

**Version:** v0.0.1
