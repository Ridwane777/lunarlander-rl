"""Hand-written controller: a reference point that uses no learning.

Adapted from the heuristic shipped with Gymnasium's LunarLander environment
(gymnasium/envs/box2d/lunar_lander.py, MIT license). Two proportional-derivative
targets drive the discrete engines:

- angle: lean toward the pad, more when far from it or drifting sideways;
- hover: the further from the pad horizontally, the higher to hover.

The previous version only reacted to large tilts and fast falls; it scored worse
than random actions (-220 vs -194), which made it a meaningless baseline.
"""

import numpy as np

NOOP, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT = 0, 1, 2, 3


class HeuristicAgent:
    def __init__(self, env=None):  # optional env, unused
        pass

    def select_action(self, obs):
        x, y, vx, vy, angle, angular_velocity, left_contact, right_contact = obs

        angle_target = float(np.clip(x * 0.5 + vx * 1.0, -0.4, 0.4))
        hover_target = 0.55 * abs(x)

        angle_todo = (angle_target - angle) * 0.5 - angular_velocity * 1.0
        hover_todo = (hover_target - y) * 0.5 - vy * 0.5

        if left_contact or right_contact:
            # Touching down: stop rotating, only soften the vertical speed.
            angle_todo = 0.0
            hover_todo = -vy * 0.5

        if hover_todo > abs(angle_todo) and hover_todo > 0.05:
            return FIRE_MAIN
        if angle_todo < -0.05:
            return FIRE_RIGHT
        if angle_todo > 0.05:
            return FIRE_LEFT
        return NOOP
