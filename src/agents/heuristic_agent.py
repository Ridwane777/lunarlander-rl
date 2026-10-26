class HeuristicAgent:
    def __init__(self, env=None):  # optional env, unused
        pass

    def select_action(self, obs):
        x, y, vx, vy, theta, theta_dot, leg_L, leg_R = obs

        # Angle correction first
        if theta > 0.2:
            return 1  # fire left thruster
        if theta < -0.2:
            return 3  # fire right thruster

        # Vertical braking only when falling fast AND low
        if vy < -1.0 and y < 0.5:
            return 2  # fire main engine

        return 0  # do nothing