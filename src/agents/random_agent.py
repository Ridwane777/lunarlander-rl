import gymnasium as gym

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, obs):
        return self.env.action_space.sample()