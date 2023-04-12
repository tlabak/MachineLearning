from gym.spaces.discrete import Discrete
import numpy as np

class DummyEnv:
    def __init__(self, rewards):
        self.rewards = rewards
        self.action_space = Discrete(len(rewards))
        self.observation_space = Discrete(1)
    
    def reset(self):
        return 0

    def step(self, action):
        return 0, self.rewards[action], True, {}
