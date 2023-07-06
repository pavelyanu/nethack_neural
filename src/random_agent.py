import numpy as np
from src.abstract_agent import AbstractAgent

class RandomAgent(AbstractAgent):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

    def act(self, state):
        return np.random.randint(0, self.action_space.n)

    def learn(self, state, action, reward, next_state, done):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass