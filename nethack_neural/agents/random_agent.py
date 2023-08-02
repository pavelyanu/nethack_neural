import numpy as np
from nethack_neural.agents.abstract_agent import AbstractAgent
from nethack_neural.utils.env_specs import EnvSpecs

class RandomAgent(AbstractAgent):
    def __init__(self, env_specs):
        super().__init__(env_specs)

    def act(self, state):
        return np.random.randint(0, self._num_actions)

    def learning_step(self, state, action, reward, next_state, done):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass