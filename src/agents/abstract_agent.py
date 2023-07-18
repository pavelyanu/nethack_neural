import abc

class AbstractAgent(abc.ABC):
    @abc.abstractmethod
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def learning_step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass