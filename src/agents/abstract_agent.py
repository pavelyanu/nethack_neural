import abc

from src.utils.env_specs import EnvSpecs

class AbstractAgent(abc.ABC):
    @abc.abstractmethod
    def __init__(self, env_specs: EnvSpecs) -> None:
        self._observation_space = env_specs.observation_shape
        self._num_actions = env_specs.num_actions
        self._num_envs = env_specs.num_envs

    @abc.abstractmethod
    def act(self, state, train=True):
        pass

    @abc.abstractmethod
    def save_transition(self, *args):
        pass

    @abc.abstractmethod
    def train():
        pass

    @abc.abstractmethod
    def preprocess(self, state):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass
