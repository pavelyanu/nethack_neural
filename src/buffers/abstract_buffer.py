from abc import ABC, abstractmethod
from numpy import ndarray


class AbstractBuffer(ABC):
    def __init__(self, buffer_size, n_agents=1):
        self.buffer_size = buffer_size
        self.n_agents = n_agents

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass