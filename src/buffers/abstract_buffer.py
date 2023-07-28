from abc import ABC, abstractmethod
from numpy import ndarray


class AbstractBuffer(ABC):
    def __init__(self, buffer_size, num_envs=1):
        self.buffer_size = buffer_size
        self.num_envs = num_envs

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass