from abc import ABC, abstractmethod

class AbstractLogger(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def log(self, message: str):
        pass