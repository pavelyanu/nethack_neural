from abc import ABC, abstractmethod
from datetime import datetime

class AbstractLogger(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def log_message(self, message: str):
        pass

    @abstractmethod
    def log_stats(self, *, timestep, time: datetime, reward, length):
        pass

    @abstractmethod
    def finalize(self):
        pass

    def format_time(self, time: datetime):
        return time.strftime('%Y-%m-%d %H:%M:%S')