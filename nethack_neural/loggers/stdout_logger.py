from .abstract_logger import AbstractLogger
from datetime import datetime

class StdoutLogger(AbstractLogger):
    def __init__(self, name="stdout_logger"):
        super().__init__(name)

    def log_message(self, time: datetime, msg):
        time = self.format_time(time)
        print(f"{time}: {msg}")

    def log_stats(self, *, timestep, time: datetime, reward, length):
        time = self.format_time(time)
        print(f"{timestep},{time},{reward},{length}")