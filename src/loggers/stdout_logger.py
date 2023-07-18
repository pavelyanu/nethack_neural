from .abstract_logger import AbstractLogger

class StdoutLogger(AbstractLogger):
    def __init__(self, name="stdout_logger"):
        super().__init__(name)

    def log(self, msg):
        print(msg)