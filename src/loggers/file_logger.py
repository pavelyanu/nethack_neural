from .abstract_logger import AbstractLogger

class FileLogger(AbstractLogger):
    def __init__(self, path, name="file_logger"):
        super().__init__(name)
        self.path = path


    def log(self, msg):
        with open(self.path, 'a') as f:
            f.write(msg + '\n')
