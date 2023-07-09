from src.loggers.file_logger import FileLogger
from src.loggers.stdout_logger import StdoutLogger
import unittest
import os

class LoggerTest(unittest.TestCase):
    def test_stdout_logger(self):
        logger = StdoutLogger()
        logger.log("Hello, World!")

    def test_file_logger(self):
        logger = FileLogger("test.log")
        logger.log("Hello, World!")
        with open("test.log", 'r') as f:
            self.assertEqual(f.read(), "Hello, World!\n")
        os.remove("test.log")
