import sys
import os

from nethack_neural.loggers.file_message_logger import FileMessageLogger
from nethack_neural.loggers.stdout_logger import StdoutLogger
import unittest
import os

class LoggerTest(unittest.TestCase):
    def test_stdout_logger(self):
        logger = StdoutLogger()
        logger.log_message("Hello, World!")

    def test_file_logger(self):
        logger = FileMessageLogger("test.log")
        logger.log_message("Hello, World!")
        with open("test.log", 'r') as f:
            self.assertEqual(f.read(), "Hello, World!\n")
        os.remove("test.log")
