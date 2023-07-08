import sys
from importlib import import_module
import unittest
from unittest import mock
import importlib

class TestTorchRL(unittest.TestCase):
    def test_torchrl_import(self):
        try:
            import_module('torchrl')
        except:
            self.fail("torchrl import failed")

class TestTorch(unittest.TestCase):
    def test_torch_import(self):
        try:
            import_module('torch')
        except:
            self.fail("torch import failed")

class TestNLE(unittest.TestCase):
    def test_nle_import(self):
        try:
            import_module('nle')
        except:
            self.fail("nle import failed")

class TestMinihack(unittest.TestCase):
    def test_minihack_import(self):
        try:
            import_module('minihack')
        except:
            self.fail("minihack import failed")