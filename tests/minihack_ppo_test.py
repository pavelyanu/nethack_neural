import sys
import os

import unittest
import minihack
import gym
from src.agents.ppo_simple import PPOSimpleAgent
from src.runners.default_runner import Runner
from src.loggers.file_logger import FileLogger

class TestPPO(unittest.TestCase):
    def test_ppo(self):
        env = gym.make("NetHackScore-v0")
        agent = PPOSimpleAgent(env.observation_space, env.action_space)
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
        env.close()