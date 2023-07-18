import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import minihack
import gym
from src.agents.ppo_agent import PPOAgent
from src.runner import Runner
from src.loggers.file_logger import FileLogger

class TestPPO(unittest.TestCase):
    def test_ppo(self):
        env = gym.make("NetHackScore-v0")
        agent = PPOAgent(env.observation_space, env.action_space)
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
        env.close()