import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import nle
import gym
from src.agents.random_agent import RandomAgent
from src.runner import Runner
from src.loggers.file_logger import FileLogger

class RandomAgentTest(unittest.TestCase):
    def test_random_agent(self):
        env = gym.make("NetHackScore-v0")
        agent = RandomAgent(env.observation_space, env.action_space)
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
        env.close()

    def test_with_runner(self):
        env = gym.make("NetHackScore-v0")
        agent = RandomAgent(env.observation_space, env.action_space)
        runner = Runner(env, agent, num_episodes=100, loggers=FileLogger(path="tests/test.log"))
        runner.train()
        runner.evaluate()



