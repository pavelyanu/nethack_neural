import sys
import os

import unittest
import nle
import gym
from nethack_neural.agents.random_agent import RandomAgent
from nethack_neural.runners.default_runner import Runner
from nethack_neural.loggers.file_message_logger import FileMessageLogger
from nethack_neural.utils.env_specs import EnvSpecs

class RandomAgentTest(unittest.TestCase):
    def test_random_agent(self):
        env = gym.make("NetHackScore-v0")
        env_specs = EnvSpecs()
        env_specs.init_with_gym_env(env, num_envs=1)
        agent = RandomAgent(env_specs)
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
        env.close()

    def test_with_runner(self):
        env = gym.make("NetHackScore-v0")
        agent = RandomAgent(env.observation_space, env.action_space)
        runner = Runner(env, agent, num_episodes=100, loggers=FileMessageLogger(path="tests/test.log"))
        runner.train()
        runner.evaluate()



