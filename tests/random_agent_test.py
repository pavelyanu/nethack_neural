import unittest
import nle
import gym
from src.random_agent import RandomAgent

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



