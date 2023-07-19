import gym
import torch

class MinihackWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self._process_observation(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = self._process_observation(observation)
        return observation

    def _process_observation(self, observation):
        return observation['glyphs']
