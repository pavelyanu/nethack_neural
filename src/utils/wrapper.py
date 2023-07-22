import gym
import torch
from tensordict import TensorDict

class MinihackObservationWrapper(gym.Wrapper):
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
        for key in observation.keys():
            observation[key] = torch.tensor(observation[key]).unsqueeze(0).to(torch.float32)
        return observation

class MinihackObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, observation):
        for key in observation.keys():
            observation[key] = torch.tensor(observation[key]).unsqueeze(0).to(torch.float32)
        return observation
    
    
class MinihackTensorDictWrapper(gym.Wrapper):
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
        dictionary = {}
        for key in observation.keys():
            dictionary[key] = torch.tensor(observation[key]).unsqueeze(0).to(torch.float32)
        ret = TensorDict(dictionary, batch_size=1)
        return ret