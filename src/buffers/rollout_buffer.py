import torch
import numpy as np

from src.buffers.abstract_buffer import AbstractBuffer

class RolloutBuffer(AbstractBuffer):
    def __init__(
        self,
        buffer_size,
        observation_shape,
        action_shape,
        n_agents=1,
        device='cpu',
        dtype=torch.float32
    ):
        super().__init__(buffer_size, n_agents)
        self.device = device
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.dtype = dtype
        if isinstance(observation_shape, dict):
            self.keys = list(observation_shape.keys())
            self.states = {key: torch.zeros((buffer_size, n_agents, *observation_shape[key]), dtype=self.dtype, device=self.device) for key in self.keys}
            self.add_state = self.add_state_dict
        else:
            self.states = torch.zeros((buffer_size, n_agents, *observation_shape), dtype=self.dtype, device=self.device)
            self.add_state = self.add_state_tensor
        self.actions = torch.zeros((buffer_size, n_agents, *action_shape), dtype=self.dtype, device=self.device)
        self.rewards = torch.zeros((buffer_size, n_agents, 1), dtype=self.dtype, device=self.device)
        self.logprobs = torch.zeros((buffer_size, n_agents, 1), dtype=self.dtype, device=self.device)
        self.state_values = torch.zeros((buffer_size, n_agents, 1), dtype=self.dtype, device=self.device)
        self.done = torch.zeros((buffer_size, n_agents, 1), dtype=torch.bool, device=self.device)
        self.returns = torch.zeros((buffer_size, n_agents, 1), dtype=self.dtype, device=self.device)
        self.advantages = torch.zeros((buffer_size, n_agents, 1), dtype=self.dtype, device=self.device)
        
    def add(self, state, action, reward, logprob, state_value, done):
        self.add_state(state)
        self.actions[self.pointer] = torch.tensor(action, device=self.device)
        self.rewards[self.pointer] = torch.tensor(reward, device=self.device)
        self.logprobs[self.pointer] = torch.tensor(logprob, device=self.device)
        self.state_values[self.pointer] = torch.tensor(state_value, device=self.device)
        self.done[self.pointer] = torch.tensor(done, device=self.device)
        self.pointer += 1

    def prepare(self):
        self.compute_GAE()

    def compute_GAE(self, gamma=0.99, lambda_=0.95):
        advantages, returns = [], []
        for t in reversed(range(self.buffer_size)):
            td_error = self.rewards[t] + gamma * (1 - self.done[t]) * self.values[t+1] - self.values[t]
            advantages.append(td_error + gamma * lambda_ * (1 - self.done[t]) * (advantages[-1] if advantages else 0))
            returns.append(advantages[-1] + self.values[t])
        advantages.reverse(), returns.reverse()
        self.advantages = torch.stack(advantages)
        self.returns = torch.stack(returns)

    def add_state_tensor(self, state):
        self.states[self.pointer] = torch.tensor(state, device=self.device)

    def add_state_dict(self, state):
        for key in self.keys:
            self.states[key][self.pointer] = torch.tensor(state[key], device=self.device)
