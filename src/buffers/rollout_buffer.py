import torch
from torch.utils.data import Dataset
import numpy as np

from src.buffers.abstract_buffer import AbstractBuffer
from src.utils.transition import Transition

class RolloutBuffer(AbstractBuffer):
    def __init__(
        self,
        buffer_size,
        observation_shape,
        action_shape,
        num_envs=1,
        device=torch.device('cpu'),
        dtype=torch.float32
    ):
        super().__init__(buffer_size, num_envs)
        self.device = device
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.dtype = dtype
        self.pointer = 0
        self.init_zeros()
        
    def add(self, transition: Transition):
        transition.to_tensor()
        transition.unsqueeze()
        transition.to(self.device)
        self.add_state(transition.state)
        self.actions[self.pointer] = transition.action
        self.rewards[self.pointer] = transition.reward
        self.logprobs[self.pointer] = transition.logprob
        self.state_values[self.pointer] = transition.state_value
        self.done[self.pointer] = transition.done
        self.pointer += 1

    def set_last_values(self, last_values):
        if not isinstance(last_values, torch.Tensor):
            self.last_values = torch.tensor(last_values, device=self.device)
        else:
            self.last_values = last_values.to(self.device)

    def prepare(self):
        self.compute_GAE()
        self.concat_feilds()

    def compute_GAE(self, gamma=0.99, lambda_=0.95):
        advantages, returns = [], []
        self.done = self.done.to(self.dtype)
        state_values = torch.cat((self.state_values, self.last_values.unsqueeze(0)), dim=0)
        for t in reversed(range(self.buffer_size)):
            td_error = self.rewards[t] + gamma * (1 - self.done[t]) * state_values[t+1] - self.state_values[t]
            advantages.append(td_error + gamma * lambda_ * (1 - self.done[t]) * (advantages[-1] if advantages else 0))
            returns.append(advantages[-1] + self.state_values[t])
        advantages.reverse(), returns.reverse()
        self.advantages = torch.stack(advantages)
        self.returns = torch.stack(returns)

    def add_state_tensor(self, state):
        self.states[self.pointer] = state

    def add_state_dict(self, state):
        for key in self.keys:
            self.states[key][self.pointer] = state[key]

    def concat_feilds(self):
        if isinstance(self.states, dict):
            self.states = {key: self.states[key].reshape(-1, *self.observation_shape[key]) for key in self.keys}
        else:
            self.states = self.states.reshape(-1, *self.observation_shape)
        self.actions = self.actions.reshape(-1, *self.action_shape)
        self.rewards = self.rewards.reshape(-1, 1)
        self.logprobs = self.logprobs.reshape(-1, 1)
        self.state_values = self.state_values.reshape(-1, 1)
        self.done = self.done.reshape(-1, 1)
        self.returns = self.returns.reshape(-1, 1)
        self.advantages = self.advantages.reshape(-1, 1)

    def clear(self):
        self.pointer = 0
        self.init_zeros()
    
    def init_zeros(self):
        if isinstance(self.observation_shape, dict):
            self.keys = list(self.observation_shape.keys())
            self.states = {key: torch.zeros((self.buffer_size, self.num_envs, *self.observation_shape[key]), dtype=self.dtype, device=self.device) for key in self.keys}
            self.add_state = self.add_state_dict
        else:
            self.states = torch.zeros((self.buffer_size, self.num_envs, *self.observation_shape), dtype=self.dtype, device=self.device)
            self.add_state = self.add_state_tensor
        self.actions = torch.zeros((self.buffer_size, self.num_envs, *self.action_shape), dtype=self.dtype, device=self.device)
        self.rewards = torch.zeros((self.buffer_size, self.num_envs, 1), dtype=self.dtype, device=self.device)
        self.logprobs = torch.zeros((self.buffer_size, self.num_envs, 1), dtype=self.dtype, device=self.device)
        self.state_values = torch.zeros((self.buffer_size, self.num_envs, 1), dtype=self.dtype, device=self.device)
        self.done = torch.zeros((self.buffer_size, self.num_envs, 1), dtype=torch.bool, device=self.device)
        self.returns = torch.zeros((self.buffer_size, self.num_envs, 1), dtype=self.dtype, device=self.device)
        self.advantages = torch.zeros((self.buffer_size, self.num_envs, 1), dtype=self.dtype, device=self.device)
        self.last_values = torch.zeros((self.num_envs, 1), dtype=self.dtype, device=self.device)


class RolloutBufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(next(iter(self.buffer.states.values())))

    def __getitem__(self, idx):
        if isinstance(self.buffer.states, dict):
            states = {key: value[idx] for key, value in self.buffer.states.items()}
        else:
            states = self.buffer.states[idx]
        return (states, self.buffer.actions[idx], self.buffer.rewards[idx], 
                self.buffer.logprobs[idx], self.buffer.state_values[idx], 
                self.buffer.done[idx], self.buffer.returns[idx], 
                self.buffer.advantages[idx])