import torch
import numpy as np

from src.buffers.abstract_buffer import AbstractBuffer
from src.utils.transition import Transition, TransitionFactory

class RolloutBuffer(AbstractBuffer):
    def __init__(
        self,
        transition_factory: TransitionFactory,
        buffer_size: int,
        num_envs=1,
        device=torch.device('cpu'),
        dtype=torch.float32
    ):
        super().__init__(buffer_size, num_envs)
        self.transition_factory = transition_factory
        self.device = device
        self.dtype = dtype
        self.pointer = 0
        self.feilds = ['state', 'action', 'reward', 'logprob', 'done', 'state_value', 'return', 'advantage']
        self._add = self._add_first

    def init_reshape(self):
        for feild in self.feilds:
            if feild in ['state',]:
                setattr(self, feild + '_reshape', lambda x: x)
                continue
            shape = getattr(self, feild + '_shape')
            if shape == (self.num_envs,):
                setattr(self, feild + '_reshape', lambda x: x.unsqueeze(-1))
            else:
                setattr(self, feild + '_reshape', lambda x: x)

    def reshape(self):
        for feild in self.feilds:
            reshape = getattr(self, feild + '_reshape')
            setattr(self, feild + 's', reshape(getattr(self, feild + 's')))

    def add(self, transition: Transition):
        self._add(transition)

    def _add_subsequent(self, transition: Transition):
        self.add_state(transition.state)
        self.actions[self.pointer] = transition.action
        self.rewards[self.pointer] = transition.reward
        self.logprobs[self.pointer] = transition.logprob
        self.state_values[self.pointer] = transition.state_value
        self.dones[self.pointer] = transition.done
        self.pointer += 1

    def _add_first(self, transition: Transition):
        self.state_shape = self.transition_factory.state_shape
        self.action_shape = self.transition_factory.action_shape
        self.reward_shape = self.transition_factory.reward_shape
        self.logprob_shape = self.transition_factory.logprob_shape
        self.done_shape = self.transition_factory.done_shape
        self.state_value_shape = self.transition_factory.state_value_shape
        self.return_shape = self.reward_shape 
        self.advantage_shape = self.reward_shape
        if isinstance(self.state_shape, dict):
            self.add_state = self.add_state_dict
        else:
            self.add_state = self.add_state_tensor
        self.init_reshape()
        self.init_zeros()
        self._add = self._add_subsequent
        self.add(transition)

    def set_last_values(self, last_values):
        if not isinstance(last_values, torch.Tensor):
            self.last_values = torch.tensor(last_values, device=self.device)
        else:
            self.last_values = last_values.to(self.device)

    def prepare(self):
        self.reshape()
        self.compute_GAE()
        self.concat_feilds()

    def compute_GAE(self, gamma=0.99, lambda_=0.95):
        advantages, returns = [], []
        state_values = torch.cat((self.state_values, self.last_values.unsqueeze(0)), dim=0)
        for t in reversed(range(self.buffer_size)):
            td_error = self.rewards[t] + gamma * (1 - self.dones[t]) * state_values[t+1] - self.state_values[t]
            advantages.append(td_error + gamma * lambda_ * (1 - self.dones[t]) * (advantages[-1] if advantages else 0))
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
            for key in self.keys:
                shape = self.states[key].shape
                shape = (shape[0] * shape[1], *shape[2:])
                self.states[key] = self.states[key].reshape(*shape)
        else:
            shape = self.states.shape
            shape = (shape[0] * shape[1], *shape[2:])
            self.states = self.states.reshape(*shape)
        for feild in self.feilds:
            if feild == 'state':
                continue
            shape = getattr(self, feild + 's').shape
            shape = (shape[0] * shape[1], *shape[2:])
            setattr(self, feild + 's', getattr(self, feild + 's').reshape(*shape))

    def clear(self):
        self.pointer = 0
        self.init_zeros()
    
    def init_zeros(self):
        if isinstance(self.state_shape, dict):
            self.keys = list(self.state_shape.keys())
            self.states = {key: torch.zeros((self.buffer_size, *self.state_shape[key]), dtype=self.dtype, device=self.device) for key in self.keys}
        else:
            self.states = torch.zeros((self.buffer_size, *self.state_shape), dtype=self.dtype, device=self.device)
        for feild in self.feilds:
            if feild == 'state':
                continue
            shape = getattr(self, feild + '_shape')
            setattr(self, feild + 's', torch.zeros((self.buffer_size, *shape), dtype=self.dtype, device=self.device))

    def get_batches(self, batch_size, seed=42, device=None):
        if device is None:
            device = self.device
        if isinstance(self.states, dict):
            for key in self.keys:
                self.states[key] = self.states[key].to(device)
        else:
            self.states = self.states.to(device)
        for feild in self.feilds:
            if feild == 'state':
                continue
            setattr(self, feild + 's', getattr(self, feild + 's').to(device))
        # get random indices
        indices = np.arange(self.buffer_size)
        np.random.seed(seed)
        np.random.shuffle(indices)
        indices = indices[:self.buffer_size - self.buffer_size % batch_size]
        # get batches
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            if isinstance(self.states, dict):
                state_batch = {key: self.states[key][indices[start:end]] for key in self.keys}
            else:
                state_batch = self.states[indices[start:end]]
            action_batch = self.actions[indices[start:end]]
            reward_batch = self.rewards[indices[start:end]]
            logprob_batch = self.logprobs[indices[start:end]]
            state_value_batch = self.state_values[indices[start:end]]
            done_batch = self.dones[indices[start:end]]
            return_batch = self.returns[indices[start:end]]
            advantage_batch = self.advantages[indices[start:end]]
            yield state_batch, action_batch, reward_batch, logprob_batch, state_value_batch, done_batch, return_batch, advantage_batch