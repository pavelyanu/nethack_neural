import torch
import numpy as np

from src.buffers.abstract_buffer import AbstractBuffer
from src.utils.transition import Transition, TransitionFactory

class RolloutBuffer(AbstractBuffer):
    """A class that manages the rollout buffer used for policy optimization.

    This class inherits from the AbstractBuffer and is specifically designed for Proximal Policy Optimization (PPO). 
    It keeps track of the transitions (state, action, reward, next_state, done) and provides functionality for adding new transitions, 
    computing Generalized Advantage Estimation (GAE), and preparing the data for learning.

    Args:
        transition_factory (TransitionFactory): An object of class TransitionFactory that provides functionality to create new transitions.
        buffer_size (int): Maximum number of transitions that can be stored in the buffer.
        num_envs (int, optional): Number of parallel environments. Default is 1.
        device (torch.device, optional): Device where the tensors will be stored. Default is CPU.
        dtype (torch.dtype, optional): Data type of the stored tensors. Default is float32.

    Attributes:
        transition_factory (TransitionFactory): An object of class TransitionFactory that provides functionality to create new transitions.
        buffer_size (int): Maximum number of transitions that can be stored in the buffer.
        num_envs (int): Number of parallel environments.
        device (torch.device): Device where the tensors will be stored.
        dtype (torch.dtype): Data type of the stored tensors.
        pointer (int): Pointer to the next empty spot in the buffer.
        fields (list of str): List of names of the fields stored in the buffer.
        _add (method): Method for adding a new transition into the buffer.
    """
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
        """Initializes the reshaping functions for each field based on their shapes."""
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
        """Reshapes the tensors of each field according to their reshaping functions."""
        for feild in self.feilds:
            reshape = getattr(self, feild + '_reshape')
            setattr(self, feild + 's', reshape(getattr(self, feild + 's')))

    def add(self, transition: Transition):
        """Adds a new transition to the buffer.

        Args:
            transition (Transition): The transition to be added.
        """
        self._add(transition)

    def _add_subsequent(self, transition: Transition):
        """Adds a new transition to the buffer after the first one has been added.

        Args:
            transition (Transition): The transition to be added.
        """
        self.add_state(transition.state)
        self.actions[self.pointer] = transition.action
        self.rewards[self.pointer] = transition.reward
        self.logprobs[self.pointer] = transition.logprob
        self.state_values[self.pointer] = transition.state_value
        self.dones[self.pointer] = transition.done
        self.pointer += 1

    def _add_first(self, transition: Transition):
        """Adds the first transition to the buffer and initializes the shapes and addition functions for the fields.

        Args:
            transition (Transition): The transition to be added.
        """
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
        """Sets the value estimates of the final state in the last transition of the buffer.

        Args:
            last_values (torch.Tensor or ndarray): Value estimates of the final state.
        """
        if not isinstance(last_values, torch.Tensor):
            self.last_values = torch.tensor(last_values, device=self.device)
        else:
            self.last_values = last_values.to(self.device)

    def prepare(self):
        """Prepares the buffer for learning.

        Reshapes the tensors, computes the Generalized Advantage Estimation (GAE), and concatenates the fields.
        """
        self.reshape()
        self.compute_GAE()
        self.concat_feilds()

    def compute_GAE(self, gamma=0.99, lambda_=0.95):
        """Computes the Generalized Advantage Estimation (GAE) for all transitions in the buffer.

        Args:
            gamma (float, optional): Discount factor. Default is 0.99.
            lambda_ (float, optional): GAE hyperparameter. Default is 0.95.
        """
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
        """Adds a new state to the buffer.

        Args:
            state (torch.Tensor): The state to be added.
        """
        self.states[self.pointer] = state

    def add_state_dict(self, state):
        """Adds a new state to the buffer in case the state is represented as a dictionary.

        Args:
            state (dict): The state to be added.
        """
        for key in self.keys:
            self.states[key][self.pointer] = state[key]

    def concat_feilds(self):
        """Concatenates the fields of the buffer."""
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
        """Clears the buffer and resets the pointer."""
        self.pointer = 0
        self.init_zeros()
    
    def init_zeros(self):
        """Initializes the fields of the buffer with zeros."""
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
        """Generates batches of transitions.

        Args:
            batch_size (int): The size of the batches to be returned.
            seed (int, optional): Seed for the random number generator. Default is 42.
            device (torch.device, optional): Device where the tensors will be moved. Default is None, which means they will stay on the original device.

        Yields:
            tuple: A tuple containing batches of states, actions, rewards, log-probs, dones, returns, and advantages.
        """
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
        indices = np.arange(self.buffer_size)
        np.random.seed(seed)
        np.random.shuffle(indices)
        indices = indices[:self.buffer_size - self.buffer_size % batch_size]
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