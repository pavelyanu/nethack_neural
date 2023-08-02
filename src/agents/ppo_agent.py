from abc import abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from torchrl.data import ReplayBuffer

from src.networks.input_heads import GlyphHeadFlat, GlyphHeadConv, GlyphBlstatHead, CartPoleHead, ActivationWrapper
from src.agents.abstract_agent import AbstractAgent
from src.utils.env_specs import EnvSpecs
from src.buffers.rollout_buffer import RolloutBuffer
from src.utils.transition import Transition, TransitionFactory

class AbstractPPOAgent(AbstractAgent):
    """Abstract base class for Proximal Policy Optimization (PPO) agents.

    This class provides a generic implementation of PPO, an algorithm for training policy-based agents in reinforcement learning. 
    It defines methods for acting in the environment, saving and loading models, and performing training updates. 
    Specific PPO agents can be implemented by inheriting from this class and implementing the abstract methods.

    Args:
        env_specs (EnvSpecs): The specifications of the environment.
        actor_lr (float): The learning rate for the actor network.
        critic_lr (float): The learning rate for the critic network.
        gamma (float): The discount factor.
        epochs (int): The number of epochs to run during each update.
        eps_clip (float): The clip range for the policy update.
        batch_size (int): The size of the mini-batches used for updates.
        buffer_size (int): The size of the buffer used for storing transitions.
        hidden_layer (int): The size of the hidden layers in the actor and critic networks.
        storage_device (str): The device to use for storing transitions ('cpu' or 'cuda').
        training_device (str): The device to use for training ('cpu' or 'cuda').
        tensor_type (torch.dtype): The type of the tensors used in computations.
    """
    def __init__(
                self,
                env_specs,
                actor_lr=0.0001,
                critic_lr=0.001,
                gamma=0.99,
                epochs=10,
                eps_clip=0.2,
                batch_size=64,
                buffer_size=2000,
                hidden_layer=64,
                storage_device='cpu',
                training_device=None,
                tensor_type=torch.float32):
        super().__init__(env_specs)
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._gamma = gamma
        self._epochs = epochs
        self._eps_clip = eps_clip
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._hidden_layer = hidden_layer
        self._storage_device = torch.device(storage_device)
        self._tensor_type = tensor_type
        if training_device is None:
            self._training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            if training_device not in ['cuda', 'cpu']:
                raise ValueError('Invalid device')
            elif training_device == 'cuda' and not torch.cuda.is_available():
                raise ValueError('Cuda device not available')
            else:
                self._training_device = torch.device(training_device)
        self._transition_factory = TransitionFactory(self, device=self._storage_device, dtype=self._tensor_type)
        self._buffer = RolloutBuffer(
            self._transition_factory,
            buffer_size=self._buffer_size,
            num_envs=self._num_envs,
            device=self._storage_device)
        self.counter = 0

    @abstractmethod
    def preprocess(self, state):
        """Preprocesses the state before feeding it into the networks.

        Args:
            state: The state to be preprocessed.

        Returns:
            The preprocessed state.
        """
        return super().preprocess(state)

    @abstractmethod
    def hasbatchdim(self, state):
        """Checks if the state has a batch dimension.

        Args:
            state: The state to be checked.

        Returns:
            bool: True if the state has a batch dimension, False otherwise.
        """
        pass

    def critic(self, state):
        """Computes the value estimates for the given state using the critic network.

        Args:
            state: The state for which to compute the value estimates.

        Returns:
            The value estimates computed by the critic network.
        """
        with torch.no_grad():
            state_value = self._critic(state)
        return state_value
    
    def actor(self, state):
        """Computes the action probabilities for the given state using the actor network.

        Args:
            state: The state for which to compute the action probabilities.

        Returns:
            The action probabilities computed by the actor network.
        """
        with torch.no_grad():
            action_probs = self._actor(state)
        return action_probs

    def act(self, state, train=True):
        """Selects an action for the given state using the actor network.

        Args:
            state: The state for which to select an action.
            train (bool): Whether the agent is training or not.

        Returns:
            The selected action and the corresponding log-probability.
        """
        state = self.preprocess(state)
        action_probs = self.actor(state)
        distribution = Categorical(action_probs)
        if train:
            action = distribution.sample()
        else:
            action = torch.argmax(action_probs)
        return action.cpu().numpy(), distribution.log_prob(action)
    
    def save_transition(self, *, state, action, reward, logprob, done):
        """Stores a transition in the buffer.

        Args:
            state: The state from which the action was taken.
            action: The action that was taken.
            reward: The reward that was obtained.
            logprob: The log-probability of the action.
            done: Whether the episode ended after this action.
        """
        transition = self._transition_factory.create(state, action, reward, logprob, done)
        self._buffer.add(transition)
    
    def last_state(self, state):
        """Processes the final state of an episode.

        Args:
            state: The final state of the episode.
        """
        state = self.preprocess(state)
        with torch.no_grad():
            state_value = self._critic(state)
        self._buffer.set_last_values(state_value)

    def train(self):
        """Performs a training update."""
        self._buffer.prepare()
        for _ in range(self._epochs):
            for batch in self._buffer.get_batches(self._batch_size):
                states_batch, actions_batch, rewards_batch, logprobs_batch, state_values_batch, dones_batch, returns_batch, advantages_batch = batch
                action_probs = self._actor(states_batch)
                distribution = Categorical(action_probs)
                action_logprobs = distribution.log_prob(torch.squeeze(actions_batch, -1)).unsqueeze(-1)
                state_values = self._critic(states_batch)

                ratios = torch.exp(action_logprobs - logprobs_batch.detach())

                surrogate1 = ratios * advantages_batch
                surrogate2 = torch.clamp(ratios, 1 - self._eps_clip, 1 + self._eps_clip) * advantages_batch
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                critic_loss = nn.MSELoss()(state_values, returns_batch)
                
                self._actor_optimizer.zero_grad()
                actor_loss.backward()
                self._actor_optimizer.step()

                self._critic_optimizer.zero_grad()
                critic_loss.backward()
                self._critic_optimizer.step()
        
        self._actor_old.load_state_dict(self._actor.state_dict())
        self._buffer.clear()


    def load(self, path):
        """Loads the model parameters from the specified path.

        Args:
            path (str): The path from which to load the model parameters.
        """ 
        paths = [path + 'actor.pkl', path + 'critic.pkl']
        for model, path in zip([self._actor, self._critic], paths):
            model.load_state_dict(torch.load(path))
        self._actor_old.load_state_dict(self._actor.state_dict())

    def save(self, path):
        """Saves the model parameters to the specified path.

        Args:
            path (str): The path to which to save the model parameters.
        """
        paths = [path + 'actor.pkl', path + 'critic.pkl']
        for model, path in zip([self._actor, self._critic], paths):
            torch.save(model.state_dict(), path)


class GlyphPPOAgent(AbstractPPOAgent):
    """PPO agent that uses a GlyphHeadFlat network.

    This agent is specifically designed for environments where the states are represented as glyphs.

    Args:
        env_specs (EnvSpecs): The specifications of the environment.
        actor_lr (float): The learning rate for the actor network.
        critic_lr (float): The learning rate for the critic network.
        gamma (float): The discount factor.
        epochs (int): The number of epochs to run during each update.
        eps_clip (float): The clip range for the policy update.
        batch_size (int): The size of the mini-batches used for updates.
        buffer_size (int): The size of the buffer used for storing transitions.
        hidden_layer (int): The size of the hidden layers in the actor and critic networks.
        storage_device (str): The device to use for storing transitions ('cpu' or 'cuda').
        training_device (str): The device to use for training ('cpu' or 'cuda').
        tensor_type (torch.dtype): The type of the tensors used in computations.
    """
    def __init__(
            self,
            env_specs,
            actor_lr=0.0001,
            critic_lr=0.001,
            gamma=0.99,
            epochs=10,
            eps_clip=0.2,
            batch_size=64,
            buffer_size=2000,
            hidden_layer=64,
            storage_device='cpu',
            training_device=None,
            tensor_type=torch.float32):
        super().__init__(env_specs, actor_lr, critic_lr, gamma, epochs, eps_clip, batch_size, buffer_size, hidden_layer, storage_device, training_device, tensor_type)
        actor_net = GlyphHeadFlat(
            self._observation_space['glyphs'],
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        actor_old_net = GlyphHeadFlat(
            self._observation_space['glyphs'],
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        critic_net = GlyphHeadFlat(
            self._observation_space['glyphs'],
            1,
            self._hidden_layer,
            device=self._training_device,)
        self._actor = ActivationWrapper(actor_net, nn.Softmax(dim=-1))
        self._actor_old = ActivationWrapper(actor_old_net, nn.Softmax(dim=-1))
        self._critic = critic_net
        self._actor_old.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self._critic_lr)

    def preprocess(self, observation):
        observation = torch.from_numpy(observation['glyphs']).to(dtype=self._tensor_type, device=self._training_device)
        if not self.hasbatchdim(observation):
            observation = observation.unsqueeze(0)
        return observation
    
    def hasbatchdim(self, state):
        state = state['glyphs']
        if state.shape == self._observation_space['glyphs']:
            return False
        else:
            return True


class GlyphBlstatsPPOAgent(AbstractPPOAgent):
    """PPO agent that uses a GlyphBlstatHead network.

    This agent is specifically designed for environments where the states are represented as both glyphs and blstats.

    Args:
        env_specs (EnvSpecs): The specifications of the environment.
        actor_lr (float): The learning rate for the actor network.
        critic_lr (float): The learning rate for the critic network.
        gamma (float): The discount factor.
        epochs (int): The number of epochs to run during each update.
        eps_clip (float): The clip range for the policy update.
        batch_size (int): The size of the mini-batches used for updates.
        buffer_size (int): The size of the buffer used for storing transitions.
        hidden_layer (int): The size of the hidden layers in the actor and critic networks.
        storage_device (str): The device to use for storing transitions ('cpu' or 'cuda').
        training_device (str): The device to use for training ('cpu' or 'cuda').
        tensor_type (torch.dtype): The type of the tensors used in computations.
    """
    def __init__(
            self,
            env_specs,
            actor_lr=0.0001,
            critic_lr=0.001,
            gamma=0.99,
            epochs=10,
            eps_clip=0.2,
            batch_size=64,
            buffer_size=2000,
            hidden_layer=64,
            storage_device='cpu',
            training_device=None,
            tensor_type=torch.float32):
        super().__init__(env_specs, actor_lr, critic_lr, gamma, epochs, eps_clip, batch_size, buffer_size, hidden_layer, storage_device, training_device, tensor_type)
        self._actor = GlyphBlstatHead(
            self._observation_space['glyphs'],
            self._observation_space['blstats'],
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        self._actor_old = GlyphBlstatHead(
            self._observation_space['glyphs'],
            self._observation_space['blstats'],
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        self._critic = GlyphBlstatHead(
            self._observation_space['glyphs'],
            self._observation_space['blstats'],
            1,
            self._hidden_layer, actor=False,
            device=self._training_device)
        self._actor_old.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self._critic_lr)

    def preprocess(self, observation):
        observation = {key: torch.from_numpy(observation[key]).to(dtype=self._tensor_type, device=self._training_device) for key in observation.keys()}
        if not self.hasbatchdim(observation):
            for key in observation.keys():
                observation[key] = observation[key].unsqueeze(0)
        return observation

    def hasbatchdim(self, state):
        state = state['glyphs']
        if state.shape == self._observation_space['glyphs']:
            return False
        else:
            return True
    
class CartPolePPOAgent(AbstractPPOAgent):
    """PPO agent that uses a CartPoleHead network.

    This agent is specifically designed for the CartPole environment.

    Args:
        env_specs (EnvSpecs): The specifications of the environment.
        actor_lr (float): The learning rate for the actor network.
        critic_lr (float): The learning rate for the critic network.
        gamma (float): The discount factor.
        epochs (int): The number of epochs to run during each update.
        eps_clip (float): The clip range for the policy update.
        batch_size (int): The size of the mini-batches used for updates.
        buffer_size (int): The size of the buffer used for storing transitions.
        hidden_layer (int): The size of the hidden layers in the actor and critic networks.
        storage_device (str): The device to use for storing transitions ('cpu' or 'cuda').
        training_device (str): The device to use for training ('cpu' or 'cuda').
        tensor_type (torch.dtype): The type of the tensors used in computations.
    """
    def __init__(
            self,
            env_specs,
            actor_lr=0.0001,
            critic_lr=0.001,
            gamma=0.99,
            epochs=10,
            eps_clip=0.2,
            batch_size=64,
            buffer_size=2000,
            hidden_layer=64,
            storage_device='cpu',
            training_device=None,
            tensor_type=torch.float32):
        super().__init__(env_specs, actor_lr, critic_lr, gamma, epochs, eps_clip, batch_size, buffer_size, hidden_layer, storage_device, training_device, tensor_type)
        actor_net = CartPoleHead(
            self._observation_space,
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        actor_old_net = CartPoleHead(
            self._observation_space,
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        critic_net = CartPoleHead(
            self._observation_space,
            1,
            self._hidden_layer,
            device=self._training_device)
        self._actor = ActivationWrapper(actor_net, nn.Softmax(dim=-1))
        self._actor_old = ActivationWrapper(actor_old_net, nn.Softmax(dim=-1))
        self._critic = critic_net
        self._actor_old.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self._critic_lr)

    def preprocess(self, observation):
        observation = torch.from_numpy(observation).to(dtype=self._tensor_type, device=self._training_device)
        if not self.hasbatchdim(observation):
            observation = observation.unsqueeze(0)
        return observation
    
    def hasbatchdim(self, state):
        if state.shape == self._observation_space:
            return False
        else:
            return True