from abc import abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from torchrl.data import ReplayBuffer

from src.networks.input_heads import GlyphHeadFlat, GlyphHeadConv, GlyphBlstatHead, CartPoleHead
from src.agents.abstract_agent import AbstractAgent
from src.utils.env_specs import EnvSpecs
from src.buffers.rollout_buffer import RolloutBuffer
from src.utils.transition import Transition, TransitionFactory

class AbstractPPOAgent(AbstractAgent):
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
        return super().preprocess(state)

    def critic(self, state):
        with torch.no_grad():
            state_value = self._critic(state)
        return state_value
    
    def actor(self, state):
        with torch.no_grad():
            action_probs = self._actor(state)
        return action_probs

    def act(self, state, train=True):
        action_probs = self.actor(state)
        distribution = Categorical(action_probs)
        if train:
            action = distribution.sample()
        else:
            action = torch.argmax(action_probs)
        return action.cpu().numpy(), distribution.log_prob(action)
    
    def save_transition(self, *, state, action, reward, logprob, done):
        transition = self._transition_factory.create(state, action, reward, logprob, done)
        self._buffer.add(transition)
    
    def last_state(self, state):
        with torch.no_grad():
            state_value = self._critic(state)
        self._buffer.set_last_values(state_value)

    def train(self):
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
        
        self.actor_old.load_state_dict(self._actor.state_dict())
        self._buffer.clear()


    def load(self, path):
        paths = [path + 'actor.pkl', path + 'critic.pkl']
        for model, path in zip([self._actor, self._critic], paths):
            model.load_state_dict(torch.load(path))
        self.actor_old.load_state_dict(self._actor.state_dict())

    def save(self, path):
        paths = [path + 'actor.pkl', path + 'critic.pkl']
        for model, path in zip([self._actor, self._critic], paths):
            torch.save(model.state_dict(), path)


class GlyphPPOAgent(AbstractPPOAgent):
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
        self._actor = GlyphHeadFlat(
            self._observation_space['glyphs'],
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        self._actor_old = GlyphHeadFlat(
            self._observation_space['glyphs'],
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        self._critic = GlyphHeadFlat(
            self._observation_space['glyphs'],
            1,
            self._hidden_layer, actor=False,
            device=self._training_device,)
        self._actor_old.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self._critic_lr)

    def preprocess(self, observation, add_batch_dim=False):
        observation = torch.from_numpy(observation['glyphs']).to(dtype=self._tensor_type, device=self._training_device)
        if add_batch_dim:
            observation = observation.unsqueeze(0)
        return observation

class GlyphBlstatsPPOAgent(AbstractPPOAgent):
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
        self.actor_old = GlyphBlstatHead(
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
        self.actor_old.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self._critic_lr)

    def preprocess(self, observation, add_batch_dim=False):
        for key in observation.keys():
            observation[key] = torch.from_numpy(observation[key]).to(dtype=self._tensor_type, device=self._training_device)
            if add_batch_dim:
                observation[key] = observation[key].unsqueeze(0)
        return observation

class CartPolePPOAgent(AbstractPPOAgent):
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
        self._actor = CartPoleHead(
            self._observation_space,
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        self.actor_old = CartPoleHead(
            self._observation_space,
            self._num_actions,
            self._hidden_layer,
            device=self._training_device)
        self._critic = CartPoleHead(
            self._observation_space,
            1,
            self._hidden_layer, actor=False,
            device=self._training_device)
        self.actor_old.load_state_dict(self._actor.state_dict())
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self._critic_lr)

    def preprocess(self, observation, add_batch_dim=False):
        observation = torch.from_numpy(observation).to(dtype=self._tensor_type, device=self._training_device)
        if add_batch_dim:
            observation = observation.unsqueeze(0)
        return observation