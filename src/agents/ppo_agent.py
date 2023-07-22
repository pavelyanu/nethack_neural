from abc import abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchrl.data import ReplayBuffer

from src.networks.input_heads import GlyphHeadFlat, GlyphHeadConv, BlstatsHead
from src.agents.abstract_agent import AbstractAgent
from src.utils.env_specs import EnvSpecs
from src.buffers.rollout_buffer import RolloutBuffer

class Buffer:
    def __init__(self, keys=['glyphs', 'blstats']) -> None:
        self.keys = keys
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.done = []
        self.prepared = False
        for key in self.keys:
            setattr(self, key, [])
        
    def add(self, state, action, logprob, reward, state_value, done):
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.done.append(done)
        for key in self.keys:
            getattr(self, key).append(state[key])
    
    def prepare(self):
        length = len(self.actions)
        self.action_tensors = torch.tensor(self.actions).view(length, 1)
        self.logprob_tensors = torch.cat(self.logprobs, dim=0).view(length, 1)
        self.reward_tensors = torch.tensor(self.rewards).view(length, 1)
        self.state_value_tensors = torch.cat(self.state_values, dim=0)
        self.done_tensors= torch.tensor(self.done).view(length, 1)
        self.prepared = True
        state_tensors = {}
        for key in self.keys:
            key_tensors = torch.cat(getattr(self, key), dim=0)
            state_tensors[key] = key_tensors
        self.state_tensors = state_tensors


    def clear(self):
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.done.clear()
        for key in self.keys:
            getattr(self, key).clear()
        if self.prepared:
            del self.action_tensors
            del self.logprob_tensors
            del self.reward_tensors
            del self.state_value_tensors
            del self.done_tensors
            del self.state_tensors
            self.prepared = False

class GlyphBlstatHead(nn.Module):
    def __init__(self, glyph_shape, blstats_shape, output_shape, hidden_layer, actor=True) -> None:
        super().__init__()
        self.glyph_head = GlyphHeadFlat(glyph_shape, hidden_layer)
        self.blstats_head = BlstatsHead(blstats_shape, hidden_layer)
        self.fc = nn.Linear(2 * hidden_layer, output_shape)
        self.activation = nn.Softmax(dim=-1) if actor else nn.Identity()

    def forward(self, x):
        glyph = x['glyphs']
        blstats = x['blstats']
        glyph = self.glyph_head(glyph)
        blstats = self.blstats_head(blstats)
        x = torch.cat([glyph, blstats], dim=-1)
        x = self.fc(x)
        x = self.activation(x)
        return x

class AbstractPPOAgent(AbstractAgent):
    def __init__(
                self,
                env_specs,
                actor_lr=0.0001,
                critic_lr=0.001,
                gamma=0.99,
                epochs=5,
                eps_clip=0.2,
                batch_size=32,
                buffer_size=1000,
                hidden_layer=64):
        super().__init__(env_specs)
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._gamma = gamma
        self._epochs = epochs
        self._eps_clip = eps_clip
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._hidden_layer = hidden_layer
        self._buffer = RolloutBuffer(
            buffer_size=self._buffer_size,
            observation_shape=self._observation_space,
            action_shape=(1,),
            num_envs=self._num_envs,
        )

    @abstractmethod
    def preprocess(self, state):
        return super().preprocess(state)

    def act(self, state, train=True):
        with torch.no_grad():
            action_probs = self.actor(state)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        return action.numpy(), distribution.log_prob(action)
    
    def _save_transition(self, state, action, logprob, reward, next_state, done): 
        with torch.no_grad():
            state_value = self.critic(state)
        self.buffer.add(state, action, logprob, reward, state_value, done)
    
    def save_transition(self, state, action, logprob, reward, next_state, done):
        with torch.no_grad():
            state_value = self.critic(state)
        self._buffer.add(state, action, logprob, reward, state_value, done)

    def _train(self):
        self.buffer.prepare()
        old_states =self.buffer.state_tensors
        old_actions = self.buffer.action_tensors
        old_logprobs = self.buffer.logprob_tensors
        state_values = self.buffer.state_value_tensors
        rewards = self.buffer.reward_tensors
        dones = self.buffer.done_tensors

        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self._gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)

        std = returns.std()
        if torch.isnan(std):
            std = 1e-5

        returns = (returns - returns.mean()) / (std + 1e-5)
        returns = returns.unsqueeze(1)

        advantages = returns - state_values

        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._critic_lr)

        for _ in range(self._epochs):
            action_probs = self.actor(old_states)
            distribution = Categorical(action_probs)
            action_logprobs = distribution.log_prob(old_actions)
            state_values = self.critic(old_states)

            ratios = torch.exp(action_logprobs - old_logprobs.detach())

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self._eps_clip, 1 + self._eps_clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            critic_loss = nn.MSELoss()(state_values, returns)
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        self.actor_old.load_state_dict(self.actor.state_dict())

        self.buffer.clear()

    def train(self):
        self._buffer.prepare()
        old_states = self._buffer.states
        old_actions = self._buffer.actions
        old_logprobs = self._buffer.logprobs
        state_values = self._buffer.state_values
        rewards = self._buffer.rewards
        dones = self._buffer.done
        returns = self._buffer.returns
        advantages = self._buffer.advantages

        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._critic_lr)

        for _ in range(self._epochs):
            action_probs = self.actor(old_states)
            distribution = Categorical(action_probs)
            action_logprobs = distribution.log_prob(old_actions)
            state_values = self.critic(old_states)

            ratios = torch.exp(action_logprobs - old_logprobs.detach())

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self._eps_clip, 1 + self._eps_clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            critic_loss = nn.MSELoss()(state_values, returns)
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        self.actor_old.load_state_dict(self.actor.state_dict())

        self._buffer.clear()


    def load(self, path):
        paths = [path + 'actor.pkl', path + 'critic.pkl']
        for model, path in zip([self.actor, self.critic], paths):
            model.load_state_dict(torch.load(path))
        self.actor_old.load_state_dict(self.actor.state_dict())

    def save(self, path):
        paths = [path + 'actor.pkl', path + 'critic.pkl']
        for model, path in zip([self.actor, self.critic], paths):
            torch.save(model.state_dict(), path)


class GlyphPPOAgent(AbstractPPOAgent):
    def __init__(
        self,
        env_specs,
        actor_lr=0.0001,
        critic_lr=0.001,
        gamma=0.99,
        k_epochs=10,
        eps_clip=0.2,
        batch_size=32,
        buffer_size=10000,
        hidden_layer=64):
        super().__init__(env_specs, actor_lr, critic_lr, gamma, k_epochs, eps_clip, batch_size, buffer_size, hidden_layer)
        self.actor = GlyphHeadFlat(
            self._observation_space['glyphs'],
            self._num_actions,
            self._hidden_layer)
        self.actor_old = GlyphHeadFlat(
            self._observation_space['glyphs'],
            self._num_actions,
            self._hidden_layer)
        self.critic = GlyphHeadFlat(
            self._observation_space['glyphs'],
            1,
            self._hidden_layer, actor=False)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer = Buffer(keys=['glyphs'])

    def preprocess(self, observation):
        observation = torch.from_numpy(observation['glyphs']).to(torch.float32)
        return observation

class GlyphBlstatsPPOAgent(AbstractPPOAgent):
    def __init__(
        self,
        env_specs,
        actor_lr=0.0001,
        critic_lr=0.001,
        gamma=0.99,
        k_epochs=10,
        eps_clip=0.2,
        batch_size=32,
        buffer_size=10000,
        hidden_layer=64):
        super().__init__(env_specs, actor_lr, critic_lr, gamma, k_epochs, eps_clip, batch_size, buffer_size, hidden_layer)
        self.actor = GlyphBlstatHead(
            self._observation_space['glyphs'],
            self._observation_space['blstats'],
            self._num_actions,
            self._hidden_layer)
        self.actor_old = GlyphBlstatHead(
            self._observation_space['glyphs'],
            self._observation_space['blstats'],
            self._num_actions,
            self._hidden_layer)
        self.critic = GlyphBlstatHead(
            self._observation_space['glyphs'],
            self._observation_space['blstats'],
            1,
            self._hidden_layer, actor=False)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer = Buffer(keys=['glyphs', 'blstats'])

    def preprocess(self, observation):
        for key in observation.keys():
            observation[key] = torch.from_numpy(observation[key]).to(torch.float32)
        return observation