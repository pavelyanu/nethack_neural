import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchrl.data import ReplayBuffer

from src.networks.input_heads import GlyphHeadFlat, GlyphHeadConv, BlstatsHead
from src.agents.abstract_agent import AbstractAgent

class Buffer:
    def __init__(self):
        self.actions = []
        self.glyphs = []
        self.blstats = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.done = []
        self.prepared = False
        
    def add(self, state, action, logprob, reward, state_value, done):
        self.actions.append(action)
        self.glyphs.append(state['glyphs'])
        self.blstats.append(state['blstats'])
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.done.append(done)
    
    def prepare(self):
        length = len(self.actions)
        self.action_tensors = torch.tensor(self.actions).view(length, 1)
        self.glyph_tensors = torch.cat(self.glyphs, dim=0)
        self.blstat_tensors = torch.cat(self.blstats, dim=0)
        self.logprob_tensors = torch.cat(self.logprobs, dim=0).view(length, 1)
        self.reward_tensors = torch.tensor(self.rewards).view(length, 1)
        self.state_value_tensors = torch.cat(self.state_values, dim=0)
        self.done_tensors= torch.tensor(self.done).view(length, 1)
        self.prepared = True

    def clear(self):
        self.actions.clear()
        self.glyphs.clear()
        self.blstats.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.done.clear()
        if self.prepared:
            del self.action_tensors
            del self.glyph_tensors
            del self.blstat_tensors
            del self.logprob_tensors
            del self.reward_tensors
            del self.state_value_tensors
            del self.done_tensors
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

class PPOAgent(AbstractAgent):
    def __init__(
                self,
                observation_space,
                action_space,
                actor_lr=0.0001,
                critic_lr=0.001,
                gamma=0.99,
                k_epochs=10,
                eps_clip=0.2,
                batch_size=32,
                buffer_size=10000,
                hidden_layer=64):
        super().__init__(observation_space, action_space)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = Buffer()
        self.actor = GlyphBlstatHead(
            observation_space.spaces['glyphs'].shape,
            observation_space.spaces['blstats'].shape,
            action_space.n,
            hidden_layer)
        self.actor_old = GlyphBlstatHead(
            observation_space.spaces['glyphs'].shape,
            observation_space.spaces['blstats'].shape,
            action_space.n,
            hidden_layer)
        self.critic = GlyphBlstatHead(
            observation_space.spaces['glyphs'].shape,
            observation_space.spaces['blstats'].shape,
            1,
            hidden_layer, actor=False)
        self.actor_old.load_state_dict(self.actor.state_dict())

    def act(self, state, train=True):
        state = {key: torch.from_numpy(state[key]).float().unsqueeze(0) for key in state.keys()}
        with torch.no_grad():
            action_probs = self.actor(state)
        distribution = Categorical(action_probs)
        # if train:
        #     action = distribution.sample()
        # else:
        #     action = torch.argmax(action_probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def save_transition(self, state, action, logprob, reward, next_state, done): 
        glyphs = torch.from_numpy(state['glyphs']).float().unsqueeze(0)
        blstats = torch.from_numpy(state['blstats']).float().unsqueeze(0)
        with torch.no_grad():
            state_value = self.critic({'glyphs': glyphs, 'blstats': blstats})
        self.buffer.add({'glyphs': glyphs, 'blstats': blstats}, action, logprob, reward, state_value, done)



    def train(self):
        self.buffer.prepare()
        old_glyphs = self.buffer.glyph_tensors
        old_blstats = self.buffer.blstat_tensors
        old_states = {'glyphs': old_glyphs, 'blstats': old_blstats}
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
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        returns = returns.unsqueeze(1)

        advantages = returns - state_values

        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        for _ in range(self.k_epochs):
            action_probs = self.actor(old_states)
            distribution = Categorical(action_probs)
            action_logprobs = distribution.log_prob(old_actions)
            state_values = self.critic(old_states)

            ratios = torch.exp(action_logprobs - old_logprobs.detach())

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
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

    def load(self, path):
        paths = [path + 'actor.pkl', path + 'critic.pkl']
        for model, path in zip([self.actor, self.critic], paths):
            model.load_state_dict(torch.load(path))
        self.actor_old.load_state_dict(self.actor.state_dict())

    def save(self, path):
        paths = [path + 'actor.pkl', path + 'critic.pkl']
        for model, path in zip([self.actor, self.critic], paths):
            torch.save(model.state_dict(), path)
