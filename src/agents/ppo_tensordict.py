import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from src.networks.input_heads import GlyphHeadFlat, GlyphHeadConv, BlstatsHead
from src.agents.abstract_agent import AbstractAgent

def create_heads(observation_space, output_shape, hidden_size=64, use_conv=False):
    spaces = observation_space.spaces
    heads = []
    if "glyphs" in spaces:
        if use_conv == True:
            glyph_head = GlyphHeadConv(spaces["glyphs"].shape, output_shape, hidden_size)
        else:
            glyph_head = GlyphHeadFlat(spaces["glyphs"].shape, output_shape, hidden_size)
        glyph_tensordict_module = TensorDictModule(glyph_head, in_keys=["glyphs"], out_keys=["glyphs_out"])
        heads.append(glyph_tensordict_module)
    if "blstats" in spaces:
        blstats_head = BlstatsHead(spaces["blstats"].shape, output_shape, hidden_size)
        blstats_tensordict_module = TensorDictModule(blstats_head, in_keys=["blstats"], out_keys=["blstats_out"])
        heads.append(blstats_tensordict_module)

    return heads

class HeadConcat(nn.Module):
    def __init__(self, input_shape, output_shape, input_keys, hidden_size=64):
        super().__init__()
        # concat input of every key
        input_shape = sum([input_shape[key] for key in input_keys])
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        x = torch.cat([x["glyphs_out"], x["blstats_out"]], dim=-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

class PPOHead(TensorDictSequential):
    def __init__(self, observation_space, action_space, hidden_size=64, use_conv=False):
        self.heads = create_heads(observation_space, action_space.n, hidden_size, use_conv)
        super().__init__(*self.heads)

class PPOAgent(AbstractAgent):
    def __init__(self, observation_space, action_space, lr=0.002, gamma=0.99, k_epochs=4, eps_clip=0.2):
        super().__init__(observation_space, action_space)
        
        self.policy = PPOHead(observation_space, action_space)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
    def act(self, state):
        assert isinstance(state, TensorDict)
        action_probs = self.policy(state)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        return action.item()
        
    def learning_step(self, state, action, reward, next_state, done):
        # convert to tensor
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = torch.tensor([action]).unsqueeze(0)
        reward = torch.tensor([reward]).unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        done = torch.tensor([done]).unsqueeze(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        if done:
            self.update_policy()
            
    def update_policy(self):
        returns = self.compute_returns(self.rewards, self.gamma)
        returns = returns.view(-1, 1)
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_action_probs = self.policy(states).gather(1, actions.view(-1, 1)).detach()
        
        for _ in range(self.k_epochs):
            action_probs = self.policy(states).gather(1, actions.view(-1, 1))
            ratios = action_probs / old_action_probs
            advantages = returns - self.policy(states).detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
    @staticmethod
    def compute_returns(rewards, gamma):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))



    

        