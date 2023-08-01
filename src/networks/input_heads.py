import torch
import torch.nn as nn
from tensordict import TensorDict

class GlyphHeadFlat(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=64):
        super().__init__()
        input_shape = input_shape[0] * input_shape[1]
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

class BlstatsHead(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=64):
        super().__init__()
        self.input_shape = input_shape[0]
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

class GlyphHeadConv(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=64):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 21 * 79, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_shape)

    def forward(self, x):
        x = torch.tanh(self.conv1(x.unsqueeze(1)))
        x = torch.tanh(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value


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