"""Small MLP velocity field v(x, t, y). y is a multi-hot vector: 1 = class present, 0 = unknown/masked."""
import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(100.0), dim // 2)))

    def forward(self, t):
        a = t[:, None] * self.freqs[None] * 2 * math.pi
        return torch.cat([a.sin(), a.cos()], dim=-1)


class VelocityMLP(nn.Module):
    def __init__(self, n_classes, hidden=256, depth=4, t_dim=32):
        super().__init__()
        self.temb = TimeEmbedding(t_dim)
        layers, d = [], 2 + t_dim + n_classes
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, y):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if y.dim() == 1:
            y = y.expand(x.shape[0], -1)
        return self.net(torch.cat([x, self.temb(t), y], dim=-1))
