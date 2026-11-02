"""UNet velocity field v(x, t, y) for 64x64 images (DDPM-style, ~35M params).

y is multi-hot over the stage-2 labels: 1 = label present, 0 = unknown/masked (as in flowtoy).
The label embedding is a bias-free linear map added to the time embedding, initialised to zero,
so a stage-1 (unconditional, n_labels=0) checkpoint loads as-is and y=0 is exactly unconditional.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = 1000.0 * t[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


def norm(c):
    return nn.GroupNorm(32, c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout):
        super().__init__()
        self.n1, self.c1 = norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.n2, self.c2 = norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n, self.qkv, self.out, self.heads = norm(c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1), heads
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, self.heads, c // self.heads, h * w).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v)
        return x + self.out(o.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, n_labels=0, ch=128, mult=(1, 2, 2, 2), n_res=2, attn_res=(16,), dropout=0.1, res=64):
        super().__init__()
        self.n_labels, emb_dim = n_labels, 4 * ch
        self.temb = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.ch = ch
        if n_labels:
            self.yemb = nn.Linear(n_labels, emb_dim, bias=False)
            nn.init.zeros_(self.yemb.weight)
        self.inp = nn.Conv2d(3, ch, 3, padding=1)
        self.down, chans, c, r = nn.ModuleList(), [ch], ch, res
        for i, m in enumerate(mult):
            for _ in range(n_res):
                blk = nn.ModuleList([ResBlock(c, ch * m, emb_dim, dropout)])
                c = ch * m
                if r in attn_res:
                    blk.append(Attention(c))
                self.down.append(blk)
                chans.append(c)
            if i < len(mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                chans.append(c)
                r //= 2
        self.mid = nn.ModuleList([ResBlock(c, c, emb_dim, dropout), Attention(c), ResBlock(c, c, emb_dim, dropout)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(n_res + 1):
                blk = nn.ModuleList([ResBlock(c + chans.pop(), ch * m, emb_dim, dropout)])
                c = ch * m
                if r in attn_res:
                    blk.append(Attention(c))
                self.up.append(blk)
            if i > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"),
                                              nn.Conv2d(c, c, 3, padding=1)]))
                r *= 2
        self.out = nn.Sequential(norm(c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, y=None):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        emb = self.temb(timestep_embedding(t, self.ch))
        if self.n_labels:
            if y.dim() == 1:
                y = y.expand(x.shape[0], -1)
            emb = emb + self.yemb(y.to(emb.dtype))
        h, hs = self.inp(x), []
        hs.append(h)
        for blk in self.down:
            h = self._run(blk, h, emb)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = torch.cat([h, hs.pop()], 1)
            h = self._run(blk, h, emb)
        return self.out(h)

    @staticmethod
    def _run(blk, h, emb):
        for layer in blk:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return h
