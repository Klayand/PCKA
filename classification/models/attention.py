"""
These modules are based on the implementation of https://github.com/lucidrains/vit-pytorch.
"""
import numpy as np

import torch
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops import rearrange
from .layer import DropPath, ln2d


class FeedForward(nn.Module):

    def __init__(self, dim_in, hidden_dim, dim_out=None, *,
                 dropout=0.0,
                 f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.net = nn.Sequential(
            f(dim_in, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(hidden_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention1d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class Attention2d(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dropout=0.0, k=1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out

        self.to_q = nn.Conv2d(dim_in, inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', y=y)

        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=64, dim_mlp=1024, dropout=0.0, sd=0.0,
                 attn=Attention1d, norm=nn.LayerNorm,
                 f=nn.Linear, activation=nn.GELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out

        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)

        self.norm1 = norm(dim_in)
        self.attn = attn(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout)
        self.sd1 = DropPath(sd) if sd > 0.0 else nn.Identity()

        self.norm2 = norm(dim_out)
        self.ff = FeedForward(dim_out, dim_mlp, dim_out, dropout=dropout, f=f, activation=activation)
        self.sd2 = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, attn = self.attn(x, mask=mask)
        x = self.sd1(x) + skip
        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.sd2(x) + skip
        return x


class CyclicShift(nn.Module):

    def __init__(self, d, dims=(2, 3)):
        super().__init__()
        self.d = d
        self.dims = dims

    def forward(self, x):
        x = torch.roll(x, shifts=(self.d, self.d), dims=self.dims)
        return x


class PatchMerging(nn.Module):

    def __init__(self, in_channels, out_channels, pool):
        super().__init__()
        self.patch_merge = nn.Conv2d(in_channels, out_channels, kernel_size=pool, stride=pool)

    def forward(self, x):
        x = self.patch_merge(x)
        return x


class WindowAttention(nn.Module):

    def __init__(self, dim_in, dim_out=None, *,
                 heads=8, dim_head=32, dropout=0.0, window_size=7, shifted=False):
        super().__init__()
        self.attn = Attention1d(dim_in, dim_out,
                                heads=heads, dim_head=dim_head, dropout=dropout)
        self.window_size = window_size
        self.shifted = shifted
        self.d = window_size // 2

        self.shift = CyclicShift(-1 * self.d) if shifted else nn.Identity()
        self.backshift = CyclicShift(self.d) if shifted else nn.Identity()

        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p

        mask = torch.zeros(p ** 2, p ** 2, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]
        if self.shifted:
            mask = mask + self._upper_lower_mask(h // p, w // p, p, p, self.d, x.device)
            mask = mask + self._left_right_mask(h // p, w // p, p, p, self.d, x.device)
            mask = repeat(mask, "n h i j -> (b n) h i j", b=b)

        x = self.shift(x)
        x = rearrange(x, "b c (n1 p1) (n2 p2) -> (b n1 n2) (p1 p2) c", p1=p, p2=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, "(b n1 n2) (p1 p2) c -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p)
        x = self.backshift(x)

        return x, attn

    @staticmethod
    def _upper_lower_mask(n1, n2, i, j, d, device=None):
        m = torch.zeros(i ** 2, j ** 2, device=device)
        m[-d * i:, :-d * j] = float('-inf')
        m[:-d * i, -d * j:] = float('-inf')

        mask = torch.zeros(n1 * n2, 1, i ** 2, j ** 2, device=device)
        mask[-n2:] = mask[-n2:] + m

        return mask

    @staticmethod
    def _left_right_mask(n1, n2, i, j, d, device=None):
        m = torch.zeros(i ** 2, j ** 2, device=device)
        m = rearrange(m, '(i k) (j l) -> i k j l', i=i, j=j)
        m[:, -d:, :, :-d] = float('-inf')
        m[:, :-d, :, -d:] = float('-inf')
        m = rearrange(m, 'i k j l -> (i k) (j l)')

        mask = torch.zeros(n1 * n2, 1, i ** 2, j ** 2, device=device)
        mask[-n1 - 1::n1] += m

        return mask

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        d = i[None, :, :] - i[:, None, :]

        return d
