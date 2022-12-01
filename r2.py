from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

import time

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
    ):
        q = self.query(x)

        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)

        #n_batch, n_ctx, n_state = q.shape
        #scale = (n_state // self.n_head) ** -0.25
        #q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        #k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        #v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        #qk = q @ k
        #w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        #wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        return self.out(q)


