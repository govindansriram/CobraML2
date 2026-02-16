from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch import nn
from transformers.activations import ACT2FN

if TYPE_CHECKING:
    from cobraml.models.config import GPT2Config


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self._intermediate_size = config.intermediate_size
        self._embed_dim = config.embedding_dim

        self.c_fc = nn.Linear(self._embed_dim, self._intermediate_size)
        self.c_proj = nn.Linear(self._intermediate_size, self._embed_dim)
        self.act = ACT2FN[config.activation_function]

        # avoid using nn.Sequential as it makes weight loading more
        # challenging

    def forward(self, hidden_state: torch.Tensor):

        hidden_state = self.c_fc(hidden_state)
        hidden_state = self.act(hidden_state)
        return self.c_proj(hidden_state)
