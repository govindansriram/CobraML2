from __future__ import annotations
import torch
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cobraml.models.config import ModelConfig


class MultiHeadAttention:
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal=True
    ) -> torch.Tensor:
        """
        Naive MHA: Q @ K.T @ V with no optimizations
        """

        _, N, _, d = q.shape
        scale = d**-0.5

        # [B, N, H, d] -> [B, H, N, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores [B, H, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal:
            mask = torch.triu(
                torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return out.transpose(1, 2)  # back to [B, N, H, d]

    def post_process(out_tensor: torch.Tensor):
        return out_tensor.contiguous()


class FusedMultiHeadAttention(MultiHeadAttention):
    """
    Custom Optimized Multi head Attention
    """

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal=True):
        return torch.ops.cobraml.fmha(q, k, v, causal)

    def post_process(self, out_tensor):
        return out_tensor


class AttentionLayer(nn.Module):
    def __init__(self, model_config: ModelConfig):

        self._num_heads = model_config.num_heads
        self._head_dim = model_config.head_dim
        self._embed_dim = model_config.embedding_dim

        self._causal = True

        # creates QKV projections by multiplying by weights
        # [B, N, D] @ [D * 3, D].T -> [B, N, D * 3] (weight is stored transposed)
        self._fc1 = nn.Linear(self._embed_dim, 3 * self._embed_dim)

        # applied to output of attention
        # [B, N, D] @ [D, D].T -> [B, N, D]
        self._fc2 = nn.Linear(self._embed_dim, self._embed_dim)

        self._attn_mechanism = (
            MultiHeadAttention()
            if model_config.use_naive_attention
            else FusedMultiHeadAttention()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projection = self._fc1(hidden_states)
        q_proj, k_proj, v_proj = projection.split(self._embed_dim, dim=2)
        new_shape = (*q_proj.shape[:-1], self._num_heads, self._head_dim)

        q_proj = q_proj.view(new_shape)
        k_proj = k_proj.view(new_shape)
        v_proj = v_proj.view(new_shape)

        out_tensor = self._attn_mechanism.post_process(
            self._attn_mechanism(q_proj, k_proj, v_proj, causal=self._causal)
        )

        out_view = out_tensor.view(*out_tensor.shape[:2], self._embed_dim)

        return self._fc2(out_view)
