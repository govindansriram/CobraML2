from __future__ import annotations
import torch
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cobraml.models.config import ModelConfig


class MultiHeadAttention:
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal=True,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Naive MHA: Q @ K.T @ V with no optimizations
        """

        _, N_q, _, d = q.shape
        N_kv = k.size(1)
        scale = d**-0.5

        # [B, N, H, d] -> [B, H, N, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores [B, H, N_q, N_kv]
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal:
            q_positions = torch.arange(N_q, device=q.device, dtype=torch.int64)
            kv_positions = torch.arange(N_kv, device=q.device, dtype=torch.int64)
            mask = kv_positions.view(1, N_kv) > (start_pos + q_positions.view(N_q, 1))
            attn = attn.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return out.transpose(1, 2)  # back to [B, N_q, H, d]

    def post_process(self, out_tensor: torch.Tensor):
        return out_tensor.contiguous()


class FusedMultiHeadAttention(MultiHeadAttention):
    """
    Custom Optimized Multi head Attention
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal=True,
        start_pos: int = 0,
    ):
        # Keep compatibility with both the legacy 4-argument op schema and the
        # newer schema that accepts an explicit start position.
        if start_pos == 0:
            return torch.ops.cobraml.fmha(q, k, v, causal)
        return torch.ops.cobraml.fmha(q, k, v, causal, start_pos)

    def post_process(self, out_tensor):
        return out_tensor


class AttentionLayer(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self._num_heads = model_config.num_heads
        self._head_dim = model_config.head_dim
        self._embed_dim = model_config.embedding_dim

        self._causal = True

        # creates QKV projections by multiplying by weights
        # [B, N, D] @ [D * 3, D].T -> [B, N, D * 3] (weight is stored transposed)
        self.c_attn = nn.Linear(self._embed_dim, 3 * self._embed_dim)

        # applied to output of attention
        # [B, N, D] @ [D, D].T -> [B, N, D]
        self.c_proj = nn.Linear(self._embed_dim, self._embed_dim)

        self._attn_mechanism = (
            MultiHeadAttention()
            if model_config.use_naive_attention
            else FusedMultiHeadAttention()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projection = self.c_attn(hidden_states)
        q_proj, k_proj, v_proj = projection.split(self._embed_dim, dim=2)
        new_shape = (*q_proj.shape[:-1], self._num_heads, self._head_dim)

        q_proj = q_proj.view(new_shape)
        k_proj = k_proj.view(new_shape)
        v_proj = v_proj.view(new_shape)

        out_tensor = self._attn_mechanism.post_process(
            self._attn_mechanism(q_proj, k_proj, v_proj, causal=self._causal)
        )

        out_view = out_tensor.view(*out_tensor.shape[:2], self._embed_dim)

        return self.c_proj(out_view)
