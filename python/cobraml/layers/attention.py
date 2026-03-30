from __future__ import annotations
import math
import torch
from torch import nn
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cobraml.models.config import ModelConfig


def build_cu_seqlens(seq_lens: list[int], device: torch.device) -> torch.Tensor:
    """Build prefix-sum array from a list of sequence lengths."""
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seq_lens):
        cu[i + 1] = cu[i] + l
    return cu


def build_cu_tiles_q(seq_lens_q: list[int], B_r: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """Build prefix-sum of tile counts and return (cu_tiles_q, total_tiles)."""
    cu = torch.zeros(len(seq_lens_q) + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seq_lens_q):
        cu[i + 1] = cu[i] + math.ceil(l / B_r)
    return cu, int(cu[-1].item())


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


class FusedMultiHeadAttention:
    """
    Custom Optimized Multi head Attention with ragged batching.
    Q/K/V are (total_tokens, H, d). Requires cu_seqlens and cu_tiles_q.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        cu_tiles_q: torch.Tensor,
        total_tiles: int,
    ):
        return torch.ops.cobraml.fmha(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, cu_tiles_q, total_tiles
        )

    def post_process(self, out_tensor):
        return out_tensor


class AttentionLayer(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self._num_heads = model_config.num_heads
        self._head_dim = model_config.head_dim
        self._embed_dim = model_config.embedding_dim
        self._B_r = 32 if self._head_dim >= 128 else 64

        self._causal = True

        self.c_attn = nn.Linear(self._embed_dim, 3 * self._embed_dim)
        self.c_proj = nn.Linear(self._embed_dim, self._embed_dim)

        self._attn_mechanism = (
            MultiHeadAttention()
            if model_config.use_naive_attention
            else FusedMultiHeadAttention()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        projection = self.c_attn(hidden_states)
        q_proj, k_proj, v_proj = projection.split(self._embed_dim, dim=-1)

        new_shape = (*q_proj.shape[:-1], self._num_heads, self._head_dim)
        q_proj = q_proj.view(new_shape)
        k_proj = k_proj.view(new_shape)
        v_proj = v_proj.view(new_shape)

        if isinstance(self._attn_mechanism, FusedMultiHeadAttention):
            assert cu_seqlens_q is not None and cu_seqlens_kv is not None, \
                "FusedMultiHeadAttention requires cu_seqlens_q and cu_seqlens_kv"
            seq_lens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
            cu_tiles_q, total_tiles = build_cu_tiles_q(
                seq_lens_q, self._B_r, hidden_states.device,
            )
            out_tensor = self._attn_mechanism(
                q_proj, k_proj, v_proj,
                cu_seqlens_q, cu_seqlens_kv, cu_tiles_q, total_tiles,
            )
        else:
            out_tensor = self._attn_mechanism(
                q_proj, k_proj, v_proj, causal=self._causal,
            )

        out_tensor = self._attn_mechanism.post_process(out_tensor)
        out_view = out_tensor.view(*out_tensor.shape[:-2], self._embed_dim)
        return self.c_proj(out_view)
