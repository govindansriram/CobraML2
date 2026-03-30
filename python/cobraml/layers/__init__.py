import torch
from pathlib import Path

# Load the compiled extension to register torch.ops.cobraml
_lib_path = Path(__file__).parent.parent.glob("_C*.so")
for lib in _lib_path:
    torch.ops.load_library(lib)
    break

from .attention import MultiHeadAttention, FusedMultiHeadAttention, AttentionLayer, build_cu_seqlens, build_cu_tiles_q  # noqa: E402
from .mlp import GPT2MLP  # noqa: E402

__all__ = ["MultiHeadAttention", "FusedMultiHeadAttention", "AttentionLayer", "GPT2MLP", "build_cu_seqlens", "build_cu_tiles_q"]
