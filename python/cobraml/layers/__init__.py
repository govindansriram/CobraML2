import torch
from pathlib import Path

# Load the compiled extension to register torch.ops.cobraml
_lib_path = Path(__file__).parent.parent.glob("_C*.so")
for lib in _lib_path:
    torch.ops.load_library(lib)
    break

from .attention import MultiHeadAttention, FusedMultiHeadAttention, AttentionLayer  # noqa: E402
from .mlp import GPT2MLP

__all__ = [
    "MultiHeadAttention", 
    "FusedMultiHeadAttention", 
    "AttentionLayer",
    "GPT2MLP"
]
