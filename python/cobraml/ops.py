import torch


def fmha(q, k, v, causal=False):
    """
    Flash Multi-Head Attention.

    Args:
        q: Query tensor [B, N, H, d]
        k: Key tensor [B, N, H, d]
        v: Value tensor [B, N, H, d]
        causal: Apply causal masking

    Returns:
        Output tensor [B, N, H, d]
    """
    return torch.ops.cobraml.fmha(q, k, v, causal)
