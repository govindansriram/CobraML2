from __future__ import annotations

import torch


def _validate_sampling_args(
    do_sample: bool, temperature: float, top_k: int | None, top_p: float
) -> None:
    if not do_sample:
        return

    if temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1]")


def _apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k == 0:
        return logits

    k = min(top_k, logits.size(-1))
    topk_values = torch.topk(logits, k, dim=-1).values
    threshold = topk_values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    # Keep the first token where cumulative probability crosses top_p.
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    filtered_logits = torch.full_like(logits, float("-inf"))
    return filtered_logits.scatter(-1, sorted_indices, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Select next token IDs from logits of shape [B, vocab]."""
    _validate_sampling_args(do_sample, temperature, top_k, top_p)

    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    filtered_logits = _apply_top_k(scaled_logits, top_k)
    filtered_logits = _apply_top_p(filtered_logits, top_p)

    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
