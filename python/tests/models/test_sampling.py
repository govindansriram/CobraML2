import torch
import pytest

from cobraml.models import sample_next_token


def test_sample_next_token_greedy_matches_argmax():
    logits = torch.tensor([[1.0, 3.0, 2.0], [9.0, -1.0, 0.0]])
    out = sample_next_token(logits, do_sample=False)
    expected = torch.argmax(logits, dim=-1, keepdim=True)
    assert out.equal(expected)


def test_sample_next_token_top_k_one_is_deterministic():
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    out = sample_next_token(logits, do_sample=True, top_k=1)
    assert out.item() == 1


def test_sample_next_token_top_p_filters_tail_tokens():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 1.0, 0.0]])
    out = sample_next_token(logits, do_sample=True, top_p=0.5)
    assert out.item() == 0


def test_sample_next_token_top_p_keeps_boundary_token():
    # softmax([2,1,0]) ~= [0.665, 0.245, 0.090]
    # top_p=0.8 should keep first two tokens (indices 0 and 1).
    torch.manual_seed(0)
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    out = sample_next_token(logits, do_sample=True, top_p=0.8)
    assert out.item() in {0, 1}


@pytest.mark.parametrize(
    "temperature,top_k,top_p",
    [
        (0.0, None, 1.0),
        (-1.0, None, 1.0),
        (1.0, -1, 1.0),
        (1.0, None, 0.0),
        (1.0, None, 1.5),
    ],
)
def test_sample_next_token_invalid_args(temperature, top_k, top_p):
    logits = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError):
        sample_next_token(
            logits,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
