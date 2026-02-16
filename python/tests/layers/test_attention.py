import torch
from cobraml.layers import MultiHeadAttention, FusedMultiHeadAttention
import pytest


@pytest.mark.parametrize(
    "B,N,H,d,causal",
    [
        # even block size by sequence length
        (4, 512, 16, 64, False),
        (4, 512, 16, 64, True),
        # uneven block size (requires predication)
        (56, 490, 2, 64, False),
        (56, 490, 2, 64, True),
        # 1 block only, even
        (8, 64, 16, 64, False),
        (8, 64, 16, 64, True),
        # 1 block only, uneven
        (8, 59, 16, 64, False),
        (8, 59, 16, 64, True),
        # longer sequences (vanilla MHA allocates B*H*N*N attention matrix)
        (1, 2048, 16, 64, False),
        (1, 2048, 16, 64, True),
        (2, 2048, 16, 64, False),
        (1, 3722, 16, 64, False),
        (1, 3722, 16, 64, True),
        # longer sequences, uneven
        (1, 2000, 16, 64, False),
    ],
)
def test_fmha_fp32(benchmark, B, N, H, d, causal):
    iterations = 1

    fmha = FusedMultiHeadAttention()
    mha = MultiHeadAttention()

    hd = H * d
    qkv = torch.randn(B, N, 3 * hd, device="cuda", dtype=torch.float32)
    q, k, v = qkv.split(hd, dim=2)
    q = q.view(B, N, H, d)
    k = k.view(B, N, H, d)
    v = v.view(B, N, H, d)

    if benchmark:
        iterations = 100

        # Warmup
        for _ in range(10):
            _ = fmha(q, k, v, causal=causal)
            _ = mha(q, k, v, causal=causal)
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # CobraML

    cobra_total = 0
    vanilla_total = 0
    for _ in range(iterations):
        qkv.normal_()

        torch.cuda.synchronize()

        start.record()
        out_cobra = fmha(q, k, v, causal=causal)
        end.record()
        torch.cuda.synchronize()
        cobra_total += start.elapsed_time(end)

        start.record()
        out_vanilla = mha(q, k, v, causal=causal)
        end.record()
        torch.cuda.synchronize()
        vanilla_total += start.elapsed_time(end)

    cobra_ms = cobra_total / iterations
    vanilla_ms = vanilla_total / iterations

    assert torch.allclose(out_cobra, out_vanilla, atol=1e-3, rtol=1e-3)

    if benchmark:
        print(f"CobraML:  {cobra_ms:.3f} ms")
        print(f"Vanilla:  {vanilla_ms:.3f} ms")
        print(f"Speedup:  {vanilla_ms / cobra_ms:.2f}x")
