import torch
import math
from cobraml.layers import (
    MultiHeadAttention,
    FusedMultiHeadAttention,
    build_cu_seqlens,
    build_cu_tiles_q,
)
import pytest

B_r = 64


# === Uniform batch: fused vs torch, same sizes, benchmarked ===

@pytest.mark.parametrize(
    "B,N,H,d",
    [
        (4, 512, 16, 64),
        (56, 490, 2, 64),
        (8, 64, 16, 64),
        (8, 59, 16, 64),
        (1, 2048, 16, 64),
        (2, 2048, 16, 64),
        (1, 3722, 16, 64),
        (1, 2000, 16, 64),
    ],
)
def test_fmha_uniform(benchmark, B, N, H, d):
    fmha = FusedMultiHeadAttention()
    mha = MultiHeadAttention()

    hd = H * d
    device = torch.device("cuda")

    seq_lens = [N] * B
    total_tokens = B * N

    cu_seqlens_q = build_cu_seqlens(seq_lens, device)
    cu_seqlens_kv = cu_seqlens_q
    cu_tiles_q, total_tiles = build_cu_tiles_q(seq_lens, B_r, device)

    Q = torch.randn(total_tokens, H, d, device=device, dtype=torch.float32)
    K = torch.randn(total_tokens, H, d, device=device, dtype=torch.float32)
    V = torch.randn(total_tokens, H, d, device=device, dtype=torch.float32)

    print(f"\nB={B} N={N} H={H} d={d} Q.stride={Q.stride()} K.stride={K.stride()}")

    out_fmha = fmha(Q, K, V, cu_seqlens_q, cu_seqlens_kv, cu_tiles_q, total_tiles)
    torch.cuda.synchronize()

    # torch reference: reshape to (B, N, H, d) for batched MHA
    q_batched = Q.view(B, N, H, d)
    k_batched = K.view(B, N, H, d)
    v_batched = V.view(B, N, H, d)
    out_ref = mha(q_batched, k_batched, v_batched, causal=True, start_pos=0)
    out_ref = out_ref.reshape(total_tokens, H, d)

    assert torch.allclose(out_fmha, out_ref, atol=1e-3, rtol=1e-3), (
        f"Max diff: {(out_fmha - out_ref).abs().max().item()}"
    )

    if benchmark:
        iterations = 100
        for _ in range(10):
            fmha(Q, K, V, cu_seqlens_q, cu_seqlens_kv, cu_tiles_q, total_tiles)
            mha(q_batched, k_batched, v_batched, causal=True, start_pos=0)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        cobra_total = 0
        vanilla_total = 0
        for _ in range(iterations):
            start.record()
            fmha(Q, K, V, cu_seqlens_q, cu_seqlens_kv, cu_tiles_q, total_tiles)
            end.record()
            torch.cuda.synchronize()
            cobra_total += start.elapsed_time(end)

            start.record()
            mha(q_batched, k_batched, v_batched, causal=True, start_pos=0)
            end.record()
            torch.cuda.synchronize()
            vanilla_total += start.elapsed_time(end)

        cobra_ms = cobra_total / iterations
        vanilla_ms = vanilla_total / iterations
        print(f"CobraML:  {cobra_ms:.3f} ms")
        print(f"Vanilla:  {vanilla_ms:.3f} ms")
        print(f"Speedup:  {vanilla_ms / cobra_ms:.2f}x")


# === Ragged correctness: fused ragged batch vs torch per-element ===

def check_ragged(batch_entries, H, d):
    fmha = FusedMultiHeadAttention()
    mha = MultiHeadAttention()

    device = torch.device("cuda")

    seq_lens_q = [e[0] for e in batch_entries]
    seq_lens_kv = [e[1] for e in batch_entries]

    total_q = sum(seq_lens_q)
    total_kv = sum(seq_lens_kv)

    cu_seqlens_q = build_cu_seqlens(seq_lens_q, device)
    cu_seqlens_kv = build_cu_seqlens(seq_lens_kv, device)
    cu_tiles_q, total_tiles = build_cu_tiles_q(seq_lens_q, B_r, device)

    Q = torch.randn(total_q, H, d, device=device, dtype=torch.float32)
    K = torch.randn(total_kv, H, d, device=device, dtype=torch.float32)
    V = torch.randn(total_kv, H, d, device=device, dtype=torch.float32)

    out_fmha = fmha(Q, K, V, cu_seqlens_q, cu_seqlens_kv, cu_tiles_q, total_tiles)
    torch.cuda.synchronize()

    # per-element reference
    out_ref = torch.zeros(total_q, H, d, device=device, dtype=torch.float32)
    for b in range(len(batch_entries)):
        N_q, N_kv = batch_entries[b]
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        kv_start = int(cu_seqlens_kv[b].item())
        kv_end = int(cu_seqlens_kv[b + 1].item())

        start_pos = N_kv - N_q

        q_b = Q[q_start:q_end].unsqueeze(0)
        k_b = K[kv_start:kv_end].unsqueeze(0)
        v_b = V[kv_start:kv_end].unsqueeze(0)

        ref_b = mha(q_b, k_b, v_b, causal=True, start_pos=start_pos)
        out_ref[q_start:q_end] = ref_b.squeeze(0)

    assert torch.allclose(out_fmha, out_ref, atol=1e-3, rtol=1e-3), (
        f"Max diff: {(out_fmha - out_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("batch_entries,H,d", [
    # mixed prefill + decode
    ([(256, 256), (64, 512), (128, 128), (64, 300)], 16, 64),
    # decode only, varying cache
    ([(64, 100), (64, 256), (64, 67), (64, 513)], 16, 64),
    # prefill varying lengths
    ([(64, 64), (128, 128), (256, 256), (192, 192)], 2, 64),
    # single decode, large cache
    ([(64, 1000)], 16, 64),
    # unpadded Q (non-multiple of B_r)
    ([(50, 50), (100, 100)], 16, 64),
    # mixed with single-token decode
    ([(50, 50), (1, 512), (13, 200)], 16, 64),
    # single-token decode
    ([(1, 1), (1, 64), (1, 500)], 16, 64),
])
def test_fmha_ragged(batch_entries, H, d):
    check_ragged(batch_entries, H, d)


# === Contiguous QKV buffer ===

@pytest.mark.parametrize("batch_entries,H,d", [
    ([(512, 512)] * 4, 16, 64),
    ([(256, 256), (64, 512), (128, 128), (64, 300)], 16, 64),
])
def test_fmha_contiguous(batch_entries, H, d):
    fmha = FusedMultiHeadAttention()
    mha = MultiHeadAttention()

    hd = H * d
    device = torch.device("cuda")

    seq_lens_q = [e[0] for e in batch_entries]
    seq_lens_kv = [e[1] for e in batch_entries]

    total_q = sum(seq_lens_q)
    total_kv = sum(seq_lens_kv)
    total_tokens = max(total_q, total_kv)

    cu_seqlens_q = build_cu_seqlens(seq_lens_q, device)
    cu_seqlens_kv = build_cu_seqlens(seq_lens_kv, device)
    cu_tiles_q, total_tiles = build_cu_tiles_q(seq_lens_q, B_r, device)

    # interleaved QKV
    qkv = torch.randn(total_tokens, 3 * hd, device=device, dtype=torch.float32)
    Q = qkv[:total_q, :hd].view(total_q, H, d)
    K = qkv[:total_kv, hd:2*hd].view(total_kv, H, d)
    V = qkv[:total_kv, 2*hd:].view(total_kv, H, d)

    out_fmha = fmha(Q, K, V, cu_seqlens_q, cu_seqlens_kv, cu_tiles_q, total_tiles)
    torch.cuda.synchronize()

    out_ref = torch.zeros(total_q, H, d, device=device, dtype=torch.float32)
    for b in range(len(batch_entries)):
        N_q, N_kv = batch_entries[b]
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        kv_start = int(cu_seqlens_kv[b].item())
        kv_end = int(cu_seqlens_kv[b + 1].item())

        start_pos = N_kv - N_q

        q_b = Q[q_start:q_end].unsqueeze(0)
        k_b = K[kv_start:kv_end].unsqueeze(0)
        v_b = V[kv_start:kv_end].unsqueeze(0)

        ref_b = mha(q_b, k_b, v_b, causal=True, start_pos=start_pos)
        out_ref[q_start:q_end] = ref_b.squeeze(0)

    assert torch.allclose(out_fmha, out_ref, atol=1e-3, rtol=1e-3), (
        f"contiguous: Max diff: {(out_fmha - out_ref).abs().max().item()}"
    )
