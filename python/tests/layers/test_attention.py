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


@pytest.mark.parametrize(
    "B,N,H,d,start_pos",
    [
        # single-token decode at various positions
        (1, 128, 16, 64, 0),
        (1, 128, 16, 64, 1),
        (1, 128, 16, 64, 63),
        (1, 128, 16, 64, 64),
        (1, 128, 16, 64, 127),
        # multi-batch
        (4, 256, 16, 64, 100),
        (4, 256, 2, 64, 255),
        # uneven sequence lengths
        (2, 59, 16, 64, 30),
        (2, 490, 2, 64, 200),
    ],
)
def test_fmha_start_pos(B, N, H, d, start_pos):
    """
    Verify start_pos by comparing single-token decode against the
    corresponding row of a full causal prefill.

    The fmha binding uses qkv_contigous_buffer=true, so Q/K/V must have
    the interleaved stride layout (seq stride = 3*H*d). We create decode
    tensors from their own QKV buffers to get the correct strides.
    """
    fmha = FusedMultiHeadAttention()
    mha = MultiHeadAttention()

    hd = H * d
    N_kv = start_pos + 1

    # full prefill QKV buffer
    qkv = torch.randn(B, N, 3 * hd, device="cuda", dtype=torch.float32)
    q, k, v = qkv.split(hd, dim=2)
    q = q.view(B, N, H, d)
    k = k.view(B, N, H, d)
    v = v.view(B, N, H, d)

    # full causal prefill (start_pos=0)
    full_out = fmha(q, k, v, causal=True, start_pos=0)
    expected = full_out[:, start_pos : start_pos + 1, :, :]

    # build decode Q in its own QKV-strided buffer so batch stride = 3*hd*1
    q_decode_buf = torch.zeros(B, 1, 3 * hd, device="cuda", dtype=torch.float32)
    q_decode_buf[:, :, :hd] = (
        q[:, start_pos : start_pos + 1, :, :].contiguous().view(B, 1, hd)
    )
    q_decode = q_decode_buf[:, :, :hd].view(B, 1, H, d)

    # build decode K/V in their own QKV-strided buffer so batch stride = 3*hd*N_kv
    kv_decode_buf = torch.zeros(B, N_kv, 3 * hd, device="cuda", dtype=torch.float32)
    kv_decode_buf[:, :, hd : 2 * hd] = k[:, :N_kv, :, :].contiguous().view(B, N_kv, hd)
    kv_decode_buf[:, :, 2 * hd :] = v[:, :N_kv, :, :].contiguous().view(B, N_kv, hd)
    k_decode = kv_decode_buf[:, :, hd : 2 * hd].view(B, N_kv, H, d)
    v_decode = kv_decode_buf[:, :, 2 * hd :].view(B, N_kv, H, d)

    decode_out_fmha = fmha(
        q_decode, k_decode, v_decode, causal=True, start_pos=start_pos
    )

    # vanilla MHA works with any strides, use contiguous tensors directly
    q_mha = q[:, start_pos : start_pos + 1, :, :].contiguous()
    k_mha = k[:, :N_kv, :, :].contiguous()
    v_mha = v[:, :N_kv, :, :].contiguous()
    decode_out_mha = mha(q_mha, k_mha, v_mha, causal=True, start_pos=start_pos)

    assert torch.allclose(decode_out_fmha, expected, atol=1e-3, rtol=1e-3), (
        "fmha decode with start_pos does not match full prefill"
    )
    assert torch.allclose(decode_out_mha, expected, atol=1e-3, rtol=1e-3), (
        "mha decode with start_pos does not match full prefill"
    )
    assert torch.allclose(decode_out_fmha, decode_out_mha, atol=1e-3, rtol=1e-3), (
        "fmha and mha decode outputs do not match"
    )
