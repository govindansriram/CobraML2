import torch                                                                                                                                                                                                                                                                                      
import cobraml
import pytest

def vanilla_mha(q, k, v, causal=True):                                                                                                                                                                                                                                                            
    """Naive MHA: Q @ K.T @ V with no optimizations"""                                                                                                                                                                                                                                            
    _, N, _, d = q.shape                                                                                                                                                                                                                                                                          
    scale = d ** -0.5                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                
    # [B, N, H, d] -> [B, H, N, d]                                                                                                                                                                                                                                                                
    q = q.transpose(1, 2)                                                                                                                                                                                                                                                                         
    k = k.transpose(1, 2)                                                                                                                                                                                                                                                                         
    v = v.transpose(1, 2)                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                
    # Attention scores [B, H, N, N]                                                                                                                                                                                                                                                               
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                
    if causal:                                                                                                                                                                                                                                                                                    
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)                                                                                                                                                                                                        
        attn = attn.masked_fill(mask, float('-inf'))                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                
    attn = torch.softmax(attn, dim=-1)                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                
    out = torch.matmul(attn, v)                                                                                                                                                                                                                                                                   
    return out.transpose(1, 2)  # back to [B, N, H, d]


@pytest.mark.parametrize("B,N,H,d,causal", [
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
])
def test_fmha_fp32(benchmark, B, N, H, d, causal):
    iterations = 1

    # Initial Tensors                                                                                                                                                                                                                                                                                       
    q = torch.randn(B, N, H, d, device='cuda', dtype=torch.float32)                                                                                                                                                                                                                                   
    k = torch.randn(B, N, H, d, device='cuda', dtype=torch.float32)                                                                                                                                                                                                                                   
    v = torch.randn(B, N, H, d, device='cuda', dtype=torch.float32)

    if benchmark:
        iterations = 100

        # Warmup                                                                                                                                                                                                                                                                                          
        for _ in range(10):                                                                                                                                                                                                                                                                               
            _ = cobraml.fmha(q, k, v, causal=causal)                                                                                                                                                                                                                                                        
            _ = vanilla_mha(q, k, v, causal=causal)                                                                                                                                                                                                                                                         
        torch.cuda.synchronize()                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                    
    start = torch.cuda.Event(enable_timing=True)                                                                                                                                                                                                                                                      
    end = torch.cuda.Event(enable_timing=True)                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                    
    # CobraML

    cobra_total = 0
    vanilla_total = 0                                                                                                                                                                                                                                                                                
    for _ in range(iterations):
        q = q.normal_()                                                                                                                                                                                                                                   
        k = k.normal_()                                                                                                                                                                                                                                 
        v = v.normal_()

        torch.cuda.synchronize()

        start.record()                                                                                                                                                                                                                                                                            
        out_cobra = cobraml.fmha(q, k, v, causal=causal)                                                                                                                                                                                                                                                        
        end.record()                                                                                                                                                                                                                                                                                      
        torch.cuda.synchronize()                                                                                                                                                                                                                                                                          
        cobra_total += start.elapsed_time(end)

        start.record()
        out_vanilla = vanilla_mha(q, k, v, causal=causal)
        end.record()                                                                                                                                                                                                                                                                                      
        torch.cuda.synchronize()
        vanilla_total += start.elapsed_time(end)

    cobra_ms = cobra_total / iterations
    vanilla_ms = vanilla_total / iterations

    assert torch.allclose(out_cobra, out_vanilla, atol=1e-3, rtol=1e-3)

    if benchmark:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        print(f"CobraML:  {cobra_ms:.3f} ms")                                                                                                                                                                                                                                                             
        print(f"Vanilla:  {vanilla_ms:.3f} ms")                                                                                                                                                                                                                                                           
        print(f"Speedup:  {vanilla_ms/cobra_ms:.2f}x")                                                                                                                                                                                                                                           
                                                    