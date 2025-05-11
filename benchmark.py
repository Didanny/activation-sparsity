import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.sparse import SparseSemiStructuredTensor
SparseSemiStructuredTensor._FORCE_CUTLASS = True
from torch.sparse import to_sparse_semi_structured
import time
from torchvision.models.vision_transformer import vit_h_14, vit_l_32

# from models.cifar_vit import 

def enforce_2_to_4_sparsity(x):
    orig_shape = x.shape
    assert x.shape[-1] % 4 == 0, "Hidden dim must be divisible by 4"
    x = x.view(*x.shape[:-1], -1, 4)  # (..., num_blocks, 4)
    _, idx = torch.topk(x.abs(), k=2, dim=-1, largest=False)
    mask = torch.ones_like(x)
    mask.scatter_(-1, idx, 0.0)
    x = x * mask
    return x.view(orig_shape)

def benchmark_sparse_enforcement(
    seq_len=1,
    in_features=4096,
    n_warmup=1000,
    n_iters=1000):

    activation = torch.rand(seq_len, in_features).half().cuda()
    
    # Warm-up
    for _ in range(n_warmup):
        enforce_2_to_4_sparsity(activation)
       
    # Time
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        enforce_2_to_4_sparsity(activation)
    t1.record(); torch.cuda.synchronize()
    enforce_ms = t0.elapsed_time(t1) / n_iters
    
    print(f"enforcement {enforce_ms} ms")
    
    return enforce_ms

def benchmark_attention(
    seq_len=1,
    embed_dim=4096,
    n_warmup=1000,
    n_iters=1000):
    
    mhattention = nn.MultiheadAttention(embed_dim, 8).half().cuda()
    mhattention.cuda()
    
    # Make input
    input_ = torch.rand(seq_len, 1, embed_dim).half().cuda()
    
    # Warm-up
    for _ in range(n_warmup):
        mhattention(input_, input_, input_)
    
    # Time
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        mhattention(input_, input_, input_)
    t1.record(); torch.cuda.synchronize()
    attn_ms = t0.elapsed_time(t1) / n_iters
    
    print(f"attention {attn_ms} ms") 
    
    return attn_ms   
    
def benchmark_sparse_conversion(
    seq_len=1,
    in_features=4096,
    n_warmup=1000,
    n_iters=1000):  
    
    in_f = in_features
    
    dense_act = torch.rand(seq_len, in_f).half().cuda()
    # mask = torch.Tensor([0, 0, 1, 1]).tile((seq_len, in_f//4)).cuda().bool()  # your 2:4 mask of shape [out_f, in_f]
    # dense_act = dense_act.masked_fill(~mask, 0)
    
    # Warm-up
    for _ in range(n_warmup):
        to_sparse_semi_structured(dense_act)
        
    # Time
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        to_sparse_semi_structured(dense_act)
    t1.record(); torch.cuda.synchronize()
    convert_ms = t0.elapsed_time(t1) / n_iters
    
    print(f"conversion {convert_ms} ms") 
    
    return convert_ms   

def benchmark_sparse_activations(
    seq_len=1,
    in_features=4096,
    out_features=4096,
    n_warmup=1000,
    n_iters=1000):
    
    in_f = in_features
    out_f = out_features
    
    print('Benchmarking MLP operations')
    
    # Linear module
    linear  = nn.Linear( in_f, out_f ).half().cuda()
    
    # mask-and-sparsify its weight
    dense_act = torch.rand(seq_len, in_f).half().cuda()
    mask = torch.Tensor([0, 0, 1, 1]).tile((seq_len, in_f//4)).cuda().bool()  # your 2:4 mask of shape [out_f, in_f]
    sparse_act = to_sparse_semi_structured(dense_act.masked_fill(~mask, 0))

    # warm-up
    for _ in range(n_warmup):
        torch.mm(dense_act, linear.weight.t()) + linear.bias
        torch.mm(sparse_act, linear.weight.t()) + linear.bias

    # time them
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        torch.mm(dense_act, linear.weight.t()) + linear.bias
    t1.record(); torch.cuda.synchronize()
    dense_ms = t0.elapsed_time(t1) / n_iters

    t0.record()
    for _ in range(n_iters):
        torch.mm(sparse_act, linear.weight.t()) + linear.bias
    t1.record(); torch.cuda.synchronize()
    sparse_ms = t0.elapsed_time(t1) / n_iters

    print(f"dense {dense_ms} ms  |  sparse {sparse_ms} ms | ratio {dense_ms/sparse_ms}")
    
    return dense_ms, sparse_ms

def benchmark_sparse_weights(
    batch_size=1,
    seq_len=1,
    in_features=4096,
    out_features=4096,
    n_warmup=1000,
    n_iters=1000):
    
    in_f = in_features
    out_f = out_features
    input_ = torch.rand(batch_size, seq_len, in_f).half().cuda()
    
    # create two identical Linear modules
    dense_lin  = nn.Linear( in_f, out_f ).half().cuda()
    sparse_lin = nn.Linear( in_f, out_f ).half().cuda()
    
    # mask-and-sparsify its weight
    mask = torch.Tensor([0, 0, 1, 1]).tile((out_f, in_f//4)).cuda().bool()  # your 2:4 mask of shape [out_f, in_f]
    w = sparse_lin.weight.data.masked_fill(~mask, 0)
    sparse_lin.weight = nn.Parameter(to_sparse_semi_structured(w))
    
    # warm-up
    for _ in range(n_warmup):
        torch.addmm(dense_lin.bias, input_.squeeze(0), dense_lin.weight.t())
        torch.addmm(sparse_lin.bias, input_.squeeze(0), sparse_lin.weight.t())

    # time them
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        # torch.addmm(dense_lin.bias, input_.squeeze(0), dense_lin.weight.t())
        torch.mm(dense_lin.weight, input_.squeeze(0).t())
    t1.record(); torch.cuda.synchronize()
    dense_ms = t0.elapsed_time(t1) / n_iters

    t0.record()
    for _ in range(n_iters):
        # torch.addmm(sparse_lin.bias, input_.squeeze(0), sparse_lin.weight.t())
        torch.mm(sparse_lin.weight, input_.squeeze(0).t())
    t1.record(); torch.cuda.synchronize()
    sparse_ms = t0.elapsed_time(t1) / n_iters

    print(f"dense {dense_ms} ms  |  sparse {sparse_ms} ms | ratio {dense_ms/sparse_ms}")

if __name__ == "__main__": 
    
    # ViT Large 16 defaults
    # n = 1
    # m = 4
    # bs = 1
    # s = 49
    
    # ViT Large 32 defaults
    # n = 1
    # m = 4
    # bs = 1
    # s = 196
    
    # ViT Huge 14 defaults
    # n = 1.25
    # m = 5
    # bs = 1
    # s = 256
    
    # Custom
    n = 1.25
    m = 5
    bs = 1
    s = 1024
    
    # MLP 1
    mlp1_dense_ms, mlp1_sparse_ms = benchmark_sparse_activations(seq_len=s, in_features=int(1024 * n), out_features=int(1024 * m))
    
    # MLP 2
    mlp2_dense_ms, mlp2_sparse_ms = benchmark_sparse_activations(seq_len=s, in_features=int(1024 * m), out_features=int(1024 * n))
    
    # Enforcing 2:4 sparsity
    # enforcement_ms = benchmark_sparse_enforcement(seq_len=s, in_features=int(1024 * m))
    
    # To semi-structured sparse format
    conversion_ms = 0.05 * (mlp1_sparse_ms + mlp2_sparse_ms)
    
    # Multi-head attention
    attention_ms = benchmark_attention(seq_len=s, embed_dim=int(1024 * n))
    
    # Compile
    mlp1_speedup_raw = mlp1_dense_ms / mlp1_sparse_ms
    mlp2_speedup_raw = mlp2_dense_ms / mlp2_sparse_ms
    mlp1_speedup = mlp1_dense_ms / (1.05 * mlp1_sparse_ms)
    mlp2_speedup = mlp2_dense_ms / (1.05 * mlp2_sparse_ms)
    raw_mlp_speedup = (mlp1_dense_ms + mlp2_dense_ms) / (mlp1_sparse_ms + mlp2_sparse_ms)
    
    actual_mlp_speedup = (mlp1_dense_ms + mlp2_dense_ms) / (mlp1_sparse_ms + mlp2_sparse_ms + conversion_ms)
    
    total_speedup = (mlp1_dense_ms + mlp2_dense_ms + attention_ms) / (mlp1_sparse_ms + mlp2_sparse_ms + conversion_ms + attention_ms)
    
    print('Raw Speedup')
    print(f'UP speedup {mlp1_speedup_raw} | Down speedup {mlp2_speedup_raw} | MLP speedup {raw_mlp_speedup}')
    print('Actual Speedup')
    print(f'UP speedup {mlp1_speedup} | Down speedup {mlp2_speedup} | MLP speedup {actual_mlp_speedup} | Total {total_speedup}')
    
    
    
    
    
    
    
    