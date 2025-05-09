import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured
import time

# def benchmark_linear(batch_size=1024,
#                      in_features=4096,
#                      out_features=4096,
#                      n_warmup=10,
#                      n_iters=10000):
#     # pick your device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Running on {device}")

#     linear = nn.Linear(64, 384)
#     linear.half()
#     linear.to(device = device)
#     weight_t = linear.weight.T
#     bias = linear.bias
    
#     a = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).half().cuda()
#     b = torch.rand(64, 64).half().cuda()
#     c = torch.mm(a, b)
#     a_sparse = to_sparse_semi_structured(a)
#     torch.allclose(c, torch.mm(a_sparse, b))
    
#     # input_ = torch.rand(64, 64).half().cuda()
#     # mask = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).cuda().bool()
#     # linear_2 = nn.Linear(64, 64).half().cuda()
#     # linear_2.weight = nn.Parameter(to_sparse_semi_structured(linear_2.weight.masked_fill(~mask, 0)))
    
#     # # F.linear(input_, linear_2.weight, None)
#     # torch.mm()
#     # # linear_2(input_)
    
#     activations_dense = torch.Tensor([0, 0, 1, 1]).tile((64, 16)).half().cuda()
#     activations_sparse = to_sparse_semi_structured(activations_dense)
#     activations_sparse_T = to_sparse_semi_structured(torch.Tensor([0, 0, 1, 1]).tile((64, 16)).half().cuda())
    
#     print(f'Activation shape: {activations_dense.shape}')
    
#     # warm-up
#     for _ in range(n_warmup):
#         _ = F.linear(weight_t.T, activations_dense.T, None).T
#         _ = F.linear(weight_t.T, activations_sparse, None).T
#         # _ = torch.mm(activations_sparse, weight_t) + bias
    
#     # benchmark dense
#     torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     for _ in range(n_iters):
#         _ = F.linear(weight_t.T, activations_dense.T, None)
#         # .T + bias
#     torch.cuda.synchronize()
#     dense_ms = (time.perf_counter() - t0) * 1000 / n_iters
    
#     # benchmark sparse
#     torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     for _ in range(n_iters):
#         _ = F.linear(weight_t.T, activations_sparse, None)
#         # .T + bias
#     torch.cuda.synchronize()
#     sparse_ms = (time.perf_counter() - t0) * 1000 / n_iters
    
#     print(f"Avg dense:  {dense_ms:.3f} ms per matmul")
#     print(f"Avg sparse: {sparse_ms:.3f} ms per matmul")
    
# if __name__ == "__main__":
#     benchmark_linear()

def benchmark_linear(batch_size=1024,
                     in_features=4096,
                     out_features=4096,
                     n_warmup=100,
                     n_iters=1000):
    in_f = in_features
    out_f = out_features
    input_ = torch.rand(64, 4096).half().cuda()
    
    # create two identical Linear modules
    dense_lin  = nn.Linear( in_f, out_f ).half().cuda()
    sparse_lin = nn.Linear( in_f, out_f ).half().cuda()
    # mask-and-sparsify its weight
    mask = torch.Tensor([0, 0, 1, 1]).tile((out_f, in_f//4)).cuda().bool()  # your 2:4 mask of shape [out_f, in_f]
    w = sparse_lin.weight.data.masked_fill(~mask, 0)
    sparse_lin.weight = nn.Parameter(to_sparse_semi_structured(w))
    # force cutlass if you’re on PyTorch ≥2.1
    from torch.sparse import SparseSemiStructuredTensor
    SparseSemiStructuredTensor._FORCE_CUTLASS = True

    # warm-up
    for _ in range(n_warmup):
        _ = dense_lin(input_)
        _ = sparse_lin(input_)

    # time them
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iters):
        _ = dense_lin(input_)
    t1.record(); torch.cuda.synchronize()
    dense_ms = t0.elapsed_time(t1) / n_iters

    t0.record()
    for _ in range(n_iters):
        _ = sparse_lin(input_)
    t1.record(); torch.cuda.synchronize()
    sparse_ms = t0.elapsed_time(t1) / n_iters

    print(f"dense {dense_ms} ms  |  sparse {sparse_ms} ms | ratio {dense_ms/sparse_ms}")

if __name__ == "__main__":
    benchmark_linear()