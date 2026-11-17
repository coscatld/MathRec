"""Prevalidated, preallocated internal launch; caller owns all contracts."""
import torch
import triton
import triton.language as tl

@triton.jit
def _row_softmax(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    offset = row * N + col
    valid = col < N
    logits = tl.load(X + offset, mask=valid, other=-float('inf')).to(tl.float32)
    peak = tl.max(logits, axis=0)
    weights = tl.exp(logits - peak)
    total = tl.sum(weights, axis=0)
    probs = weights / total
    tl.store(Y + offset, probs, mask=valid)

def launch(x, out):
    """Private path: finite contiguous CUDA input; disjoint matching output."""
    if x.shape[0] == 0:
        return
    n = x.shape[1]
    with torch.cuda.device(x.device):
        _row_softmax[(x.shape[0],)](x, out, n, triton.next_power_of_2(n), num_warps=4)
