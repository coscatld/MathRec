"""Imported only by the GPU launch path; globals are visible to Triton 3.1.0 JIT."""
import triton
import triton.language as tl

@triton.jit
def kernel(X, Y, ROW_STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    vals = tl.load(X + row * ROW_STRIDE + cols, cols < N, other=0.0)
    total = tl.sum(vals, axis=0)
    tl.store(Y + row, total)
