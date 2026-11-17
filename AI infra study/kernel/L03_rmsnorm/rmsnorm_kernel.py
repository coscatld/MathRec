"""Import only on GPU path; tl is module-global for Triton JIT resolution."""
import triton
import triton.language as tl


@triton.jit
def kernel(X, W, Y, N: tl.constexpr, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < N
    offset = row * N + col
    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + col, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / N
    inv_rms = tl.rsqrt(mean_square + EPS)
    y = (x * inv_rms) * w
    tl.store(Y + offset, y, mask=mask)
