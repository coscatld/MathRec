"""Optional educational CUDA/Triton implementation; importing CPU code needs none of this."""
import torch
import triton
import triton.language as tl

MAX_WIDTH = 65536
MAX_ABS = 1.0e30  # Prevent FP32 subtraction overflow; not an accuracy guarantee.


@triton.jit
def _kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)  # Avoid 32-bit row-offset overflow.
    lanes = tl.arange(0, BLOCK)
    m = tl.full((), -float('inf'), tl.float32)
    ell = tl.full((), 0.0, tl.float32)
    # N>0, so each visited block has at least one valid element.
    for start in range(0, N, BLOCK):
        offsets = start + lanes
        x = tl.load(X + row * N + offsets, offsets < N, other=-float('inf')).to(tl.float32)
        bm = tl.max(x, 0)
        new_m = tl.maximum(m, bm)
        # First block: exp(-inf-finite)=0; never evaluates -inf-(-inf).
        ell = ell * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), 0)
        m = new_m
    for start in range(0, N, BLOCK):
        offsets = start + lanes
        x = tl.load(X + row * N + offsets, offsets < N, other=-float('inf')).to(tl.float32)
        p = tl.exp(x - m) / ell
        tl.store(Y + row * N + offsets, p, offsets < N)


def validate(x, block_size=256):
    if not isinstance(x, torch.Tensor):
        raise TypeError('x must be a torch.Tensor')
    if x.ndim != 2 or not (1 <= x.shape[1] <= MAX_WIDTH):
        raise ValueError('expected [M,N], M>=0 and 1<=N<=65536')
    if x.device.type != 'cuda' or torch.version.hip is not None:
        raise ValueError('requires NVIDIA CUDA tensor (ROCm is outside this lesson)')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('supported dtypes: float16, bfloat16, float32')
    if not x.is_contiguous():
        raise ValueError('contiguous input required; no implicit copy')
    if x.requires_grad:
        raise ValueError('forward only; requires_grad must be False')
    if type(block_size) is not int or block_size not in (128, 256, 512, 1024):
        raise ValueError('block_size must be 128, 256, 512 or 1024')
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError('finite inputs required')
    if x.numel() and float(x.abs().max().item()) > MAX_ABS:
        raise ValueError('absolute input value must be <=1e30')


def _launch_validated(x, out, block_size):
    """Private benchmark path: caller already validated input/output shape and storage."""
    if x.shape[0]:
        with torch.cuda.device(x.device):
            _kernel[(x.shape[0],)](x, out, x.shape[1], block_size, num_warps=4)
    return out


def softmax(x, block_size=256):
    """Checked public API; validation synchronizes the GPU and scans the input."""
    validate(x, block_size)
    out = torch.empty_like(x)
    return _launch_validated(x, out, block_size)
