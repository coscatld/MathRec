"""Lesson 03: FP32 inference RMSNorm. CPU reference needs only Python."""
import math
import numbers
import struct

MAX_COLS = 4096


def validate_eps(eps):
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise TypeError('eps must be a real scalar, not bool')
    eps = float(eps)
    # Require a positive NORMAL FP32 epsilon: avoids flush-to-zero surprises.
    if not math.isfinite(eps) or not (2**-126 <= eps <= 1.0):
        raise ValueError('teaching contract: 2**-126 <= eps <= 1')
    return struct.unpack('f', struct.pack('f', eps))[0]


def reference_rows(rows, weight, eps=1e-5):
    """Python double reference; not a FP32 GPU simulation.

    Numerical domain: finite inputs and no intermediate/output overflow.
    Python double arithmetic is a diagnostic, not support for every finite double."""
    eps = validate_eps(eps)
    n = len(weight)
    if not 1 <= n <= MAX_COLS:
        raise ValueError('weight width must be 1..4096')
    if any(len(row) != n for row in rows):
        raise ValueError('each row must match weight width')
    if any(not math.isfinite(v) for v in weight) or any(
            not math.isfinite(v) for row in rows for v in row):
        raise ValueError('reference requires finite inputs')
    result = []
    for row in rows:
        inv = 1.0 / math.sqrt(math.fsum(v*v for v in row) / n + eps)
        result.append([(v * inv) * w for v, w in zip(row, weight)])
    return result


def _validate(x, weight, *, cuda, eps):
    import torch
    eps = validate_eps(eps)
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError('x and weight must be tensors')
    if x.layout != torch.strided or weight.layout != torch.strided:
        raise ValueError('dense strided tensors required')
    if x.ndim != 2 or weight.ndim != 1 or x.shape[1] != weight.shape[0]:
        raise ValueError('expected x[M,N], weight[N]')
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError('FP32 only')
    if x.device != weight.device:
        raise ValueError('devices must match')
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in (x, weight)):
        raise ValueError('ordinary contiguous tensors required')
    if x.requires_grad or weight.requires_grad:
        raise ValueError('inference only: no backward')
    if not 1 <= x.shape[1] <= MAX_COLS or x.shape[0] > 2**31-1:
        raise ValueError('requires 1 <= N <= 4096 and M <= 2**31-1')
    if cuda and x.device.type != 'cuda':
        raise ValueError('CUDA required for Triton path')
    return eps


def torch_reference(x, weight, eps=1e-5):
    """Unfused semantic baseline; accepts CPU or CUDA, same inference contract."""
    import torch
    eps = _validate(x, weight, cuda=False, eps=eps)
    return (x * torch.rsqrt((x*x).mean(dim=1, keepdim=True) + eps)) * weight


def _launch(x, weight, out, eps):
    """Internal unchecked entry. Caller ensures validated nonempty inputs/output."""
    import triton
    from rmsnorm_kernel import kernel
    kernel[(x.shape[0],)](x, weight, out, x.shape[1], eps,
        BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=4)


def rmsnorm(x, weight, eps=1e-5):
    """New output [M,N]. Finite values/no FP32 overflow are NOT runtime scanned."""
    import torch
    eps = _validate(x, weight, cuda=True, eps=eps)
    with torch.cuda.device(x.device):
        out = torch.empty_like(x)
        if x.shape[0] != 0:
            _launch(x, weight, out, eps)
        return out


if __name__ == '__main__':
    print([[round(v, 6) for v in row] for row in
           reference_rows([[3., 4.], [0., 0.]], [1., 2.])])
