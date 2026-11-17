"""Lesson 02: row reduction; strict FP32 inference contract. No third-party imports on CPU path."""
import math
import struct

MAX_COLS = 4096


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def reference_rows(rows):
    """High-accuracy Python reference, NOT a GPU implementation."""
    return [math.fsum(row) for row in rows]


def simulate_row(row):
    """One possible FP32 tree, NOT Triton's actual instruction order.

    Requires FP32-representable values and no intermediate overflow."""
    n = len(row)
    if n > MAX_COLS:
        raise ValueError('teaching limit: N <= 4096')
    if any(not math.isfinite(x) for x in row):
        raise ValueError('simulation requires finite values')
    if not n:
        return 0.0
    block = 1 << (n - 1).bit_length()
    values = [f32(row[i]) if i < n else 0.0 for i in range(block)]
    while len(values) > 1:
        values = [f32(values[i] + values[i + 1]) for i in range(0, len(values), 2)]
    return values[0]


def _validate(x):
    import torch
    if not isinstance(x, torch.Tensor):
        raise TypeError('expected torch.Tensor')
    if x.layout != torch.strided or x.ndim != 2:
        raise ValueError('expected dense strided [M,N]')
    if x.dtype != torch.float32:
        raise TypeError('only float32 is supported')
    if not x.is_contiguous() or x.is_neg() or x.is_conj():
        raise ValueError('expected ordinary contiguous tensor')
    if x.requires_grad:
        raise ValueError('inference only; backward is not implemented')
    if x.shape[0] > 2**31 - 1:
        raise ValueError('row count exceeds CUDA grid limit')
    if x.shape[1] > MAX_COLS:
        raise ValueError('teaching limit: N <= 4096')
    if x.device.type != 'cuda':
        raise ValueError('GPU wrapper requires CUDA; use reference_rows on CPU')



def _launch(x, out):
    """Internal unchecked path for benchmark: validated M>0,N>0 and matching out."""
    import triton
    from row_sum_kernel import kernel
    kernel[(x.shape[0],)](
        x, out, x.stride(0), x.shape[1],
        BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=4)


def row_sum(x):
    """[M,N] -> [M]. Finite inputs, no overflow assumed; NOT runtime-scanned.

    Metadata is checked. No autograd, no dtype conversion, no implicit contiguous copy.
    """
    import torch
    _validate(x)
    m, n = x.shape
    with torch.cuda.device(x.device):
        if m == 0:
            return torch.empty((0,), dtype=x.dtype, device=x.device)
        if n == 0:
            return torch.zeros((m,), dtype=x.dtype, device=x.device)
        out = torch.empty((m,), dtype=x.dtype, device=x.device)
        _launch(x, out)
        return out


if __name__ == '__main__':
    rows = [[1., 2., 3.], [-1., 0., 4.]]
    print('reference:', reference_rows(rows))
    print('tree simulation:', [simulate_row(row) for row in rows])
