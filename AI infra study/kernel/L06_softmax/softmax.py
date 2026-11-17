"""Lesson 06: finite-input row softmax. No dependencies for scalar reference."""
import math

def reference_rows(rows):
    """Python float reference, not a float32/GPU emulator."""
    if not rows:
        return []
    n = len(rows[0])
    if n == 0 or any(len(r) != n for r in rows):
        raise ValueError('nonempty rectangular rows required')
    if any(not math.isfinite(v) for r in rows for v in r):
        raise ValueError('only finite logits supported')
    result = []
    for row in rows:
        maximum = max(row)
        numer = [math.exp(v - maximum) for v in row]
        denom = math.fsum(numer)
        result.append([v / denom for v in numer])
    return result

def simulate_padded(row):
    """Demonstrates padding semantics only; not GPU compilation validation."""
    reference_rows([row])  # check contract
    block = 1 << (len(row) - 1).bit_length()
    values = list(row) + [-math.inf] * (block - len(row))
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    denom = math.fsum(exps)
    return [v / denom for v in exps[:len(row)]], block

def validate_tensor(x, *, cuda=False, check_values=True):
    import torch
    if not isinstance(x, torch.Tensor):
        raise TypeError('expected torch.Tensor')
    if x.ndim != 2 or x.shape[1] < 1 or x.shape[1] > 4096:
        raise ValueError('shape must be [M,N], 1 <= N <= 4096')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('expected fp16, bf16 or fp32')
    if not x.is_contiguous():
        raise ValueError('contiguous row-major input required')
    if x.requires_grad:
        raise ValueError('forward-only teaching kernel; autograd unsupported')
    if cuda and (not x.is_cuda or torch.version.hip is not None):
        raise ValueError('NVIDIA CUDA input required')
    if check_values and not torch.isfinite(x).all().item():
        raise ValueError('only finite logits supported; no semantic mask')

def torch_reference(x):
    import torch
    validate_tensor(x)
    return torch.softmax(x, dim=-1, dtype=torch.float32).to(x.dtype)

def softmax(x):
    import torch
    validate_tensor(x, cuda=True)
    from softmax_kernel import launch
    out = torch.empty_like(x)
    if x.shape[0]:
        launch(x, out)
    return out
