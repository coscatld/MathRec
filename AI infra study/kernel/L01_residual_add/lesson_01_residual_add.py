"""Lesson 01: contiguous float32 residual add, inference only.

No installation is performed. Default: stdlib simulation + optional Torch CPU tests.
--gpu: also run Triton CUDA tests. --bench: also benchmark preallocated outputs.
"""
import argparse
import math

try:
    import torch
except ImportError:
    torch = None
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

BLOCK = 256

if triton is not None:
    @triton.jit
    def _residual_add_kernel(X, R, Y, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < N
        x = tl.load(X + offsets, mask=valid, other=0.0)
        r = tl.load(R + offsets, mask=valid, other=0.0)
        y = x + r
        tl.store(Y + offsets, y, mask=valid)


def _validate(x, r, *, cuda):
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor tests/kernels")
    if not isinstance(x, torch.Tensor) or not isinstance(r, torch.Tensor):
        raise TypeError("x and r must be tensors")
    if x.shape != r.shape:
        raise ValueError("shapes must match; this lesson does not broadcast")
    if x.dtype != torch.float32 or r.dtype != torch.float32:
        raise TypeError("only float32 is supported in this lesson")
    if x.device != r.device:
        raise ValueError("devices must match")
    if not x.is_contiguous() or not r.is_contiguous():
        raise ValueError("only contiguous tensors are supported")
    if x.is_neg() or r.is_neg():
        raise ValueError("lazy negative tensor views are not supported")
    if x.requires_grad or r.requires_grad:
        raise ValueError("inference only; no autograd implementation")
    if cuda and x.device.type != "cuda":
        raise ValueError("the Triton kernel requires CUDA tensors")


def torch_reference(x, r):
    _validate(x, r, cuda=False)
    return torch.add(x, r)


def _launch(x, r, out):
    # Internal entry: inputs and output are already validated/preallocated.
    n = x.numel()
    if n == 0:
        return
    grid = (triton.cdiv(n, BLOCK),)
    with torch.cuda.device(x.device):
        _residual_add_kernel[grid](x, r, out, n, BLOCK_SIZE=BLOCK, num_warps=4)


def residual_add(x, r):
    _validate(x, r, cuda=True)
    if triton is None:
        raise RuntimeError("A compatible Triton installation is required")
    out = torch.empty_like(x)
    _launch(x, r, out)
    return out


def simulate_indexing(x, r, block=BLOCK):
    """CPU indexing model, NOT a Triton interpreter or float32 GPU test."""
    if len(x) != len(r) or block <= 0:
        raise ValueError("invalid lengths/block")
    n = len(x)
    out, writes = [None] * n, [0] * n
    for pid in range((n + block - 1) // block):
        for lane in range(block):
            i = pid * block + lane
            if i < n:  # masked loads and store
                out[i] = x[i] + r[i]
                writes[i] += 1
    assert writes == [1] * n
    return out


def test_indexing():
    for n in [0, 1, 255, 256, 257, 1025]:
        x, r = list(range(n)), [-2 * i for i in range(n)]
        assert simulate_indexing(x, r) == [-i for i in range(n)]
    print("PASS: 6 stdlib indexing/tail/unique-write cases (not GPU execution)")


def expect_error(fn, kind):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def test_tensors(device, use_kernel=False):
    operation = residual_add if use_kernel else torch_reference
    shapes = [(0,), (1,), (255,), (256,), (257,), (1025,), (3, 5), (2, 3, 7), ()]
    for shape in shapes:
        n = math.prod(shape)
        x = torch.arange(n, dtype=torch.float32, device=device).reshape(shape)
        r = (-2 * torch.arange(n, dtype=torch.float32, device=device)).reshape(shape)
        expected = (-torch.arange(n, dtype=torch.float32, device=device)).reshape(shape)
        y = operation(x, r)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
        if n:
            assert y.data_ptr() != x.data_ptr() and y.data_ptr() != r.data_ptr()
    generator = torch.Generator(device=device).manual_seed(7)
    x = torch.randn((1025,), device=device, dtype=torch.float32, generator=generator)
    r = torch.randn((1025,), device=device, dtype=torch.float32, generator=generator)
    torch.testing.assert_close(operation(x, r), x + r, rtol=1e-6, atol=1e-6)
    a = torch.tensor([float("inf"), -float("inf"), float("nan"), 0.0], device=device, dtype=torch.float32)
    b = torch.tensor([-float("inf"), 1.0, 1.0, -0.0], device=device, dtype=torch.float32)
    torch.testing.assert_close(operation(a, b), a + b, equal_nan=True)
    expect_error(lambda: operation(x, r[:1]), ValueError)
    expect_error(lambda: operation(x.half(), r.half()), TypeError)
    t = torch.zeros((3, 4), device=device).t()
    expect_error(lambda: operation(t, t), ValueError)
    expect_error(lambda: operation(x.clone().requires_grad_(), r), ValueError)
    if use_kernel:
        expect_error(lambda: residual_add(x, r.cpu()), ValueError)
        torch.cuda.synchronize(device)
    print(f"PASS: tensor cases on {device}; kernel={use_kernel}")


def benchmark():
    import triton.testing
    device = torch.device("cuda", torch.cuda.current_device())
    n = 2**22
    x = torch.randn(n, device=device, dtype=torch.float32)
    r = torch.randn_like(x)
    yt, yp = torch.empty_like(x), torch.empty_like(x)
    _validate(x, r, cuda=True)
    _launch(x, r, yt)  # compile before measurement
    torch.testing.assert_close(yt, x + r, rtol=1e-6, atol=1e-6)
    fns = {"triton_preallocated": lambda: _launch(x, r, yt),
           "torch_add_out": lambda: torch.add(x, r, out=yp)}
    for fn in fns.values():
        for _ in range(20):
            fn()
    torch.cuda.synchronize(device)
    for name, fn in fns.items():
        median_ms, p20_ms, p80_ms = triton.testing.do_bench(
            fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
        gbps = (12 * n) / (median_ms * 1e6)
        print(f"{name}: median={median_ms:.6f} ms, "
              f"p20={p20_ms:.6f}, p80={p80_ms:.6f}, logical_GB/s={gbps:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    test_indexing()
    if torch is not None:
        print(f"torch={torch.__version__}; bundled CUDA={torch.version.cuda}")
        test_tensors(torch.device("cpu"))
    else:
        print("SKIP: PyTorch CPU reference tests (PyTorch not installed)")
    if args.gpu or args.bench:
        if torch is None or triton is None or not torch.cuda.is_available():
            raise SystemExit("GPU path unavailable: compatible torch+Triton+CUDA GPU required")
        device = torch.device("cuda", torch.cuda.current_device())
        print(f"triton={triton.__version__}; GPU={torch.cuda.get_device_name(device)}; "
              f"capability={torch.cuda.get_device_capability(device)}")
        test_tensors(device, use_kernel=True)
        if args.bench:
            benchmark()


if __name__ == "__main__":
    main()
