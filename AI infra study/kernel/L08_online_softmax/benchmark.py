"""Opt-in microbenchmark. No dependency installs, network, or speed claims."""
import argparse
import importlib.util
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=256)
    parser.add_argument('--cols', type=int, default=4097)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.rows <= 4096 or not 1 <= args.cols <= 65536:
        parser.error('benchmark requires 1<=rows<=4096 and 1<=cols<=65536')
    if args.rows * args.cols > 16777216:
        parser.error('educational benchmark limited to 16M elements')
    if args.repeats < 1 or args.warmup < 1:
        parser.error('repeats and warmup must be positive')
    if any(importlib.util.find_spec(n) is None for n in ('torch', 'triton')):
        print('SKIP: optional torch/triton dependencies unavailable; no GPU timing collected.')
        return
    import torch
    if not torch.cuda.is_available() or torch.version.hip is not None:
        print('SKIP: NVIDIA CUDA unavailable; no GPU timing collected.')
        return
    import triton
    from triton_online_softmax import softmax, validate, _launch_validated
    torch.manual_seed(20261106)
    x = torch.randn(args.rows, args.cols, device='cuda', dtype=torch.float32)
    out_t = torch.empty_like(x)
    out_p = torch.empty_like(x)
    validate(x)
    torch.testing.assert_close(softmax(x), torch.softmax(x.double(), -1).float(), rtol=3e-6, atol=3e-8)

    def checked_torch():
        validate(x)  # Same shape/range/finiteness checks and synchronization as custom API.
        return torch.softmax(x, -1)

    def measure(fn, events):
        for _ in range(args.warmup):
            fn()  # Includes JIT compilation and allocator warmup outside measured samples.
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            if events:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            else:
                torch.cuda.synchronize()
                start = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    print(f'Python/PyTorch CUDA: torch={torch.__version__}, CUDA={torch.version.cuda}, triton={triton.__version__}')
    print(f'GPU={torch.cuda.get_device_name()}, shape={tuple(x.shape)}, dtype={x.dtype}, block=256')
    print(f'warmup={args.warmup}, repeats={args.repeats}; median milliseconds, warmed inputs')
    jobs = [('Triton unchecked/preallocated', lambda: _launch_validated(x, out_t, 256), True),
            ('PyTorch unchecked/preallocated', lambda: torch.softmax(x, -1, out=out_p), True),
            ('Triton checked/allocated', lambda: softmax(x), False),
            ('PyTorch checked/allocated', checked_torch, False)]
    for name, fn, events in jobs:
        print(f'{name}: {measure(fn, events):.6f} ms ({"CUDA events" if events else "host clock + sync"})')
    print('Compare only within a scope. No universal speed claim; not a cold-cache or production load test.')


if __name__ == '__main__':
    main()
