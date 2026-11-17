# Validation · Lesson 08 · 2026-11-06

## Environment observed

- CPython 3.12.14, Clang 22.1.3
- Linux 6.18.57 x86_64, glibc 2.41
- `importlib.util.find_spec('torch')`: absent
- `importlib.util.find_spec('triton')`: absent
- No dependencies installed; no models downloaded; no network/training/deployment performed.

## Checks actually run

Commands run from the package directory:

```sh
python -m compileall -q online_softmax.py triton_online_softmax.py example.py benchmark.py tests
python -m unittest discover -s tests -v
python example.py
python benchmark.py
```

- Python syntax compilation: passed. This does **not** compile or execute a Triton kernel.
- Unit test discovery: **7 methods, 4 passed, 3 skipped**, no failures.
- Passing reference tests: empty identity, empty LSE and rejected empty softmax; known outputs and extreme offsets; invalid inputs and block sizes; deterministic randomized properties.
- Randomized test seed: 20261106. 132 generated inputs (11 widths × 12 trials), each with four block sizes, variable partitions including empty blocks, shuffled block order, constant shift, sum/nonnegativity checks, and three-way approximate associativity. Widths include 127/128/129 and 255/256/257 tails, up to 1025.
- Example: passed; LSE `1002.4401896985612`; probabilities sum to approximately one; `[1e30,1e30]` returns `[0.5,0.5]`.
- Benchmark: exited normally with `SKIP: optional torch/triton dependencies unavailable; no GPU timing collected.`

## Not validated here

1. PyTorch float64 CPU oracle test skipped: PyTorch absent.
2. Triton numerical test skipped: CUDA/PyTorch/Triton stack unavailable.
3. Triton rejection test skipped for the same reason.
4. No GPU JIT compilation, GPU execution, GPU performance measurements, or hardware compatibility claims.
5. Optional BF16 numerical cases require BF16-supporting hardware and are omitted on unsupported hardware.

Optional GPU test coverage is supplied for M=0, N=1, partial blocks, N=65536, four block sizes, three dtypes when supported, a large common FP32 offset, and contract violations. It is executable test code, **not evidence these cases passed**. The stated dependency families are reference targets, not a validated environment lockfile. Run the full test command on the intended NVIDIA CUDA system before relying on the optional kernel.

This package is educational and forward-only; it is not a production or autograd softmax implementation.

## Independent audit and final review

`python audit_checks.py` passed **6 methods**. The independent test code uses the Python standard library only.

- Decimal direct-exp reference at 90-digit precision, using exact represented binary64 inputs: 56 vectors (8 curated + 48 seeded), 5 block widths each = 280 Softmax comparisons; 112 tree-order comparisons and 56 LSE comparisons.
- Includes subnormal probabilities, ±10000 offsets, late maxima, partial blocks, extreme finite binary64 values, 16 equal-offset/length combinations, one-shot iterators, Fraction values, invalid inputs/states/block sizes, 3 exactly representable shift checks and 6 CLI rejection checks.
- Four explicit negative-row padding simulations (3/127/129/257 widths), wrong-rescaling and rounded-LSE counterexamples; input non-mutation and maximum-position checks. These are mathematical simulations, **not GPU validation**.
- Review corrected the launch wrapper to enter `torch.cuda.device(x.device)` and restore the caller context. The supplied GPU test includes a differing-current-device case only on systems with multiple GPUs. This correction is statically reviewed but untested on a GPU here.
- All seven delivered Python files were syntax-checked after final edits: core, optional GPU implementation, example, benchmark, two original test modules and independent audit.
- Final original suite: **4 passed, 3 skipped**. Independent audit: **6 passed**. No failures. Optional BF16 and multiple-GPU subcases can be omitted on hardware without those capabilities, even when the GPU test method runs.

The final lesson embeds the core and GPU source verbatim. GPU function returns only probabilities, not LSE. Its ideal main-data traffic estimate is ~3*M*N*dtype_bytes, excluding validation and hardware effects.
