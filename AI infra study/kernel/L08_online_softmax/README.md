# Lesson 08 · Stable log-sum-exp and online Softmax · 2026-11-06

A small, offline, forward-only teaching package. The standard-library path requires only Python 3.10+; it imports neither NumPy nor PyTorch. No install, model download, training, network call, or deployment occurs.

## Run first

From this directory:

```sh
python example.py
python -m unittest discover -s tests -v
python audit_checks.py
python benchmark.py
```

The example and CPU reference always run without optional libraries. The benchmark and optional tests report explicit skips when dependencies/hardware are missing. Missing dependencies are never silently counted as a numerical pass.

## Files

- `online_softmax.py`: finite-input binary64 reference, mergeable state, online reduction, two-pass output.
- `triton_online_softmax.py`: optional educational NVIDIA CUDA kernel and validated public wrapper.
- `example.py`: full runnable example, including large common offsets.
- `tests/test_reference.py`: unit tests and deterministic randomized properties.
- `tests/test_optional.py`: PyTorch float64 CPU oracle and optional GPU contract/numerical tests.
- `audit_checks.py`: independent Decimal-oracle, extreme-value and boundary tests.
- `benchmark.py`: warmed, synchronized optional GPU timings with explicitly separate scopes.
- `requirements-optional.txt`: optional dependency targets, not an installation requirement for this lesson.
- `VALIDATION.md`: exactly what was and was not run.
- `TEST_OUTPUT.txt`: captured final run output.
- `lesson.md`: full Chinese lesson with core source included.

## Core invariant and API

For a nonempty block, `m = max(x)` and `l = sum(exp(x-m))`. The pair contains everything needed for its log normalizer, `m + log(l)`.

To merge states A and B, let `m = max(m_A,m_B)` and compute
`l = l_A*exp(m_A-m) + l_B*exp(m_B-m)`.
Each old sum must be rescaled into the new maximum's coordinate system. Adding `l_A+l_B` directly is generally wrong.

`State()` is `(-inf,0)`. `merge` handles either empty state **before** subtracting maxima, avoiding `-inf - -inf = NaN`. Real-arithmetic merging is associative and commutative; floating-point merging is only approximately so. Tests use tolerances, not bitwise equivalence.

```python
from online_softmax import block_state, merge, online_normalizer
from online_softmax import stable_logsumexp, two_pass_softmax

state = online_normalizer([[1000, 1001], [], [999, 1002]])
assert state == merge(block_state([1000, 1001]), block_state([999, 1002]))
print(state.m, state.l, state.logsumexp)
print(stable_logsumexp([1000, 1001, 999, 1002]))
print(two_pass_softmax([1000, 1001, 999, 1002], block_size=2))
```

Reference inputs must be finite `numbers.Real` values convertible to Python float; booleans and complex values are rejected. This is not arbitrary-precision arithmetic. Integers beyond float representability can fail conversion. Empty log-sum-exp is `-inf`; empty softmax is rejected. `block_size` must be a positive integer.

## Numerical details that matter

- Output is `exp(x-m)/l`, **not** `exp(x-logsumexp(x))`. For two equal `1e30` inputs, rounding `m+log(2)` to `m` is unavoidable; reconstructing with that rounded LSE would give `[1,1]`. Keeping `(m,l)` yields `[0.5,0.5]`.
- Subtracting the maximum prevents exponential overflow. Tiny probabilities can underflow to zero. Even a finite binary64 input pair can have a difference of `-inf` at extreme magnitudes; `exp(-inf)=0` is the appropriate limiting behavior here.
- Stability does not recover information already lost when inputs were rounded. In FP32, `1e20+1` and `1e20` become equal. GPU state/reductions use FP32 and Triton exponential approximations, so CPU binary64 and GPU outputs need not match bitwise.
- `online_normalizer` consumes blocks once, keeping only the state and current block. `two_pass_softmax` materializes its input for replay and output, using O(N) storage. It is not an O(1)-memory streaming probability emitter.
- An online normalizer is not FlashAttention. Attention can fuse accumulation of a weighted value vector and avoid materializing the probability matrix; this example only computes standalone softmax.

## Optional Triton contract

```python
import torch
from triton_online_softmax import softmax
x = torch.randn(3, 1025, device='cuda', dtype=torch.float32)
y = softmax(x, block_size=256)
torch.testing.assert_close(y, torch.softmax(x, -1), rtol=3e-6, atol=3e-8)
```

Public `softmax` validates before launching:

- NVIDIA CUDA only; ROCm is outside this lesson.
- Contiguous `[M,N]`, nonnegative M, `1 <= N <= 65536`; M=0 returns an empty tensor without a launch, N=0 is rejected.
- FP16, BF16 or FP32; the installed GPU/runtime must support the selected dtype.
- `requires_grad=False`; no backward/autograd implementation.
- Finite inputs with absolute value <= `1e30`; this modest numerical bound avoids FP32 subtraction overflow, not loss of small differences.
- Block width is exactly 128, 256 (default), 512 or 1024.

One program owns one complete row. Pass one loops over blocks and updates `(m,l)`. Pass two rereads the row, computes probabilities, and writes output in the original dtype. Tail lanes load `-inf`; every processed block has at least one real lane because N>0. No all-padding block enters the recurrence. The first state's `exp(-inf-finite)=0` is safe.

There are roughly two input reads and one output write per element in the kernel, excluding caching effects. The public wrapper adds full input validation scans and host synchronization. This is a teaching implementation, not a tuned replacement for PyTorch; long-row serial block loops or small row counts can underutilize a GPU. Width bounds are scope controls, not performance promises. The private `_launch_validated` exists only for scoped benchmarks; do not use it as an unchecked public API.

## Benchmarks

```sh
python benchmark.py --rows 256 --cols 4097 --warmup 20 --repeats 100
```

Before measurement, correctness is compared with a float64 PyTorch oracle. JIT compilation and warmup are excluded. Both implementations use the same resident input, dtype, shape and device.

1. **Unchecked/preallocated:** CUDA-event timings for the Triton launch and `torch.softmax(..., out=...)`; allocation and validation are excluded on both sides. Individual event timings can include launch starvation; they are not pure instruction latency.
2. **Checked/allocated:** host-clock timings including matching validation, output allocation, and synchronization on both sides. PyTorch uses the same validator deliberately so this comparison does not misattribute checking cost to only one algorithm.

Only compare within a scope. Repeated data is warmed; cache effects, power state, GPU model, load, and framework versions affect results. No cold-cache, concurrency, gradient, quantization, distributed, or production-throughput claims are made. No GPU timings were available in this delivery environment.

## Suggested exercises

1. Deliberately remove rescaling in `merge`; find a two-block counterexample.
2. Replace the empty identity branches with the general formula; explain the NaN.
3. Change block size and order; compare tolerance-level differences rather than exact equality.
4. Sketch how `(m,l)` extends to `(m,l,weighted_value_sum)` in fused attention. Do not claim this standalone kernel is already fused attention.

## References

- Milakov and Gimelshein, *Online normalizer calculation for softmax* (2018): https://arxiv.org/abs/1805.02867
- Triton fused softmax tutorial: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- Triton fused attention tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- PyTorch logsumexp: https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html
