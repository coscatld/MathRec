"""CPU Python teaching timings, NOT a GPU speedup/FlashAttention benchmark."""
import argparse
import math
import platform
import random
import statistics
import time
from attention import attention_materialized, attention_streaming


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error('--repeats must be 1..100')
    rng = random.Random(9)
    matrix = lambda rows, cols: [[rng.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
    q, k, v = matrix(16, 32), matrix(64, 32), matrix(64, 16)
    expected = attention_materialized(q, k, v).output
    actual = attention_streaming(q, k, v, block_size=8)
    assert all(math.isclose(a, b, rel_tol=2e-11, abs_tol=2e-11)
               for row, ref in zip(actual, expected) for a, b in zip(row, ref))
    print(f'Python {platform.python_version()}, {platform.system()} {platform.machine()}')
    print(f'Correctness checked before timing; repeats={args.repeats}, warmup=1 per method.')
    print('CPU Python, L=16 S=64 D=32 Dv=16; includes validation/allocation.')
    print('No warm GPU kernel, HBM bandwidth or GPU performance claims.')
    for name, call in [('materialized (trace)', lambda: attention_materialized(q, k, v)),
                       ('streaming block=8', lambda: attention_streaming(q, k, v, block_size=8))]:
        call()
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            call()
            timings.append(1000 * (time.perf_counter() - start))
        print(f'{name}: median {statistics.median(timings):.3f} ms')
    print('The streaming lesson is about memory structure and invariants, not Python speed.')


if __name__ == '__main__':
    main()
