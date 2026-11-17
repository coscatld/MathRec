"""Independent stdlib audit. Run: python audit_checks.py -v

Decimal oracle uses exact represented input values, not the implementation's
math.exp/max-shift reduction. GPU tests remain explicitly outside this audit.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import random
import subprocess
import sys
import unittest
from pathlib import Path
from online_softmax import State, block_state, merge, online_normalizer, stable_logsumexp, two_pass_softmax


def oracle(xs):
    # Direct exponentiation in Decimal avoids reproducing max-shift code.
    # These oracle cases deliberately stay within Decimal's exponent range.
    with localcontext() as ctx:
        ctx.prec = 90
        es = [Decimal.from_float(float(x)).exp() for x in xs]
        total = sum(es, Decimal(0))
        return [float(e / total) for e in es], float(total.ln())


def tree(states):
    if not states:
        return State()
    while len(states) > 1:
        states = [merge(states[i], states[i+1]) if i+1 < len(states) else states[i]
                  for i in range(0, len(states), 2)]
    return states[0]


class IndependentAudit(unittest.TestCase):
    def close_vector(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertTrue(math.isclose(a, b, rel_tol=2e-12, abs_tol=5e-324), (a, b))
        self.assertTrue(math.isclose(math.fsum(actual), 1., rel_tol=5e-14))
        self.assertTrue(all(0 <= a <= 1 for a in actual))

    def test_decimal_oracle_and_reduction_orders(self):
        cases = [[0.], [-0., 0.], [10000., 10000., 9999.], [-10000., -10000., -9999.],
                 [0., -700., -740., -744., -745., -746., -1000.],
                 [-100.] * 129 + [100.], [100.] + [-100.] * 129,
                 [0., math.nextafter(0., 1.), -math.nextafter(0., 1.)]]
        rng = random.Random(880611)
        for n in (2, 7, 31, 129):
            for _ in range(12):
                cases.append([rng.uniform(-200, 200) for _ in range(n)])
        for xs in cases:
            p, lse = oracle(xs)
            for width in (1, 3, 16, 128, 1024):
                actual = two_pass_softmax(iter(xs), width)
                self.close_vector(actual, p)
                self.assertEqual(actual[xs.index(max(xs))], max(actual))
            self.assertTrue(math.isclose(stable_logsumexp(xs), lse, rel_tol=3e-15, abs_tol=3e-14))
            states = [block_state(xs[i:i+3]) for i in range(0, len(xs), 3)] + [State(), State()]
            for ordered in (states, list(reversed(states))):
                s = tree(ordered)
                self.close_vector([math.exp(x-s.m)/s.l for x in xs], p)

    def test_extreme_binary64_and_common_offsets(self):
        high = sys.float_info.max
        for x in (high, -high, 1e30, -1e30):
            for n in (1, 2, 129, 1025):
                self.close_vector(two_pass_softmax([x]*n, 7), [1/n]*n)
                self.assertEqual(stable_logsumexp([x]*n), x)
        self.assertEqual(two_pass_softmax([-high, high, -high], 1), [0., 1., 0.])
        self.assertEqual(two_pass_softmax([math.nextafter(high, 0.), high], 1), [0., 1.])
        xs = [1e15, 1e15 + .125, 1e15 - .5]
        p, _ = oracle([0., .125, -.5])
        self.close_vector(two_pass_softmax(xs, 1), p)

    def test_identity_invalid_states_and_inputs(self):
        self.assertEqual(tree([]), State())
        self.assertEqual(online_normalizer(iter([iter([]), iter([])])), State())
        for m, ell in ((math.nan, 1), (math.inf, 1), (-math.inf, 1), (0, 0), (0, .5), (0, math.inf), (0, math.nan), (0, -1)):
            with self.assertRaises(ValueError):
                State(m, ell)
        for xs in ([float('nan')], [float('inf')], [-float('inf')]):
            for fn in (block_state, stable_logsumexp, two_pass_softmax):
                with self.assertRaises(ValueError):
                    fn(xs)
        for xs in ([False], [True], ['1'], [None], [1+0j], [Decimal('1')]):
            with self.assertRaises(TypeError):
                two_pass_softmax(xs)
        for width in (True, False, 0, -1, 2., None, '2'):
            with self.assertRaises(ValueError):
                two_pass_softmax([0], width)
        with self.assertRaises(OverflowError):
            block_state([10**1000])
        self.close_vector(two_pass_softmax([Fraction(1, 2), Fraction(1, 2)]), [.5, .5])

    def test_single_use_generators_and_shift(self):
        seen = []
        def inputs():
            for x in [1., 2., 3.]:
                seen.append(x)
                yield x
        self.close_vector(two_pass_softmax(inputs(), 1), oracle([1., 2., 3.])[0])
        self.assertEqual(seen, [1., 2., 3.])
        xs = [-3.5, 0., 2.25, -7.]
        for shift in (-128., 128., 1024.):
            shifted = [x+shift for x in xs]
            self.close_vector(two_pass_softmax(shifted, 1), two_pass_softmax(xs, 2))
            self.assertAlmostEqual(stable_logsumexp(shifted), stable_logsumexp(xs)+shift, places=12)

    def test_mutation_counterexamples_and_tail_simulation(self):
        # Mathematical padding simulation, NOT Triton/GPU execution.
        for n in (3, 127, 129, 257):
            row = [-100. - (i % 7) for i in range(n)]
            unchanged = row.copy()
            expected = two_pass_softmax(row, 128)
            self.assertEqual(row, unchanged)
            padded_count = ((n + 127) // 128) * 128 - n
            def simulated_padding(fill):
                padded = row + [fill] * padded_count
                m = max(padded)
                denominator = math.fsum(math.exp(v-m) for v in padded)
                return [math.exp(v-m)/denominator for v in row]
            self.close_vector(simulated_padding(-math.inf), expected)
            self.assertLess(math.fsum(simulated_padding(0.)), .01)
        # Omitting rescaling is numerically wrong even for tiny inputs.
        a, b = block_state([1., 2.]), block_state([3.])
        wrong_denominator = a.l + b.l
        wrong = [math.exp(v-3.)/wrong_denominator for v in [1., 2., 3.]]
        self.assertGreater(abs(math.fsum(wrong)-1.), .1)
        self.close_vector(two_pass_softmax([1., 2., 3.], 2), oracle([1., 2., 3.])[0])
        # A rounded huge LSE is not a safe probability reconstruction path.
        row = [1e30, 1e30]
        wrong = [math.exp(v-stable_logsumexp(row)) for v in row]
        self.assertEqual(wrong, [1., 1.])
        self.close_vector(two_pass_softmax(row, 1), [.5, .5])

    def test_benchmark_argument_checks(self):
        script = str(Path(__file__).with_name('benchmark.py'))
        for args in (['--rows', '0'], ['--cols', '0'], ['--cols', '65537'], ['--rows', '4096', '--cols', '65536'], ['--repeats', '0'], ['--warmup', '0']):
            proc = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn('error:', proc.stderr)


if __name__ == '__main__':
    unittest.main()
