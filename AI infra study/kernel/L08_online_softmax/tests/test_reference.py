import math
import random
import unittest
from online_softmax import State, block_state, merge, online_normalizer, stable_logsumexp, two_pass_softmax


class ReferenceTests(unittest.TestCase):
    def assertVectorClose(self, a, b, tol=3e-13):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(math.isclose(x, y, rel_tol=tol, abs_tol=tol), (x, y))

    def test_identity_and_empty(self):
        self.assertEqual(merge(State(), State()), State())
        self.assertEqual(stable_logsumexp([]), -math.inf)
        a = block_state([2, 3])
        self.assertEqual(merge(a, State()), a)
        self.assertEqual(merge(State(), a), a)
        self.assertEqual(online_normalizer([[], [], []]), State())
        with self.assertRaises(ValueError):
            two_pass_softmax([])

    def test_known_values_and_large_offset(self):
        self.assertVectorClose(two_pass_softmax([0, 0, 0]), [1 / 3] * 3)
        self.assertVectorClose(two_pass_softmax([1e30, 1e30]), [.5, .5])
        self.assertVectorClose(two_pass_softmax([1e308, -1e308]), [1, 0])
        self.assertAlmostEqual(stable_logsumexp([1000, 1000]), 1000 + math.log(2))
        self.assertEqual(two_pass_softmax([7]), [1])

    def test_contract(self):
        for bad in ([math.nan], [math.inf], [-math.inf]):
            with self.assertRaises(ValueError):
                block_state(bad)
        for bad in ([True], ['2'], [1j]):
            with self.assertRaises(TypeError):
                block_state(bad)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                two_pass_softmax([1], bad)
        with self.assertRaises(ValueError):
            State(0, 0)

    def test_seeded_partitions_order_shift_and_tails(self):
        rng = random.Random(20261106)
        for n in (1, 2, 3, 31, 127, 128, 129, 255, 256, 257, 1025):
            for trial in range(12):
                xs = [rng.uniform(-80, 80) for _ in range(n)]
                m = max(xs)
                es = [math.exp(x - m) for x in xs]
                expected = [v / math.fsum(es) for v in es]
                for width in (1, 7, 128, 256):
                    p = two_pass_softmax(iter(xs), width)
                    self.assertVectorClose(p, expected)
                    self.assertAlmostEqual(math.fsum(p), 1, places=13)
                    self.assertTrue(all(0 <= v <= 1 for v in p))
                # Variable partitions, empty chunks, and arbitrary merge order.
                blocks, i = [[], []], 0
                while i < n:
                    width = rng.randint(1, 33)
                    blocks.append(xs[i:i+width])
                    i += width
                rng.shuffle(blocks)
                s = online_normalizer(blocks)
                self.assertAlmostEqual(s.logsumexp, m + math.log(math.fsum(es)), places=12)
                self.assertVectorClose([math.exp(x - s.m) / s.l for x in xs], expected)
                shift = 128.0
                self.assertVectorClose(two_pass_softmax([x + shift for x in xs], 7), expected)
                a, b, c = (block_state(xs[:n//3]), block_state(xs[n//3:2*n//3]), block_state(xs[2*n//3:]))
                self.assertAlmostEqual(merge(merge(a, b), c).l, merge(a, merge(b, c)).l, places=12)


if __name__ == '__main__':
    unittest.main()
