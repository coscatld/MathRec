"""Offline tests: python -m unittest -v. PyTorch is optional, never installed here."""
import copy
from decimal import Decimal, localcontext
import math
import random
import unittest
from attention import attention_materialized, attention_streaming, causal_keep, MAX_DIM

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


class AttentionTests(unittest.TestCase):
    def assertMatrixClose(self, actual, expected, tol=2e-11):
        self.assertEqual(len(actual), len(expected))
        for row, ref in zip(actual, expected):
            self.assertEqual(len(row), len(ref))
            for a, b in zip(row, ref):
                self.assertTrue(math.isfinite(a))
                self.assertTrue(math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b))

    def test_hand_computed_uniform(self):
        trace = attention_materialized([[0, 0]], [[1, 2], [3, 4]], [[2, -2], [6, 4]])
        self.assertEqual(trace.scores, [[0, 0]])
        self.assertEqual(trace.probabilities, [[.5, .5]])
        self.assertEqual(trace.output, [[4, 1]])

    def test_singleton(self):
        self.assertEqual(attention_streaming([[7]], [[-4]], [[9, -2]]), [[9, -2]])

    def test_masked_first_block_tail_and_holes(self):
        q, k, v = [[1]], [[j] for j in range(7)], [[j, -j] for j in range(7)]
        mask = [[False, False, True, False, True, False, True]]
        trace = attention_materialized(q, k, v, mask)
        self.assertEqual(trace.probabilities[0][0:2], [0, 0])
        self.assertEqual(trace.masked_scores[0][3], -math.inf)
        for block in [1, 2, 3, 4, 7, 8]:
            self.assertMatrixClose(attention_streaming(q, k, v, mask, block), trace.output)

    def test_all_masked(self):
        q, k, v, mask = [[1], [2]], [[1], [2], [3]], [[7, -8]] * 3, [[False] * 3] * 2
        trace = attention_materialized(q, k, v, mask)
        self.assertEqual(trace.output, [[0, 0], [0, 0]])
        self.assertEqual(trace.probabilities, [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(attention_streaming(q, k, v, mask, 2), trace.output)

    def test_randomized_and_probability_invariants(self):
        rng = random.Random(20261107)
        for case in range(45):
            l, s, d, dv = [rng.randrange(1, 10) for _ in range(4)]
            matrix = lambda r, c: [[rng.uniform(-5, 5) for _ in range(c)] for _ in range(r)]
            q, k, v = matrix(l, d), matrix(s, d), matrix(s, dv)
            mask = [[rng.random() > .4 for _ in range(s)] for _ in range(l)]
            mask[0] = [False] * s
            trace = attention_materialized(q, k, v, mask)
            for block in [1, 2, 3, 5, 16, MAX_DIM]:
                with self.subTest(case=case, block=block):
                    self.assertMatrixClose(attention_streaming(q, k, v, mask, block), trace.output)
            for i, p in enumerate(trace.probabilities):
                self.assertAlmostEqual(sum(p), 1.0 if any(mask[i]) else 0.0)
                for j, weight in enumerate(p):
                    self.assertGreaterEqual(weight, 0)
                    if not mask[i][j]:
                        self.assertEqual(weight, 0)
                allowed = [j for j in range(s) if mask[i][j]]
                if allowed:
                    for c, value in enumerate(trace.output[i]):
                        self.assertGreaterEqual(value + 1e-12, min(v[j][c] for j in allowed))
                        self.assertLessEqual(value - 1e-12, max(v[j][c] for j in allowed))

    def test_absolute_causality_decode(self):
        self.assertEqual(causal_keep([5], list(range(6))), [[True] * 6])
        self.assertEqual(causal_keep([2, 3], [0, 1, 2, 3, 4]),
                         [[True, True, True, False, False], [True, True, True, True, False]])
        self.assertEqual(causal_keep([2], [3, 0, 2]), [[False, True, True]])

    def test_future_value_perturbation(self):
        q, k = [[1, 2], [3, -1]], [[1, j] for j in range(6)]
        v = [[j, j * 2] for j in range(6)]
        changed = copy.deepcopy(v)
        changed[4:] = [[99999, -99999], [-500, 800]]
        mask = causal_keep([2, 3], list(range(6)))
        for fn in [attention_materialized, attention_streaming]:
            a, b = fn(q, k, v, mask), fn(q, k, changed, mask)
            if fn is attention_materialized:
                a, b = a.output, b.output
            self.assertEqual(a, b)

    def test_large_scores_are_stable(self):
        q, k, v = [[1e6]], [[-1e6], [0], [1e6]], [[-1e6], [0], [1e6]]
        self.assertEqual(attention_materialized(q, k, v).output, [[1e6]])
        self.assertEqual(attention_streaming(q, k, v, block_size=1), [[1e6]])

    def test_decimal_independent_reference(self):
        # Independent high-precision direct exp (no shared stable-softmax helper).
        q, k, v = [[1, -2], [3, 1]], [[2, 1], [-1, 3], [1, 2]], [[2, -3], [4, 5], [-7, 2]]
        mask = [[True, False, True], [True, True, True]]
        with localcontext() as ctx:
            ctx.prec = 70
            expected = []
            for i, row in enumerate(q):
                scores = [sum(Decimal(a) * Decimal(b) for a, b in zip(row, key))
                          / Decimal(2).sqrt() for key in k]
                weights = [s.exp() if mask[i][j] else Decimal(0) for j, s in enumerate(scores)]
                expected.append([float(sum(weights[j] * Decimal(v[j][c]) for j in range(3))
                                       / sum(weights)) for c in range(2)])
        self.assertMatrixClose(attention_materialized(q, k, v, mask).output, expected)
        self.assertMatrixClose(attention_streaming(q, k, v, mask, 2), expected)

    def test_inputs_are_not_mutated(self):
        args = ([[1]], [[2]], [[3]], [[True]])
        before = copy.deepcopy(args)
        attention_materialized(*args)
        attention_streaming(*args)
        self.assertEqual(args, before)

    def test_invalid_contracts(self):
        valid = ([[1]], [[1]], [[1]], [[True]])
        bad = [([], [[1]], [[1]], None), ([[1, 2], [1]], [[1]], [[1]], None),
               ([[1]], [[1, 2]], [[1]], None), ([[1]], [[1]], [[1], [2]], None),
               ([[1]], [[1]], [[1]], [[1]]), ([[1]], [[1]], [[1]], []),
               ([[1]], [[1]], [[1]], [[True, False]]),
               ([[1]] * (MAX_DIM + 1), [[1]], [[1]], None),
               ([[1] * (MAX_DIM + 1)], [[1]], [[1]], None)]
        for value in [float('nan'), float('inf'), -float('inf'), 1e7, True, '1', 10**1000]:
            for slot in range(3):
                args = list(valid)
                args[slot] = [[value]]
                bad.append(args)
        for args in bad:
            for fn in [attention_materialized, attention_streaming]:
                with self.subTest(fn=fn.__name__, args=str(args)[:80]):
                    with self.assertRaises(ValueError):
                        fn(*args)
        for block in [0, -1, True, 1.5, MAX_DIM + 1]:
            with self.assertRaises(ValueError):
                attention_streaming(*valid, block_size=block)
        for positions in [[], [-1], [True], [1.5], [10**10]]:
            with self.assertRaises(ValueError):
                causal_keep(positions, [0])

    @unittest.skipIf(torch is None, 'optional PyTorch not installed')
    def test_optional_pytorch_explicit(self):
        q, k, v, keep = self.torch_inputs()
        scores = (q @ k.T) / math.sqrt(q.shape[-1])
        active = keep.any(dim=-1)
        out = torch.zeros((q.shape[0], v.shape[1]), dtype=torch.float64)
        out[active] = torch.softmax(scores[active].masked_fill(~keep[active], -torch.inf), dim=-1) @ v
        reference = attention_materialized(q.tolist(), k.tolist(), v.tolist(), keep.tolist()).output
        self.assertMatrixClose(out.tolist(), reference)

    @unittest.skipIf(torch is None or not hasattr(F, 'scaled_dot_product_attention'),
                     'optional PyTorch SDPA unavailable')
    def test_optional_pytorch_sdpa(self):
        q, k, v, keep = self.torch_inputs()
        active = keep.any(dim=-1)
        out = torch.zeros((q.shape[0], v.shape[1]), dtype=torch.float64)
        # Only valid query rows enter the backend; all-masked policy stays ours.
        out[active] = F.scaled_dot_product_attention(
            q[active][None, None], k[None, None], v[None, None],
            attn_mask=keep[active][None, None], dropout_p=0.0, is_causal=False)[0, 0]
        reference = attention_streaming(q.tolist(), k.tolist(), v.tolist(), keep.tolist(), 2)
        self.assertMatrixClose(out.tolist(), reference)

    @staticmethod
    def torch_inputs():
        q = torch.tensor([[1., 2.], [-1., 3.], [0., 2.]], dtype=torch.float64)
        k = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 2.], [2., -1.]], dtype=torch.float64)
        v = torch.tensor([[1., 2., 3.], [4., 1., 0.], [-2., 3., 1.], [9., 0., 2.], [4., 2., 1.]], dtype=torch.float64)
        # A non-contiguous physical mask view, with holes and a fully masked row.
        storage = torch.zeros((3, 10), dtype=torch.bool)
        keep = storage[:, ::2]
        keep[:] = torch.tensor([[True, False, True, False, True], [False] * 5,
                                [False, True, True, False, True]])
        assert not keep.is_contiguous()
        return q, k, v, keep


if __name__ == '__main__':
    unittest.main()
