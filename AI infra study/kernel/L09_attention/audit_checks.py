"""Independent offline audit: python -m unittest -v audit_checks."""
import math
import unittest
from attention import attention_materialized, attention_streaming, causal_keep


class IndependentAudit(unittest.TestCase):
    def check(self, q, k, v, mask, expected):
        results = [attention_materialized(q, k, v, mask).output]
        results += [attention_streaming(q, k, v, mask, b) for b in (1, 2, 3, 4, 7, 256)]
        for actual in results:
            self.assertEqual(len(actual), len(expected))
            for arow, erow in zip(actual, expected):
                self.assertEqual(len(arow), len(erow))
                for a, e in zip(arow, erow):
                    self.assertTrue(math.isclose(a, e, rel_tol=1e-11, abs_tol=1e-11), (a, e))

    def test_lesson_exact_answers(self):
        q, k, v = [[1]], [[0], [math.log(2)], [math.log(3)]], [[10], [20], [40]]
        self.check(q, k, v, None, [[170 / 6]])
        self.check(q, k, v, [[True, False, True]], [[32.5]])
        self.check(q, k[:2], v[:2], None, [[50 / 3]])

    def test_denominator_only_mutant_is_detected(self):
        # Deliberate bug: old numerator remains in its former exponential units.
        m, denominator, numerator = -math.inf, 0., 0.
        for score, value in [(0., 10.), (math.log(2), 20.)]:
            new_m = max(m, score)
            alpha = 0. if not denominator else math.exp(m - new_m)
            weight = math.exp(score - new_m)
            denominator = alpha * denominator + weight
            numerator = numerator + weight * value  # BUG: missing alpha
            m = new_m
        wrong = numerator / denominator
        self.assertEqual(wrong, 20.)
        self.assertFalse(math.isclose(wrong, 50 / 3, rel_tol=1e-11, abs_tol=1e-11))
        self.check([[1]], [[0], [math.log(2)]], [[10], [20]], None, [[50 / 3]])

    def test_common_score_offset(self):
        # Exact integer shifts avoid mixing invariance with decimal input rounding.
        q, v = [[1]], [[-3, 4], [7, -9], [2, 8]]
        base = [[-2], [0], [3]]
        for mask in (None, [[True, False, True]], [[False] * 3]):
            expected = attention_materialized(q, base, v, mask).output
            for offset in (-10000, -1000, 1000, 10000):
                self.check(q, [[row[0] + offset] for row in base], v, mask, expected)

    def test_arbitrary_positions_holes_padding_and_permutation(self):
        qp, kp = [4, 0, 2], [5, 0, 4, 1, 2]
        causal = causal_keep(qp, kp)
        self.assertEqual(causal, [[False, True, True, True, True],
                                  [False, True, False, False, False],
                                  [False, True, False, True, True]])
        padding = [True, False, True, False, True]
        keep = [[a and b for a, b in zip(row, padding)] for row in causal]
        q, k = [[0]] * 3, [[j] for j in kp]
        v = [[50], [0], [40], [10], [20]]
        expected = [[30], [0], [20]]
        self.check(q, k, v, keep, expected)
        order = [4, 2, 0, 3, 1]
        self.check(q, [k[j] for j in order], [v[j] for j in order],
                   [[row[j] for j in order] for row in keep], expected)

    def test_masked_extreme_score_has_no_effect(self):
        # Invalid keys exist in memory but must affect neither maximum nor denominator.
        q, k, v, keep = [[1]], [[-2], [1e6], [-2]], [[2], [-1e6], [6]], [[True, False, True]]
        self.check(q, k, v, keep, [[4]])
        trace = attention_materialized(q, k, v, keep)
        self.assertEqual(trace.probabilities, [[.5, 0., .5]])
        self.assertEqual(trace.scores[0][1], 1e6)
        self.assertEqual(trace.masked_scores[0][1], -math.inf)

    def test_distinct_feature_and_value_widths_scale(self):
        # Dot products [0, 2 log 2], divided by sqrt(D=4), give [0, log 2].
        q = [[1, 1, 1, 1]]
        k = [[0, 0, 0, 0], [2 * math.log(2), 0, 0, 0]]
        self.check(q, k, [[3, 0], [0, 6]], None, [[1, 4]])


if __name__ == '__main__':
    unittest.main()
