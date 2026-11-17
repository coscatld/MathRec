import math
import unittest
from rmsnorm import reference_rows, validate_eps


class StandardLibraryTests(unittest.TestCase):
    def test_example(self):
        out = reference_rows([[3., 4.]], [1., 2.])[0]
        self.assertAlmostEqual(out[0], 3/math.sqrt(12.5+validate_eps(1e-5)))
        self.assertAlmostEqual(out[1], 8/math.sqrt(12.5+validate_eps(1e-5)))

    def test_zero(self):
        self.assertEqual(reference_rows([[0., 0.]], [3., -2.]), [[0., 0.]])

    def test_empty_rows(self):
        self.assertEqual(reference_rows([], [1., 2.]), [])

    def test_empty_width(self):
        with self.assertRaises(ValueError): reference_rows([[]], [])

    def test_single(self):
        self.assertAlmostEqual(reference_rows([[-3.]], [2.])[0][0], -6/math.sqrt(9+validate_eps(1e-5)))

    def test_non_power_of_two(self):
        for n in (3, 31, 33, 1025, 4096):
            row = reference_rows([[2.]*n], [1.]*n)[0]
            self.assertEqual(len(row), n)
            self.assertTrue(all(abs(v-2/math.sqrt(4+validate_eps(1e-5)))<1e-12 for v in row))

    def test_bad_eps(self):
        for eps in (0, -1, float('nan'), float('inf'), 1e-50, 2):
            with self.assertRaises(ValueError): validate_eps(eps)
        for eps in (True, '1e-5', None):
            with self.assertRaises(TypeError): validate_eps(eps)

    def test_bad_shape_and_values(self):
        for rows, w in (([[1.]], [1., 2.]), ([[float('nan')]], [1.]), ([[1.]], [float('inf')]), ([[]], [1.]*4097)):
            with self.assertRaises(ValueError): reference_rows(rows, w)

    def test_eps_endpoints(self):
        for eps in (2**-126, 1.):
            self.assertEqual(validate_eps(eps), eps)
            self.assertEqual(reference_rows([[0.]], [1.], eps), [[0.]])

    def test_eps_effect(self):
        self.assertGreater(reference_rows([[0.001]], [1.], 1e-5)[0][0], reference_rows([[0.001]], [1.], 1e-2)[0][0])


if __name__ == '__main__': unittest.main()
