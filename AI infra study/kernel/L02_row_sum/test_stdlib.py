import math
import random
import unittest
from row_sum import reference_rows, simulate_row, f32

class TestIndexAndReduction(unittest.TestCase):
    def test_no_gpu_import(self):
        import subprocess, sys
        subprocess.run([sys.executable, "-c", "import row_sum, sys; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)
    def test_empty(self):
        self.assertEqual(reference_rows([]), [])
        self.assertEqual([simulate_row([]) for _ in range(3)], [0., 0., 0.])
    def test_exact(self):
        for n in [1, 31, 32, 33, 255, 256, 257, 1025, 4096]:
            self.assertEqual(simulate_row([1.] * n), float(n))
    def test_negative(self):
        self.assertEqual(simulate_row([-2., -3., -4.]), -9.)
    def test_random(self):
        rng = random.Random(42)
        for n in [1, 3, 33, 257, 1025, 4096]:
            row = [f32(rng.uniform(-1, 1)) for _ in range(n)]
            bound = 2 * 2**-24 * max(1, math.ceil(math.log2(n))) * math.fsum(map(abs, row)) + 1e-6
            self.assertLessEqual(abs(simulate_row(row) - math.fsum(row)), bound)
    def test_cancellation(self):
        # FP32 addition is not associative. This simulated tree loses both ones.
        row = [1e8, 1., -1e8, 1.]
        self.assertEqual(math.fsum(row), 2.)
        self.assertEqual(simulate_row(row), 0.)
    def test_limit(self):
        with self.assertRaises(ValueError): simulate_row([0.] * 4097)
    def test_nonfinite(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.assertRaises(ValueError): simulate_row([value])

if __name__ == '__main__': unittest.main(verbosity=2)
