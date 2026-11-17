import math
import random
import unittest
from softmax import reference_rows, simulate_padded

class ScalarTests(unittest.TestCase):
    def assertVector(self, a, b):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b): self.assertAlmostEqual(x, y, places=12)
    def test_known(self):
        self.assertVector(reference_rows([[0., math.log(2), math.log(3)]])[0], [1/6, 2/6, 3/6])
    def test_uniform(self):
        self.assertVector(reference_rows([[7., 7., 7.]])[0], [1/3]*3)
    def test_large(self):
        self.assertVector(reference_rows([[1000.,1001.,1002.]])[0],reference_rows([[0.,1.,2.]])[0])
    def test_negative(self):
        self.assertVector(reference_rows([[-1000.,-1001.]])[0],reference_rows([[0.,-1.]])[0])
    def test_single(self): self.assertEqual(reference_rows([[999.]]), [[1.]])
    def test_empty_batch(self): self.assertEqual(reference_rows([]), [])
    def test_empty_row(self):
        with self.assertRaises(ValueError): reference_rows([[]])
    def test_ragged(self):
        with self.assertRaises(ValueError): reference_rows([[1.], [2.,3.]])
    def test_nonfinite(self):
        for x in [math.inf, -math.inf, math.nan]:
            with self.assertRaises(ValueError): reference_rows([[x]])
    def test_padding(self):
        for n in [1,3,31,32,33,127,257,4096]:
            row = [random.Random(i).uniform(-10,10) for i in range(n)]
            y, b = simulate_padded(row)
            self.assertVector(y,reference_rows([row])[0])
            self.assertGreaterEqual(b,n)
            self.assertEqual(b & (b-1),0)
    def test_probability_invariants(self):
        rng=random.Random(42)
        for n in [1,3,31,127]:
            y=reference_rows([[rng.uniform(-20,20) for _ in range(n)]])[0]
            self.assertAlmostEqual(sum(y),1.,places=12)
            self.assertTrue(all(0 <= p <= 1 for p in y))
    def test_wrong_zero_padding_is_detectable(self):
        row=[-4.,-3.,-2.]
        wrong=reference_rows([row+[0.]])[0][:3]
        self.assertLess(sum(wrong),1.)

if __name__ == '__main__': unittest.main()
