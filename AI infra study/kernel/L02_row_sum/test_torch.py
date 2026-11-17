"""Optional tests. Explicit SKIP without dependencies/device; never means GPU passed."""
import importlib.util
import unittest
from row_sum import row_sum
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
HAS_GPU = HAS_TORCH and torch.cuda.is_available() and importlib.util.find_spec('triton') is not None

@unittest.skipUnless(HAS_TORCH, 'PyTorch not installed')
class CPUReference(unittest.TestCase):
    def test_reference(self):
        x = torch.tensor([[1., 2., 3.], [-1., 0., 4.]])
        torch.testing.assert_close(x.sum(dim=1), torch.tensor([6., 3.]))
    def test_non_tensor(self):
        with self.assertRaises(TypeError): row_sum([[1.]])
    def test_cpu_rejected(self):
        with self.assertRaises(ValueError): row_sum(torch.ones(2, 3))

@unittest.skipUnless(HAS_GPU, 'requires PyTorch + Triton + CUDA')
class GPU(unittest.TestCase):
    def test_shapes(self):
        torch.manual_seed(42)
        for m,n in [(0,0),(0,33),(3,0),(1,1),(3,31),(3,32),(3,33),(7,257),(5,1025),(2,4096)]:
            x = torch.randn((m,n), device='cuda', dtype=torch.float32)
            y = row_sum(x)
            self.assertEqual(tuple(y.shape), (m,))
            self.assertEqual(y.dtype, torch.float32)
            self.assertEqual(y.device, x.device)
            ref = x.double().sum(dim=1)
            # Data-dependent absolute bound handles near-zero sums. Test range only.
            tol = 2e-6 * x.double().abs().sum(dim=1) + 1e-5
            self.assertTrue(bool(((y.double()-ref).abs() <= tol).all()))
            torch.testing.assert_close(y, x.sum(dim=1), rtol=1e-4, atol=1e-4)
    def test_cancellation_diagnostic(self):
        x = torch.tensor([[1e8,1.,-1e8,1.]], device='cuda')
        y = row_sum(x)
        # Don't require bitwise equality or falsely require the exact sum 2.
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertLessEqual(float((y.double()-x.double().sum(1)).abs().max()), 32.)
        print('cancellation diagnostic:', y.cpu().tolist(), 'FP64:', x.double().sum(1).cpu().tolist())
    def test_contract(self):
        for x, error in [
            (torch.ones(2,3,device='cuda',dtype=torch.float16), TypeError),
            (torch.ones(2,3,device='cuda').t(), ValueError),
            (torch.ones(3,device='cuda'), ValueError),
            (torch.ones(2,3,device='cuda',requires_grad=True), ValueError),
            (torch.ones(1,4097,device='cuda'), ValueError),
        ]:
            with self.assertRaises(error): row_sum(x)

if __name__ == '__main__': unittest.main(verbosity=2)
