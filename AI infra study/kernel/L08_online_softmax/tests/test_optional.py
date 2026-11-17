import importlib.util
import unittest
from online_softmax import two_pass_softmax, stable_logsumexp

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
HAS_GPU = bool(HAS_TORCH and torch.cuda.is_available() and torch.version.hip is None
               and importlib.util.find_spec('triton') is not None)


@unittest.skipUnless(HAS_TORCH, 'optional PyTorch not installed')
class TorchReferenceTests(unittest.TestCase):
    def test_cpu_float64_oracle(self):
        g = torch.Generator().manual_seed(20261106)
        for n in (1, 17, 129, 1025):
            x = torch.randn(n, generator=g, dtype=torch.float64) * 30
            actual = torch.tensor(two_pass_softmax(x.tolist(), 128), dtype=torch.float64)
            torch.testing.assert_close(actual, torch.softmax(x, 0), rtol=1e-12, atol=1e-14)
            self.assertAlmostEqual(stable_logsumexp(x.tolist()), torch.logsumexp(x, 0).item(), places=12)


@unittest.skipUnless(HAS_GPU, 'optional NVIDIA CUDA + PyTorch + Triton unavailable')
class TritonTests(unittest.TestCase):
    def test_shapes_dtypes_blocks_and_offsets(self):
        from triton_online_softmax import softmax
        torch.manual_seed(20261106)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            tol = {torch.float32: 3e-6, torch.float16: 2e-3, torch.bfloat16: 1e-2}[dtype]
            for m, n in ((0, 5), (1, 1), (3, 127), (3, 128), (3, 129), (7, 257), (2, 1025), (1, 65536)):
                x = (torch.randn((m, n), device='cuda') * 8).to(dtype)
                for block in (128, 256, 512, 1024):
                    y = softmax(x, block)
                    expected = torch.softmax(x.double(), -1).to(dtype)
                    torch.testing.assert_close(y, expected, rtol=tol, atol=tol * .01)
                    if m:
                        torch.testing.assert_close(y.float().sum(-1), torch.ones(m, device='cuda'), rtol=tol, atol=tol)
            # Identical huge FP32 inputs: exp(x-lse) would incorrectly give ones.
            if dtype == torch.float32:
                x = torch.full((2, 257), 1e20, device='cuda')
                torch.testing.assert_close(softmax(x), torch.full_like(x, 1/257), rtol=3e-6, atol=1e-8)
        if torch.cuda.device_count() > 1:
            # Caller current device differs from the tensor's device.
            with torch.cuda.device(0):
                x = torch.randn(2, 129, device='cuda:1')
                torch.testing.assert_close(softmax(x), torch.softmax(x, -1), rtol=3e-6, atol=3e-8)
                self.assertEqual(torch.cuda.current_device(), 0)

    def test_rejections(self):
        from triton_online_softmax import softmax
        cases = [torch.ones(2, 3), torch.ones(3, device='cuda'),
                 torch.ones(2, 0, device='cuda'), torch.ones(1, 65537, device='cuda'),
                 torch.ones(2, 3, dtype=torch.float64, device='cuda'),
                 torch.ones(2, 3, device='cuda', requires_grad=True),
                 torch.ones(2, 3, device='cuda').T,
                 torch.tensor([[float('nan')]], device='cuda'),
                 torch.tensor([[float('inf')]], device='cuda'),
                 torch.tensor([[2e30]], device='cuda')]
        for x in cases:
            with self.assertRaises((ValueError, TypeError)):
                softmax(x)
        with self.assertRaises(ValueError):
            softmax(torch.ones(2, 3, device='cuda'), 64)
