import importlib.util
import unittest
from softmax import torch_reference, softmax, reference_rows
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
HAS_GPU = HAS_TORCH and torch.cuda.is_available() and torch.version.hip is None and importlib.util.find_spec('triton') is not None

@unittest.skipUnless(HAS_TORCH, 'PyTorch not installed')
class TorchTests(unittest.TestCase):
    def test_cpu_reference(self):
        x=torch.tensor([[0.,1.,2.],[-1000.,-1001.,-999.]])
        torch.testing.assert_close(torch_reference(x),torch.tensor(reference_rows(x.tolist())))
    def test_contracts(self):
        cases=[torch.empty(2,0),torch.zeros(2,4097),torch.zeros(3),torch.ones(2,3,dtype=torch.int64),torch.zeros(3,2).t(),torch.ones(2,3,requires_grad=True),torch.tensor([[float('inf')]])]
        for x in cases:
            with self.assertRaises((ValueError,TypeError)): torch_reference(x)
    def test_empty(self): self.assertEqual(torch_reference(torch.empty(0,7)).shape,(0,7))

@unittest.skipUnless(HAS_GPU,'NVIDIA CUDA/PyTorch/Triton unavailable')
class GpuTests(unittest.TestCase):
    def test_shapes_dtypes(self):
        torch.manual_seed(42)
        dtypes=[torch.float32,torch.float16]
        if torch.cuda.is_bf16_supported(): dtypes.append(torch.bfloat16)
        for dt in dtypes:
            tol={torch.float32:2e-6,torch.float16:1e-3,torch.bfloat16:8e-3}[dt]
            for m,n in [(0,3),(1,1),(3,3),(7,31),(7,33),(17,127),(3,257),(2,4096)]:
                x=torch.randn(m,n,device='cuda',dtype=dt)*3
                out=softmax(x)
                torch.testing.assert_close(out,torch_reference(x),atol=tol,rtol=tol)
                self.assertEqual(out.shape,x.shape)
                self.assertEqual(out.dtype,x.dtype)
                self.assertEqual(out.device,x.device)
                if m:
                    torch.testing.assert_close(out.float().sum(-1),torch.ones(m,device='cuda'),atol=tol,rtol=tol)
    def test_large_uniform(self):
        for vals in [[[1000.,1001.,1002.]], [[-1000.,-1000.,-1000.]]]:
            x=torch.tensor(vals,device='cuda')
            torch.testing.assert_close(softmax(x),torch_reference(x),atol=2e-6,rtol=2e-6)
    def test_reject_nonfinite(self):
        for val in [float('inf'),float('-inf'),float('nan')]:
            with self.assertRaises(ValueError): softmax(torch.tensor([[val]],device='cuda'))

if __name__=='__main__': unittest.main()
