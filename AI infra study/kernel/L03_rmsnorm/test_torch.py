import importlib.util
import unittest
from rmsnorm import torch_reference, rmsnorm, reference_rows
try:
    import torch
except ImportError:
    torch = None
HAS_GPU = torch is not None and torch.cuda.is_available() and importlib.util.find_spec('triton') is not None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class TorchCPU(unittest.TestCase):
    def test_reference(self):
        x = torch.tensor([[3.,4.],[0.,0.]], dtype=torch.float32)
        w = torch.tensor([1.,2.])
        actual = torch_reference(x,w)
        torch.testing.assert_close(actual, torch.tensor(reference_rows(x.tolist(),w.tolist())), rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(actual, torch.nn.functional.rms_norm(x,(2,),w,eps=1e-5),rtol=2e-5,atol=2e-6)

    def test_contract(self):
        x=torch.ones(2,3); w=torch.ones(3)
        cases=[(x.double(),w),(x,w[:2]),(x.t(),torch.ones(2)),(x.clone().requires_grad_(),w)]
        for a,b in cases:
            with self.assertRaises((ValueError,TypeError)): torch_reference(a,b)
        with self.assertRaises(ValueError): rmsnorm(torch.ones(2,3),w)
        with self.assertRaises(ValueError): torch_reference(torch.empty(2,0),torch.empty(0))

    def test_empty(self):
        self.assertEqual(tuple(torch_reference(torch.empty(0,3),torch.ones(3)).shape),(0,3))


@unittest.skipUnless(HAS_GPU,'CUDA + Triton unavailable')
class GPU(unittest.TestCase):
    def test_shapes(self):
        torch.manual_seed(13)
        for m,n in ((0,3),(1,1),(3,31),(3,33),(7,1025),(2,4096)):
            x=torch.randn(m,n,device='cuda');w=torch.randn(n,device='cuda')
            expected=torch_reference(x,w)
            torch.testing.assert_close(rmsnorm(x,w),expected,rtol=2e-5,atol=2e-6)

    def test_numeric_cases(self):
        for value in (0.,1e-8,1.,-1.,1000.):
            x=torch.full((2,33),value,device='cuda');w=torch.linspace(-2,2,33,device='cuda')
            actual=rmsnorm(x,w)
            expected=((x.double()*torch.rsqrt(x.double().square().mean(1,keepdim=True)+1e-5))*w.double()).float()
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
            self.assertTrue(torch.isfinite(actual).all().item())

    def test_weight_grad_and_tail(self):
        x=torch.ones(2,33,device='cuda');w=torch.ones(33,device='cuda',requires_grad=True)
        with self.assertRaises(ValueError): rmsnorm(x,w)
        torch.testing.assert_close(rmsnorm(x,w.detach()),torch_reference(x,w.detach()),rtol=2e-5,atol=2e-6)


if __name__ == '__main__': unittest.main()
