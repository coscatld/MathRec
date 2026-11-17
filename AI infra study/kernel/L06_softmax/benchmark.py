"""GPU-only benchmark. No installation, network or CPU fallback timings."""
import argparse
import json
import torch
import triton
import triton.testing
from softmax import validate_tensor
from softmax_kernel import launch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=int,default=4096)
    p.add_argument('--cols',type=int,default=1024)
    args=p.parse_args()
    if args.rows < 1: p.error('rows must be positive')
    if not 1 <= args.cols <= 4096: p.error('cols must be in [1,4096]')
    x=torch.randn(args.rows,args.cols,device='cuda',dtype=torch.float32)
    validate_tensor(x,cuda=True)
    a=torch.empty_like(x); b=torch.empty_like(x)
    def custom(): launch(x,a)
    def baseline(): torch.softmax(x,dim=-1,out=b)
    custom(); baseline(); torch.cuda.synchronize()
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    print(json.dumps({'torch':torch.__version__,'triton':triton.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'shape':list(x.shape),'dtype':str(x.dtype),'boundary':'preallocated finite-input FP32 forward; validation/allocation excluded'},ensure_ascii=False))
    for name,fn in [('torch',baseline),('triton',custom)]:
        median=triton.testing.do_bench(fn,warmup=25,rep=100,return_mode='median')
        ideal_bytes=2*x.numel()*x.element_size()
        print(name, {'median_ms':median,'effective_GBps':ideal_bytes/median/1e6})

if __name__=='__main__': main()
