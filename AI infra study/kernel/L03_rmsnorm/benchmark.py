"""GPU only. Public API vs public API; allocation included on both sides."""
import argparse

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=int,default=4096)
    p.add_argument('--cols',type=int,default=1025)
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args()
    if args.rows<=0 or not 1<=args.cols<=4096: p.error('rows>0, cols in 1..4096 required')
    import torch
    import triton
    from triton.testing import do_bench
    from rmsnorm import rmsnorm, torch_reference
    if not torch.cuda.is_available(): raise SystemExit('CUDA required; no benchmark result produced')
    torch.cuda.set_device(args.device)
    x=torch.randn(args.rows,args.cols,device='cuda'); w=torch.ones(args.cols,device='cuda')
    expected = torch_reference(x,w)
    torch.testing.assert_close(rmsnorm(x,w),expected,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(torch.nn.functional.rms_norm(x,(args.cols,),w,eps=1e-5),expected,rtol=2e-5,atol=2e-6)
    print('torch',torch.__version__,'triton',triton.__version__,'GPU',torch.cuda.get_device_name())
    for name,fn in [('triton_wrapper',lambda:rmsnorm(x,w)),('torch_expression',lambda:torch_reference(x,w)),('torch_builtin',lambda:torch.nn.functional.rms_norm(x,(args.cols,),w,eps=1e-5))]:
        p50,p20,p80=do_bench(fn,warmup=25,rep=100,quantiles=[0.5,0.2,0.8])
        print(name, 'p50/p20/p80 ms:',p50,p20,p80)
    print('CUDA-event timings, warm JIT, output allocation included; not HTTP/request latency.')

if __name__=='__main__': main()
