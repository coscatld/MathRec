"""Optional GPU benchmark; no installation/download/API calls."""
import argparse

def main():
    import torch
    import triton
    import triton.testing
    from row_sum import row_sum, _launch
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=4096)
    parser.add_argument('--cols', type=int, default=1025)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.rows <= 0 or not 1 <= args.cols <= 4096:
        parser.error('rows > 0; 1 <= cols <= 4096')
    with torch.cuda.device(args.device), torch.inference_mode():
        x = torch.randn((args.rows,args.cols), device=f'cuda:{args.device}', dtype=torch.float32)
        y = row_sum(x)  # compile + validate, outside timing
        ref = torch.empty_like(y)
        torch.sum(x, dim=1, out=ref)
        ref64 = x.double().sum(dim=1)
        tol = 2e-6 * x.double().abs().sum(dim=1) + 1e-5
        for result in (y, ref):
            if not bool(((result.double() - ref64).abs() <= tol).all()):
                raise AssertionError('row-wise FP64 reference error exceeded test tolerance')
        torch.cuda.synchronize()
        print({'torch':torch.__version__, 'triton':triton.__version__,
               'device':torch.cuda.get_device_name(args.device), 'shape':list(x.shape)})
        for name, fn in [('triton', lambda: _launch(x,y)),
                         ('torch', lambda: torch.sum(x,dim=1,out=ref))]:
            p50,p20,p80 = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[.5,.2,.8])
            logical_bytes = 4 * (args.rows*args.cols + args.rows)
            print(name, {'p50_ms':p50,'p20_ms':p20,'p80_ms':p80,
                         'logical_GB_s':logical_bytes/(p50*1e6)})
if __name__ == '__main__': main()
