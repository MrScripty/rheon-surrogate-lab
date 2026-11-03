"""A real returned-field regression, distinct from the internal increment."""
import argparse,importlib,sys
from v5_support import ROOT,read,write,sha
def run(args):
    import torch,numerics
    from solver import load
    state,wrapper=load(ROOT/'runs/pressure-methods-v5/entries/v5-raw.json')
    module=importlib.import_module(args.module)
    shape=(1,1,16,16,16);base=torch.full(shape,1e20,dtype=torch.float64);base[0,0,0,0,0]=0
    exact=torch.zeros_like(base);exact[0,0,8,8,8]=1000.;exact[0,0,8,8,9]=-1000.
    h=torch.tensor([.125],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64);dt=torch.tensor([1/60],dtype=torch.float64)
    residual=numerics.fine_action(exact,h,rho);rhs=residual+numerics.fine_action(base,h,rho)
    if not torch.equal(rhs-numerics.fine_action(base,h,rho),residual):raise ValueError('Test equation input is inconsistent')
    sample=dict(base=base,native_input_residual=residual,rhs=rhs,reference=base+exact,cell_size_m=h,rho=rho,dt=dt)
    policy=module.Policy(state,wrapper,neural=False)
    with torch.inference_mode():pressure,work=policy.solve(sample,'large-base-representability')
    policy.commit(False)
    returned_residual=residual-numerics.fine_action(pressure-base,h,rho)
    norm=float(torch.linalg.vector_norm(returned_residual));denominator=max(float(torch.linalg.vector_norm(rhs)),1e-300)
    divergence=float(dt[0]*returned_residual.abs().max()/h[0]**3)
    actual=norm<=max(1e-12,denominator*1e-6)and divergence<=1e-7
    passed=bool(work['qualified'])==actual
    write(args.output,dict(status='passed'if passed else'failed_returned_pressure_gate',module=args.module,reported_qualified=work['qualified'],
        returned_pressure_qualified=actual,returned_predicted_divergence=divergence,reported_predicted_divergence=work['predicted_divergence'],
        equation_input_exact=True,pressure_bitwise_unchanged=torch.equal(pressure,base),optimizer_updates=0,
        source_sha256=sha(__file__),policy_source_sha256=sha(module.__file__)))
    print('Returned-field gate:',passed,'reported',work['qualified'],'actual',actual,'divergence',divergence,flush=True)
    if args.require and not passed:raise AssertionError('Reported qualification does not describe the actual returned float64 field')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--module',default='solver');p.add_argument('--output',required=True,type=__import__('pathlib').Path);p.add_argument('--require',action='store_true');run(p.parse_args())
