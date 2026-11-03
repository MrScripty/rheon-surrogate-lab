"""Full128³ cold/warm captured local graph and independent first-GCR transfer."""
import argparse,os
from pathlib import Path
from v6_support import ROOT,HERE,BASE,read,write,sha
from solver import load,Policy
def run(args):
    if args.output.exists():raise ValueError('Fresh deployment-transfer result required')
    args.output.parent.mkdir(parents=True,exist_ok=True);os.sched_setaffinity(0,{22})
    state,network=load(args.entry);t=state.torch;t.set_num_threads(1)
    import historical_physical,numerics,multilevel
    from evaluate_v3 import native
    population=[(i,s)for i,s in native(state,'development')if i['counts_xyz']==[128]*3]
    if len(population)!=2:raise ValueError('Two existing full128³ cold/warm requests required')
    rows=[];scale=historical_physical.scales(state.normalization)['laplacian_rhs']
    with t.inference_mode():
        for meta,s in population:
            initial=s['rhs']-numerics.fine_action(s['base'],s['cell_size_m'],s['rho'])
            coarse=multilevel.correction(initial,s['cell_size_m'],s['rho']);r=initial-numerics.fine_action(coarse,s['cell_size_m'],s['rho'])
            q=((r*(s['rho']/s['cell_size_m']).reshape(1,1,1,1,1))/scale).to(t.float32);raw=network(q)
            expected=raw.to(t.float64)*scale;expected=expected-expected[:,:,:1,:1,:1].clone()
            actual=state.model.predict_pressure(network,{**s,'base':t.zeros_like(r),'native_input_residual':r},state.normalization)
            if not t.equal(expected,actual):raise ValueError('Captured local inference differs from independently reconstructed features/graph/scale')
            proposal=coarse+actual;image=numerics.fine_action(proposal,s['cell_size_m'],s['rho'])
            length=t.linalg.vector_norm(image);direction=proposal/length;unit_image=image/length
            manual=s['base']+(initial*unit_image).sum()*direction
            policy=Policy(state,network,optimized=True)
            p,work=policy.solve(s,'transfer',max_passes=1);policy.commit(False)
            relative=float(t.linalg.vector_norm(p-manual))/max(float(t.linalg.vector_norm(manual)),1e-300)
            if relative>1e-12:raise ValueError('Complete first GCR iteration differs from independent construction')
            native_rows=[]
            if sum(p.numel()for p in network.parameters())==648:
                from native_local import NativeLocal
                for streaming in [False,True]:
                    local=NativeLocal(network,state.normalization,streaming=streaming);out=local(r,s['cell_size_m'],s['rho']);local.close()
                    delta=float(t.linalg.vector_norm(out-actual))/max(float(t.linalg.vector_norm(actual)),1e-300)
                    if delta>2e-6:raise ValueError('Fitted native stage differs from captured graph')
                    native_rows.append(dict(streaming=streaming,relative_l2_difference=delta))
            rows.append(dict(**meta,raw_local_bitwise_equal=True,complete_first_iteration_relative_difference=relative,native=native_rows))
    write(args.output,dict(status='passed',entry=state.entry,rows=rows,source_sha256=sha(__file__),solver_sha256=sha(HERE/'solver.py'),optimizer_updates=0))
    print('Full128³ cold/warm deployed first-iteration transfer passed',state.entry['id'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
