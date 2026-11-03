"""Independent full128³ cold/warm reconstruction of deployed GCR proposals."""
import argparse,os
from pathlib import Path
from v5_support import ROOT,HERE,DOCS,read,write,sha
from solver_v3 import load,Policy
def run(args):
    if args.output.exists():raise ValueError('Fresh transfer qualification required')
    args.output.parent.mkdir(exist_ok=True,parents=True);os.sched_setaffinity(0,{22})
    import torch,numerics,fast_multilevel,multilevel
    from evaluate_v3 import native
    state,wrapper=load(args.entry)
    import historical_physical
    policy=Policy(state,wrapper,optimized=True,channels_last=True)
    rows=[];cycles=read(ROOT/state.entry['project']/'fine-tuning.json')['coarse_training_cycles']
    with torch.inference_mode():
        for identity,sample in native(state,'development'):
            if sample['base'].shape[2:]!=(128,128,128):continue
            captured=[]
            hook=policy.wrapper.register_forward_hook(lambda module,inputs,out:captured.append((inputs[0].clone(),out.clone())))
            actual,work=policy.solve(sample,('transfer',identity['frame']),max_passes=1);policy.commit(False);hook.remove()
            if len(captured)!=1:raise ValueError('One real complete-proposal network forward required')
            h=sample['cell_size_m'];rho=sample['rho'];initial=sample['rhs']-numerics.fine_action(sample['base'],h,rho)
            coarse=fast_multilevel.correction(initial,h,rho);remaining=initial-numerics.fine_action(coarse,h,rho);zero=torch.zeros_like(initial)
            values=historical_physical.build_inputs(remaining,zero,h,rho,sample['dt'],state.normalization,state.model.owner).contiguous(memory_format=torch.channels_last_3d)
            raw=policy.wrapper(values)
            if not torch.equal(values,captured[0][0])or not torch.equal(raw,captured[0][1]):raise ValueError('Captured raw input/forward differs from independent deployment formula')
            fine=raw.to(torch.float64)*historical_physical.scales(state.normalization)['laplacian_rhs'];fine-=fine[:,:,:1,:1,:1].clone()
            direction=coarse+fine;image=numerics.fine_action(direction,h,rho);length=torch.linalg.vector_norm(image);direction=direction/length;image=image/length
            alpha=torch.sum(initial*image);expected=sample['base']+alpha*direction
            error=float(torch.linalg.vector_norm(actual-expected))/max(float(torch.linalg.vector_norm(expected)),1e-300)
            if error>1e-12:raise ValueError('Complete deployed first GCR iteration differs')
            training_r=sample['native_input_residual']
            if cycles:
                old_coarse=multilevel.correction(training_r,h,rho)
                training_r=training_r-state.model.full_operator(old_coarse,h,rho)
            training_values=historical_physical.build_inputs(training_r,zero,h,rho,sample['dt'],state.normalization,state.model.owner)
            feature_difference=float(torch.linalg.vector_norm(values-training_values))/max(float(torch.linalg.vector_norm(training_values)),1e-300)
            if cycles and feature_difference>1e-8:raise ValueError('Post-coarse training/deployment features diverge')
            rows.append(dict(frame=identity['frame'],counts_xyz=identity['counts_xyz'],raw_input_bitwise_equal=True,raw_forward_bitwise_equal=True,
                complete_first_iteration_relative_difference=error,trained_post_coarse=bool(cycles),training_deployment_feature_relative_difference=feature_difference,
                interpretation='Raw-residual control intentionally trains on a different input distribution.'if not cycles else'Only validated f64 operator/reconstruction rounding separates frozen training preprocessing from deployed numerical preprocessing.'))
    if {r['frame']for r in rows}!={0,7}or torch.cuda.is_initialized():raise ValueError('Both full128³ native snapshots on CPU required')
    write(args.output,dict(status='passed',entry=state.entry,rows=rows,source_sha256=sha(__file__),optimizer_updates=0,
        dependencies={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in ['solver_v3.py','recycling_v3.py','fast_multilevel.py']}))
    print('Qualified full128³ staged proposal and GCR transfer',state.entry['id'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
