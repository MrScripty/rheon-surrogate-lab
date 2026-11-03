"""Check direct-SVD recovery against analytic cases and retained GPU factors."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from support import ROOT,read,write,digest
from capacity_rank_v3 import compress as previous_compress
from capacity_rank_v4 import compress,rank_statistics


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda': raise ValueError('Fresh CUDA-only algebra controls required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h: raise ValueError('Frozen SVD recovery source changed: '+p)
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    result=dict(status='running',analytic_rows=[],retained_factor_rows=[],cpu_model_forward=False,
        model_forward=False,optimizer_updates=0,coefficient_fitting=False,expanded_reserved52_references_opened=0,
        inference_device='cuda',source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),latency_claim=False)
    args.output.mkdir(parents=True)
    with torch.no_grad():
        for name in ('orthogonal','zero','rank_deficient','tiny_reachable','underdetermined','streamed'):
            a=torch.zeros((10,3),device='cuda',dtype=torch.float64);b=torch.zeros(10,device='cuda',dtype=torch.float64)
            expected_rank=3
            if name=='orthogonal':a[:3]=torch.eye(3,device='cuda',dtype=torch.float64);b[3]=1.
            elif name=='zero':b[0]=1.;expected_rank=0
            elif name=='rank_deficient':a[0,:2]=1.;b[1]=1.;expected_rank=1
            elif name=='tiny_reachable':a[0,0]=1.;a[1,1]=1e-18;b[1]=1.;expected_rank=1
            elif name=='underdetermined':
                a=torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]],device='cuda',dtype=torch.float64)
                b=torch.tensor([0.,1.],device='cuda',dtype=torch.float64);expected_rank=2
            else:
                a=torch.arange(150,device='cuda',dtype=torch.float64).reshape(50,3).sin()
                b=torch.arange(50,device='cuda',dtype=torch.float64).cos();expected_rank=2
            local=dict(reg,maximum_design_rows_per_chunk=7)
            old,oldstats=previous_compress(a,b,a.new_tensor(1.),local)
            new,newstats=compress(a,b,a.new_tensor(1.),local)
            if not torch.equal(old,new):raise ValueError('Direct SVD change altered QR factor')
            for k in ('effective_rank_projection_residual','computed_bounded_head_residual_lower_bound'):
                if abs(newstats[k]-oldstats[k])>reg['recovery_absolute_tolerance']:raise ValueError('Analytic recovered diagnostic differs')
            if newstats['effective_rank']!=expected_rank:raise ValueError('Analytic rank control failed: '+name)
            result['analytic_rows'].append(dict(case=name,QR_factor_bits_exact=True,effective_rank=newstats['effective_rank'],bound=newstats['computed_bounded_head_residual_lower_bound']))
        original=read(ROOT/reg['partial_progress'])
        if original['status']!='failed' or 'KeyboardInterrupt' not in original.get('error','') or len(original['rows'])!=3:
            raise ValueError('Retained interrupted three-window run required')
        parts={p['path']:p for p in original['parts']}
        for row in original['rows']:
            p=ROOT/reg['partial_directory']/row['part']
            if digest(p)!=parts[row['part']]['sha256']:raise ValueError('Retained factor changed')
            with np.load(p,allow_pickle=False) as archive:
                for metric in ('equation','tracer'):
                    factor=torch.from_numpy(archive[metric].copy()).to('cuda');old=row['metrics'][metric]
                    new=rank_statistics(factor,(old['full_grid_rows'],old['feature_columns']),reg)
                    if new['effective_rank']!=old['effective_rank']:raise ValueError('Retained factor numerical rank changed')
                    delta=max(abs(new[k]-old[k]) for k in ('effective_rank_projection_residual','computed_bounded_head_residual_lower_bound'))
                    if delta>reg['recovery_absolute_tolerance']:raise ValueError('Recovered retained factor diagnostic changed')
                    result['retained_factor_rows'].append(dict(identifier=row['identifier'],start=row['start'],metric=metric,effective_rank=new['effective_rank'],maximum_absolute_diagnostic_difference=delta))
        result.update(status='passed_direct_SVD_QR_factor_and_rank_recovery_controls',peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(args.output/'results.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
