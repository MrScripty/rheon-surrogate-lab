"""GPU component equivalence for generic fused learned gathers/readouts."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
from support import ROOT,read,write,digest
from debug.gather_fused_graph_v1 import GatherFusedGraph
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists():raise ValueError('Fresh fused component evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU component inference only')
    for path,h in reg['source_sha256'].items():
        if digest(ROOT/path)!=h:raise ValueError('Frozen component dependency changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(reg['seed'])
    args.output.mkdir(parents=True);model=GatherFusedGraph(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model);core=model.model;rows=[]
    def compare(label,actual,expected):
        if len(actual)!=len(expected):raise ValueError('Complete component outputs required')
        result=[]
        for a,b in zip(actual,expected):
            if a.dtype!=b.dtype or a.shape!=b.shape:raise ValueError('Component dtype/shape changed')
            dtype=torch.int64 if a.dtype==torch.float64 else torch.int32
            result.append(int((a.contiguous().view(dtype)!=b.contiguous().view(dtype)).sum()))
        return dict(component=label,different_bits_by_output=result)
    try:
        with torch.inference_mode():
            for case in reg['cases']:
                scene=read(ROOT/case['receipt'])['scene'];counts=tuple(scene['counts']);shape=tuple(reversed(counts));sizes=dimensions(counts)[:3]
                spec=case['primitive_state'];h=1/scene['cells_per_metre'];dt=1/(60*case['reference_substeps']);rho=1000.
                with np.load(ROOT/spec['path'],allow_pickle=False) as a:
                    native_v=[torch.from_numpy(a[n].copy()).to('cuda').reshape(s) for n,s in zip(['vx','vy','vz'],sizes)]
                    native_p=torch.from_numpy(a['pressure'].copy()).to('cuda').reshape(shape)
                for pattern in ['actual_current','arbitrary_current']:
                    fields=native_v if pattern=='actual_current' else [torch.randn(s,dtype=torch.float32,device='cuda') for s in sizes]
                    p=native_p if pattern=='actual_current' else torch.randn(shape,dtype=torch.float64,device='cuda')
                    viscosity=[core.linear(fields,s,'viscosity',c) for c,s in enumerate(sizes)]
                    ref_v=model.affine_ops.three(fields,viscosity,counts,.001/rho*dt/h**2,wall=True)
                    got_v=model.gather_ops.three(core,fields,fields,counts,'viscosity',.001/rho*dt/h**2)
                    flux=[core.linear(ref_v,shape,'flux',0,c) for c in range(3)]
                    ref_rhs=model.affine_ops.ordered_three(flux,h*h,dt)
                    got_rhs=model.gather_ops.ordered_rhs(core,ref_v,shape,'flux',0,h*h,dt)
                    momentum=[core.linear([p],s,'momentum',c) for c,s in enumerate(sizes)]
                    ref_h=model.affine_ops.three(None,momentum,counts,(h*h)*(1/(rho*h)),multiply_only=True,as_double=True,wall=True)
                    got_h=model.gather_ops.three(core,[p],None,counts,'momentum',(h*h)*(1/(rho*h)),multiply_only=True,as_double=True)
                    action=core.linear(ref_h,shape,'flux',0,sign=-1.)
                    ref_q=model.affine_ops.binary('difference_scale',ref_rhs,action,rho/h)
                    got_q=model.gather_ops.difference(core,ref_h,ref_rhs,shape,'flux',0,-1.,rho/h)
                    ref_out=model.affine_ops.three(ref_v,momentum,counts,dt*(1/(rho*h)),wall=True)
                    got_out=model.gather_ops.three(core,[p],ref_v,counts,'momentum',dt*(1/(rho*h)))
                    comparisons=[compare('three_float_affine',got_v,ref_v),compare('ordered_three_channel_rhs',[got_rhs],[ref_rhs]),
                        compare('three_double_multiply',got_h,ref_h),compare('learned_difference_scale',[got_q],[ref_q]),compare('three_float_projection',got_out,ref_out)]
                    rows.append(dict(identifier=case['identifier'],counts_xyz=list(counts),cells_per_metre=scene['cells_per_metre'],internal_substeps=case['reference_substeps'],pattern=pattern,source_state=spec,comparisons=comparisons))
                    print('Generic gather readout components',case['identifier'],pattern,'different',sum(sum(r['different_bits_by_output']) for r in comparisons),flush=True)
        unchanged=inventory(model)==before
        passed=unchanged and len(rows)==52 and all(not any(r['different_bits_by_output']) for row in rows for r in row['comparisons'])
        write(args.output/'results.json',dict(status='passed_all26_generic_gather_readout_components' if passed else 'failed_generic_gather_readout_components',rows=rows,
            compared_output_arrays=572,every_component_output_bitwise_equal=passed,learned_weights_unchanged=unchanged,
            inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),cpu_model_forward=False,optimizer_updates=0,latency_claim=False,
            expanded_reserved52_references_opened=0,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
        if not passed:raise AssertionError('Fused gather component bits/weights changed')
    finally:model.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
