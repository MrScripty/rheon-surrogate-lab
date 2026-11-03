"""DEBUG ONLY: matched native2->3 request and generic transform rounding.

All learned operations stay on CUDA. CPU numerical geometry/correction and
native intermediate arrays are isolated teacher controls, never candidate inputs
except the actual native CURRENT state2. No candidate or coefficients change.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT, read, write, digest
from debug.affine_generator_graph_v2 import AffineGeneratorGraph
from cuda_native_transfer_v1 import FIELDS, upload_state, download_outputs
from diagnose_causal_composition import inventory, measure
from probe_late_first_divergence_v1 import stages
from probe_high_volume_dct_boundaries_v1 import forward_trace, inverse_trace
from probe_high_volume_first_step_v1 import difference
import numerics


def bits(a,b):
    a=a.detach().cpu().contiguous().reshape(-1);b=torch.as_tensor(b).contiguous().reshape(-1)
    return int((a.view(torch.int64)!=b.view(torch.int64)).sum())


def run(args):
    if args.output.exists():raise ValueError('Fresh boundary evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU-only learned diagnostic required')
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen source/fixture changed: '+p)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=reg['case'];scene=read(ROOT/case['receipt'])['scene'];counts=tuple(scene['counts']);shape=tuple(reversed(counts))
    specs={r['step']:r for r in case['states']}
    with np.load(ROOT/specs[2]['path'],allow_pickle=False) as a:initial={n:torch.from_numpy(a[n].copy()) for n in FIELDS}
    with np.load(ROOT/specs[3]['path'],allow_pickle=False) as a:truth={n:a[n].copy() for n in a.files}
    observation=read(ROOT/reg['observation']);expected=next(r['output_sha256'] for r in observation['rows'] if r['K']==1 and r['step']==3)
    previous_reference=next(r['actual_field_sha256'] for r in observation['reference_rows'] if r['step']==2)
    previous_output=next(r['output_sha256'] for r in observation['rows'] if r['K']==1 and r['step']==2)
    if any(previous_reference[n]!=previous_output[n] for n in FIELDS):raise ValueError('Matched-state minimization requires exact actual state2')
    args.output.mkdir(parents=True)
    model=AffineGeneratorGraph(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model)
    try:
        with torch.inference_mode():
            model.prepare_geometry(counts);core=model.model;core.configure_transport(1/scene['cells_per_metre'],1/60)
            state=upload_state(initial);external=model.controls.generate(scene,2,1,1)[0]
            rhs,residual,pressure,output=stages(core,state,external,scene)
            emitted=download_outputs([output(pressure)])[0]
            hashes={n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in emitted.items()}
            if hashes!=expected:raise ValueError('Matched-state stage decomposition differs from actual failing history step3')
            axes_cpu,eigen=numerics.geometry(shape,None);axes_gpu=core.geometry_cache[counts][1]
            if any(bits(a,b) for a,b in zip(axes_cpu,axes_gpu)):raise ValueError('Candidate fixed bases differ from native bases')
            h=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
            native_residual=torch.from_numpy(truth['native_input_residual']).reshape(1,1,*shape)
            native_base=torch.from_numpy(truth['pressure_base']).reshape_as(native_residual)
            native_pressure=torch.from_numpy(truth['pressure']).reshape_as(native_residual)
            native=native_base+numerics.correction(native_residual,h,rho,None);native=native-native[:,:,:1,:1,:1].clone()
            if bits(native,native_pressure):raise ValueError('Numerical teacher control does not exactly reproduce native pressure3')
            qcpu=native_residual*(rho/h).reshape(-1,1,1,1,1)
            qgpu=(residual*(1000./(1/scene['cells_per_metre']))).reshape_as(qcpu)
            forward=forward_trace(qcpu,axes_cpu);modes=forward[-1]/eigen;modes[:,0,0,0]=0.;inverse=inverse_trace(modes,axes_cpu)
            src=[qcpu[:,0],*forward[:2],modes,*inverse[:2]];ref=[*forward,*inverse]
            layout=[(2,False),(1,False),(0,False),(0,True),(1,True),(2,True)]
            local=[]
            for i,(a,b,(axis,inv)) in enumerate(zip(src,ref,layout)):
                got=model.fixed_transform.contract(a.to('cuda').reshape(shape),axes_gpu[2-axis],axis,inv)
                local.append(dict(stage=i,axis=axis,inverse=inv,different_bits=bits(got,b),difference=difference(got,b)))
            learned_divisor=core.divisors['g'+'_'.join(map(str,shape))]
            qpressure=core.pressure(qcpu.to('cuda'),counts)
            from_native_q=native_base.to('cuda')+qpressure;from_native_q=from_native_q-from_native_q[:,:,:1,:1,:1]
            q_mode=forward[-1].to('cuda')/learned_divisor;q_mode[:,0,0,0]=0.
            native_division=modes
            # Native transformed modes are oracle controls only, not candidate state.
            value=qcpu.to('cuda').reshape(shape)
            own_forward=[]
            for axis in (2,1,0):
                value=model.fixed_transform.contract(value,axes_gpu[2-axis],axis,False);own_forward.append(value)
            own_modes=value/learned_divisor;own_modes[0,0,0]=0.
            own_inverse=[]
            for axis in (0,1,2):
                own_modes=model.fixed_transform.contract(own_modes,axes_gpu[2-axis],axis,True);own_inverse.append(own_modes)
            report=dict(status='completed_matched_native2to3_request_and_fixed_transform_diagnosis',actual_full_history_step3_exact=True,actual_native_state2_exact=True,
                stage3_output_sha256=hashes,metrics=measure(emitted,truth,scene,read(ROOT/reg['normalization'])),
                rhs_vs_native=difference(rhs,truth['rhs']),rhs_different_bits=bits(rhs,truth['rhs']),
                residual_vs_native=difference(residual,truth['native_input_residual']),residual_different_bits=bits(residual,truth['native_input_residual']),
                scaled_request_different_bits=bits(qgpu,qcpu),base_different_bits=bits(state['pressure'],truth['pressure_base']),
                all_fixed_basis_bytes_equal=True,native_teacher_pressure_exact=True,
                learned_divisor_vs_native=difference(learned_divisor,eigen),learned_divisor_different_bits=bits(learned_divisor,eigen),
                local_axis_arithmetic=local,own_forward_different_bits=[bits(a,b) for a,b in zip(own_forward,forward)],
                own_inverse_different_bits=[bits(a,b) for a,b in zip(own_inverse,inverse)],
                division_same_native_forward_different_bits=bits(q_mode,native_division),
                restored_pressure_from_exact_native_q_vs_native=difference(from_native_q,native_pressure),
                restored_pressure_from_exact_native_q_different_bits=bits(from_native_q,native_pressure),
                cpu_model_forward=False,optimizer_updates=0,native_numerical_oracles_debug_only=True,candidate_changed=False,latency_claim=False,
                expanded_reserved52_references_opened=0,inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
                protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'])
        if inventory(model)!=before:raise ValueError('Diagnostic changed learned weights')
        write(args.output/'results.json',report)
        print('Actual state2->3 reproduced; RHS/residual/request differing bits',report['rhs_different_bits'],report['residual_different_bits'],report['scaled_request_different_bits'],flush=True)
        print('Learned divisor differing bits',report['learned_divisor_different_bits'],'local transform axis differing bits',[r['different_bits'] for r in local],flush=True)
    finally:model.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
