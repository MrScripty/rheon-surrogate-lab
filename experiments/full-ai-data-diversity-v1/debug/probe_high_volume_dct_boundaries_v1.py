"""DEBUG numerical-teacher oracle: fixed-DCT forward/inverse rounding boundaries.

Native numerical coefficients/eigenvalues are diagnostic references only. No
candidate source, learned coefficient or physical-state prediction is changed.
"""
import argparse,hashlib,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT,read,write,digest
from complete_cuda_graph_v1 import CompleteCudaGraphGenerator
from probe_high_volume_first_step_v1 import difference
import numerics

def forward_trace(q,axes):
    z,y,x=axes;a=torch.einsum('kz,bzyx->bkyx',z,q[:,0]);b=torch.einsum('ly,bkyx->bklx',y,a);c=torch.einsum('mx,bklx->bklm',x,b)
    return [a,b,c]
def inverse_trace(value,axes):
    z,y,x=axes;a=torch.einsum('mx,bklm->bklx',x,value);b=torch.einsum('ly,bklx->bkyx',y,a);c=torch.einsum('kz,bkyx->bzyx',z,b)[:,None]
    return [a,b,c]
def run(args):
    if args.output.exists():raise ValueError('Fresh numerical boundary diagnosis required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('GPU diagnostic backend required')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen boundary source changed: '+path)
    os.sched_setaffinity(0,{6});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=reg['case'];scene=read(ROOT/case['receipt'])['scene'];counts=tuple(scene['counts']);shape=tuple(reversed(counts));spec=next(r for r in case['states'] if r['step']==1)
    if digest(ROOT/spec['path'])!=spec['sha256']:raise ValueError('Actual native first reference changed')
    with np.load(ROOT/spec['path'],allow_pickle=False) as a:
        residual=torch.from_numpy(a['native_input_residual'].copy()).reshape(1,1,*shape);base=torch.from_numpy(a['pressure_base'].copy()).reshape_as(residual)
        expected=torch.from_numpy(a['pressure'].copy()).reshape_as(residual)
    h=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
    model=CompleteCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);model.prepare_geometry(counts);axes_gpu=model.model.geometry_cache[counts][1]
    args.output.mkdir(parents=True)
    try:
        with torch.inference_mode():
            original=base+numerics.correction(residual,h,rho,None);original=original-original[:,:,:1,:1,:1].clone()
            if not torch.equal(original.view(torch.int64),expected.view(torch.int64)):raise ValueError('CPU numerical teacher oracle does not reproduce actual native pressure bytes')
            axes_cpu,eigen=numerics.geometry(shape,None)
            if any(not torch.equal(a.view(torch.int64),b.cpu().view(torch.int64)) for a,b in zip(axes_cpu,axes_gpu)):raise ValueError('Fixed bases differ between teacher/candidate')
            q=residual*(rho/h).reshape(-1,1,1,1,1);qgpu=q.to('cuda');native_forward=forward_trace(q,axes_cpu);gpu_forward=forward_trace(qgpu,axes_gpu)
            # Isolate inverse arithmetic by giving both devices identical native
            # corrected modes; this is a numerical oracle, never candidate data.
            modes=native_forward[-1]/eigen;modes[:,0,0,0]=0.
            native_inverse=inverse_trace(modes,axes_cpu);gpu_inverse=inverse_trace(modes.to('cuda'),axes_gpu)
            corrected_gpu_forward=gpu_forward[-1].cpu()/eigen;corrected_gpu_forward[:,0,0,0]=0.
            native_inverse_from_gpu=inverse_trace(corrected_gpu_forward,axes_cpu)[-1]
            gpu_full_modes=gpu_forward[-1]/eigen.to('cuda');gpu_full_modes[:,0,0,0]=0.
            gpu_full=inverse_trace(gpu_full_modes,axes_gpu)[-1]
            def pressure(value):
                value=base+value.detach().cpu();return value-value[:,:,:1,:1,:1]
            outcomes=dict(cpu_native_forward_and_inverse=difference(original,expected),
                gpu_forward_native_cpu_inverse=difference(pressure(native_inverse_from_gpu),expected),
                native_cpu_forward_gpu_inverse=difference(pressure(gpu_inverse[-1]),expected),
                gpu_forward_and_inverse_with_native_diagonal=difference(pressure(gpu_full),expected))
            # A local inverse-axis probe feeds each next GPU axis the exact CPU
            # previous output, distinguishing axis-local error from propagation.
            z,y,x=axes_gpu
            exact_input_inverse=[torch.einsum('mx,bklm->bklx',x,modes.to('cuda')),
                torch.einsum('ly,bklx->bkyx',y,native_inverse[0].to('cuda')),
                torch.einsum('kz,bkyx->bzyx',z,native_inverse[1].to('cuda'))[:,None]]
            report=dict(status='completed_fixed_dct_cpu_gpu_boundary_diagnosis',all_fixed_basis_bytes_equal=True,cpu_teacher_pressure_bitwise_equal_actual=True,
                forward_axis_error=[difference(a,b) for a,b in zip(gpu_forward,native_forward)],
                inverse_axis_error_from_native_modes=[difference(a,b) for a,b in zip(gpu_inverse,native_inverse)],
                inverse_axis_local_error_exact_native_input=[difference(a,b) for a,b in zip(exact_input_inverse,native_inverse)],pressure_outcomes=outcomes,
                source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),native_numerical_oracles_diagnostic_only=True,
                no_candidate_change=True,learned_cpu_forward=False,optimizer_updates=0,latency_claim=False)
            write(args.output/'results.json',report)
            print('Exact CPU teacher and fixed bases verified; pressure boundary outcomes',outcomes,flush=True)
    finally:model.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
