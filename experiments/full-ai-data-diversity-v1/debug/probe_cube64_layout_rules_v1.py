"""DEBUG generic matrix accumulation patterns on actual and arbitrary arrays."""
import argparse,ctypes as C,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from support import ROOT,read,write,digest
from cuda_kernel_runtime_v1 import CudaKernels,check
from probe_high_volume_dct_boundaries_v1 import forward_trace,inverse_trace
import numerics

def bits(a,b):
    return int((a.detach().cpu().contiguous().reshape(-1).view(torch.int64)!=b.detach().cpu().contiguous().reshape(-1).view(torch.int64)).sum())

def run(args):
    if args.output.exists():raise ValueError('Fresh generic arithmetic controls required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen layout input changed: '+p)
    os.sched_setaffinity(0,{5});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=reg['case'];scene=read(ROOT/case['receipt'])['scene'];shape=tuple(reversed(scene['counts']));nx,ny,nz=reversed(shape)
    spec=next(s for s in case['states'] if s['step']==3)
    with np.load(ROOT/spec['path'],allow_pickle=False) as a:residual=torch.from_numpy(a['native_input_residual'].copy()).reshape(1,1,*shape)
    axes,eigen=numerics.geometry(shape,None);q=residual*(1000./(1/scene['cells_per_metre']))
    cases=[('actual_native3',q,axes,eigen)]
    for seed in reg['arbitrary_seeds']:
        generator=torch.Generator(device='cpu').manual_seed(seed)
        aq=torch.randn((1,1,*shape),dtype=torch.float64,generator=generator)
        aa=tuple(torch.randn((n,n),dtype=torch.float64,generator=generator) for n in shape)
        denom=1+torch.rand(shape,dtype=torch.float64,generator=generator)
        cases.append(('arbitrary_'+str(seed),aq,aa,denom))
    torch.empty(0,device='cuda');kernels=CudaKernels(reg['debug_build']);function=C.c_void_p()
    check(kernels.driver.cuModuleGetFunction(C.byref(function),kernels.module,b'fixed_axis_transform'),'get generic transform')
    kernels.functions['fixed_axis_transform']=function;ptr=lambda v:C.c_void_p(v.data_ptr())
    def contract(a,b,axis,inv,group):
        a=a.to('cuda').reshape(shape).contiguous();b=b.to('cuda').contiguous();out=torch.empty_like(a)
        params=[ptr(a),ptr(b),ptr(out),C.c_int(nx),C.c_int(ny),C.c_int(nz),C.c_int(axis),C.c_int(inv),C.c_int(group),C.c_int(1)]
        kernels.launch('fixed_axis_transform',a.numel(),params);return out
    rows=[];args.output.mkdir(parents=True)
    try:
        with torch.inference_mode():
            for label,query,bases,denom in cases:
                forward=forward_trace(query,bases);modes=forward[-1]/denom;modes[:,0,0,0]=0.;inverse=inverse_trace(modes,bases)
                inputs=[query[:,0],*forward[:2],modes,*inverse[:2]];truths=[*forward,*inverse]
                for i,(src,truth,(axis,inv)) in enumerate(zip(inputs,truths,[(2,False),(1,False),(0,False),(0,True),(1,True),(2,True)])):
                    seq=contract(src,bases[2-axis],axis,inv,1);two=contract(src,bases[2-axis],axis,inv,2)
                    row=torch.arange(seq.numel(),device='cuda',dtype=torch.int64);stride=(1,nx,nx*ny)[axis];n=(nx,ny,nz)[axis]
                    coord=(row//stride)%n
                    span=(nx*ny if axis==2 else nx*nz if axis==1 else ny*nz)
                    pos=(row%(nx*ny) if axis==2 else row%nx+nx*(row//(nx*ny)) if axis==1 else row//nx)
                    cm=((coord>=n-4)&(n%6==4)).reshape(shape);pm=((pos>=span-4)&(span%6==4)).reshape(shape)
                    rules={'sequential':seq,'two_lanes_all':two,'coordinate_tail':torch.where(cm,two,seq),
                           'position_tail':torch.where(pm,two,seq),'current_axis_gt0_tail':torch.where(pm if axis>0 else torch.zeros_like(pm),two,seq)}
                    differences={name:bits(value,truth) for name,value in rules.items()}
                    mismatch=(seq.cpu().contiguous().reshape(-1).view(torch.int64)!=truth.cpu().contiguous().reshape(-1).view(torch.int64)).nonzero().flatten()
                    histogram=torch.bincount(((mismatch//stride)%n),minlength=n)
                    rows.append(dict(case=label,stage=i,axis=axis,inverse=inv,source_shape=list(src.shape),source_stride=list(src.stride()),truth_shape=list(truth.shape),truth_stride=list(truth.stride()),
                        different_bits_by_rule=differences,sequential_mismatch_by_coordinate={str(k):int(v) for k,v in enumerate(histogram) if v}))
                    print(label,'stage',i,differences,flush=True)
        write(args.output/'results.json',dict(status='completed_actual_and_arbitrary_generic_layout_rules',rows=rows,
            numerical_native_controls_only=True,arbitrary_controls_are_not_physical_solvers=True,candidate_changed=False,learned_model_forward=False,cpu_model_forward=False,
            optimizer_updates=0,latency_claim=False,expanded_reserved52_references_opened=0,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
    finally:kernels.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
