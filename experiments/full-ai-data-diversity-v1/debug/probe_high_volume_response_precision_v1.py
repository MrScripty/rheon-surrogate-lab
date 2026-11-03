"""DEBUG: one-variable numerical evaluation of unchanged learned Fourier map."""
import argparse,hashlib,math,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT,read,write,digest
from complete_cuda_graph_v1 import CompleteCudaGraphGenerator
from cuda_native_transfer_v1 import upload_state,download_outputs
from diagnose_causal_composition import inventory,measure
FIELDS=('pressure','vx','vy','vz','tracer')

def response(model,shape,mode):
    if mode=='baseline_gpu_cos':return
    coefficients=model.model.shared_pressure.coefficients
    device='cpu' if mode in ('cpu_fixed_cos','stable_cpu_fixed_trig') else 'cuda'
    if mode=='cpu_fixed_cos':
        z,y,x=torch.meshgrid(*(torch.arange(n,dtype=torch.float64,device='cpu')/n for n in shape),indexing='ij')
        features=torch.stack((torch.ones_like(x),torch.cos(math.pi*x),torch.cos(math.pi*y),torch.cos(math.pi*z)),dim=-1).to('cuda')
        value=model.model.shared_pressure(features)
    else:
        # Generic identity valid for arbitrary affine Fourier coefficients:
        # a0+sum(ai*cos(theta_i)) = (a0+sum(ai))
        #     +sum(ai*(-2*sin(theta_i/2)**2)). No physical coefficient assigned.
        z,y,x=torch.meshgrid(*(torch.arange(n,dtype=torch.float64,device=device) for n in shape),indexing='ij')
        contrast=[-2*torch.sin(math.pi*axis/(2*n)).square() for axis,n in zip((x,y,z),reversed(shape))]
        value=coefficients[0]+coefficients[1]+coefficients[2]+coefficients[3]
        for index,fixed in enumerate(contrast,1):value=value+coefficients[index]*fixed.to('cuda')
    if not bool(value.flatten()[1:].isfinite().all()) or not bool((value.flatten()[1:]>0).all()):raise ValueError('Finite positive restored response required')
    value=value.clone();value[0,0,0]=1.
    model.model.divisors['g'+'_'.join(map(str,shape))]=value

def run(args):
    if args.output.exists():raise ValueError('Fresh precision probe evidence required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('CUDA-only learned mapping evaluation required')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen precision probe source changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=reg['case'];scene=dict(read(ROOT/case['receipt'])['scene'],reference_substeps=case['reference_substeps'],forcing_control_substeps=1)
    spec=next(s for s in case['states'] if s['step']==0)
    with np.load(ROOT/spec['path'],allow_pickle=False) as a:initial={name:torch.from_numpy(a[name].copy()) for name in FIELDS}
    f=read(ROOT/reg['fixture_receipt'])
    with np.load(ROOT/f['reference_path'],allow_pickle=False) as a:truth={name:a[name].copy() for name in (*FIELDS,'rhs')}
    expected=read(ROOT/reg['observation'])['first_failing_K1']['output_sha256'];norm=read(ROOT/reg['normalization'])
    args.output.mkdir(parents=True);rows=[]
    # Generic algebra identity on arbitrary coefficients/phases, independent of
    # the trained values, is checked onGPU before using the stable evaluation.
    torch.manual_seed(reg['seed']);coefficients=torch.randn((2048,4),dtype=torch.float64,device='cuda');theta=torch.rand((2048,3),dtype=torch.float64,device='cuda')*math.pi
    old=coefficients[:,0];stable=coefficients[:,0]+coefficients[:,1]+coefficients[:,2]+coefficients[:,3]
    for i in range(3):old=old+coefficients[:,i+1]*torch.cos(theta[:,i]);stable=stable+coefficients[:,i+1]*(-2*torch.sin(theta[:,i]/2).square())
    scaled=float(((old-stable).abs()/(1+coefficients.abs().sum(1))).max())
    if scaled>reg['generic_identity_maximum_scaled_error']:raise ValueError('Generic arbitrary-coefficient Fourier identity check failed')
    for mode in reg['modes']:
        model=CompleteCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model)
        try:
            with torch.inference_mode():
                current=upload_state(initial);model.prepare_geometry(tuple(scene['counts']));response(model,tuple(reversed(scene['counts'])),mode)
                model.prepare_block(current,scene,0,1,case['reference_substeps'])
                first=None
                for start in range(reg['endpoint_step']):
                    current=model.predict(current,scene,start,1,case['reference_substeps'])[0]
                    if start==0:first={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in download_outputs([current])[0].items()}
                emitted=download_outputs([current])[0];actual={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in emitted.items()}
                if mode=='baseline_gpu_cos' and actual!=expected:raise ValueError('Precision-isolation baseline did not reproduce original endpoint')
                metrics=measure(emitted,truth,scene,norm)
                if inventory(model)!=before:raise ValueError('Probe changed learned coefficients')
                rows.append(dict(mode=mode,metrics=metrics,all_accuracy_pass=all(v<=reg['maximum_error'] for v in metrics['relative'].values()),first_step_output_sha256=first,endpoint_output_sha256=actual,
                    original_endpoint_bitwise_equal=actual==expected))
                print('Unchanged learned response precision',mode,'equation',metrics['relative']['equation'],flush=True)
        finally:model.close()
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Precision probe dependency changed: '+path)
    write(args.output/'results.json',dict(status='completed_unchanged_fourier_response_numerical_evaluation_probe',rows=rows,
        generic_arbitrary_coefficient_identity_maximum_scaled_error=scaled,learned_coefficients_unchanged=True,
        inference_device='cuda',cpu_model_forward=False,optimizer_updates=0,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],latency_claim=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
