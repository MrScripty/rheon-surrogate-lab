"""DEBUG: one-variable numerical evaluation of unchanged learned Fourier map."""
import argparse,hashlib,math,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT,read,write,digest
from complete_cuda_graph_v1 import CompleteCudaGraphGenerator
from fused_cuda_generator_v1 import FusedCudaGenerator
from dynamic_cuda_controls_v1 import CudaKnownControls
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import upload_state,download_outputs
from diagnose_causal_composition import inventory,measure
FIELDS=('pressure','vx','vy','vz','tracer')

class PreciseRhsFusedGenerator(FusedCudaGenerator):
    @torch.inference_mode()
    def predict(self, fields, scene, step, horizons, external, substeps=1):
        counts = tuple(scene['counts']); shape = tuple(reversed(counts))
        if (horizons not in (1, 2, 3, 4) or substeps not in (1, 8) or not 0 < scene['cells_per_metre'] <= 1024
                or scene.get('sgs', False) or counts not in self.geometry_cache):
            raise ValueError('Prepared registered reference-regime geometry required')
        if set(fields) != set(FIELDS) or len(external) != horizons*substeps:
            raise ValueError('Complete current state and prescribed input block required')
        for name, expected_shape in [('pressure', shape), *zip(FIELDS[1:], dimensions(counts))]:
            value = fields[name]
            if (value.device != self.device or value.dtype != (torch.float64 if name == 'pressure' else torch.float32)
                    or value.numel() != math.prod(expected_shape) or not value.is_contiguous()):
                raise ValueError('Exact native dtype and contiguous CUDA state required')
        for block in external:
            if len(block) != 3 or any(v.device != self.device or v.dtype != torch.float64
                                    or tuple(v.shape) != size for v, size in zip(block, dimensions(counts)[:3])):
                raise ValueError('Complete CUDA double prescribed velocities required')
        current = [fields[name].reshape(size) for name, size in zip(FIELDS[1:], dimensions(counts))]
        previous = fields['pressure'].reshape(1, 1, *shape)
        dt, h, rho = 1/(60*substeps), 1/scene['cells_per_metre'], 1000.
        results = []
        for lead in range(horizons*substeps):
            advected = self.transport(current, counts, dt/h)
            entering = [(value.double()+addition).float() for value, addition in zip(advected[:3], external[lead])]
            provisional = [self.wall((entering[c].double()+self.linear(entering, size, 'viscosity', c)
                                      *(.001/rho*dt/h**2)).float(), c)
                           for c, size in enumerate(dimensions(counts)[:3])]
            rhs = torch.zeros(shape, dtype=torch.float64, device=self.device)
            for channel in range(3):
                rhs = rhs+(h*h)*self.linear(provisional, shape, 'flux', 0, channel)
            rhs = rhs/torch.full((),dt,dtype=torch.float64,device=self.device)
            history = []
            scale = (h*h)*(1/(rho*h))
            for component, size in enumerate(dimensions(counts)[:3]):
                history.append(self.wall(self.linear([previous[0, 0]], size, 'momentum', component)*scale, component))
            action = self.linear(history, shape, 'flux', 0, sign=-1)
            residual = rhs-action
            correction = self.pressure((residual*(rho/h)).reshape(1, 1, *shape), counts)
            pressure = previous+correction
            pressure = pressure-pressure[:, :, :1, :1, :1]
            velocity = [self.wall((provisional[c].double()+self.linear([pressure[0, 0]], size, 'momentum', c)
                                   *(dt*(1/(rho*h)))).float(), c)
                        for c, size in enumerate(dimensions(counts)[:3])]
            if (lead+1) % substeps == 0:
                results.append(dict(pressure=pressure.contiguous().flatten(),
                    vx=velocity[0].flatten(), vy=velocity[1].flatten(), vz=velocity[2].flatten(), tracer=advected[3].flatten()))
            current, previous = [*velocity, advected[3]], pressure
        return results


class ProbeGraph(CompleteCudaGraphGenerator):
    def __init__(self,pair,shared_pair,build,control_build,precise):
        self.model=(PreciseRhsFusedGenerator if precise else FusedCudaGenerator)(pair,shared_pair,build,'transport_and_linear')
        self.controls=CudaKnownControls(control_build)
        self.asset_identity=dict(self.model.asset_identity,execution='diagnostic unchanged learned map exact-evaluation probe')
        self.graphs={};self.setup_observations=[];self.pending_packets=[]

def response(model,shape,stable):
    if not stable:return
    c=model.model.shared_pressure.coefficients
    terms=[]
    # One-dimensional fixed trig features, Z/Y/X accumulation. All four
    # arbitrary learned coefficients remain GPU values; -2 is solely the
    # generic cos(theta)-1 identity factor, never a physical stencil value.
    for n,index in zip(shape,(3,2,1)):
        phase=math.pi*torch.arange(n,dtype=torch.float64)/(2*n)
        fixed=-2*torch.sin(phase).square()
        terms.append(c[index]*fixed.to('cuda'))
    dc=c[0]+c[1]+c[2]+c[3]
    value=((dc+terms[0][:,None,None])+terms[1][None,:,None])+terms[2][None,None,:]
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
        model=ProbeGraph(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build'],mode in ('precise_rhs_only','stable_response_and_precise_rhs'));before=inventory(model)
        try:
            with torch.inference_mode():
                current=upload_state(initial);model.prepare_geometry(tuple(scene['counts']));response(model,tuple(reversed(scene['counts'])),mode in ('stable_response_only','stable_response_and_precise_rhs'))
                model.prepare_block(current,scene,0,1,case['reference_substeps'])
                first=None
                for start in range(reg['endpoint_step']):
                    current=model.predict(current,scene,start,1,case['reference_substeps'])[0]
                    if start==0:first={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in download_outputs([current])[0].items()}
                emitted=download_outputs([current])[0];actual={name:hashlib.sha256(value.numpy().tobytes()).hexdigest() for name,value in emitted.items()}
                if mode=='baseline' and actual!=expected:raise ValueError('Precision-isolation baseline did not reproduce original endpoint')
                metrics=measure(emitted,truth,scene,norm)
                if inventory(model)!=before:raise ValueError('Probe changed learned coefficients')
                rows.append(dict(mode=mode,metrics=metrics,all_accuracy_pass=all(v<=reg['maximum_error'] for v in metrics['relative'].values()),first_step_output_sha256=first,endpoint_output_sha256=actual,
                    original_endpoint_bitwise_equal=actual==expected))
                print('Exact unchanged learned-evaluation probe',mode,'equation',metrics['relative']['equation'],flush=True)
        finally:model.close()
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Precision probe dependency changed: '+path)
    write(args.output/'results.json',dict(status='completed_generic_stable_axis_and_precise_rhs_factorial_probe',rows=rows,
        generic_arbitrary_coefficient_identity_maximum_scaled_error=scaled,learned_coefficients_unchanged=True,
        inference_device='cuda',cpu_model_forward=False,optimizer_updates=0,protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'],latency_claim=False))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
