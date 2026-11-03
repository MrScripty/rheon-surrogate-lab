"""DEBUG oracle controls localize first velocity mismatch; never candidate use."""
import argparse,hashlib,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from support import ROOT,read,write,digest
from complete_cuda_graph_v1 import CompleteCudaGraphGenerator
from cuda_native_transfer_v1 import upload_state,download_outputs
from recurrent_sampler_native import dimensions
from diagnose_causal_composition import inventory,measure
FIELDS=('pressure','vx','vy','vz','tracer')
def difference(a,b):
    a=a.detach().cpu().double().flatten();b=torch.as_tensor(b).double().flatten();d=a-b
    return dict(relative_l2=float(d.norm()/b.norm().clamp_min(1e-30)),maximum_absolute=float(d.abs().max()),different_values=int((a!=b).sum()))
def run(args):
    if args.output.exists():raise ValueError('Fresh first-step causal diagnostic required')
    reg=read(args.protocol)
    if os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('All learned diagnostic operations requireGPU')
    for path,want in reg['source_sha256'].items():
        if digest(ROOT/path)!=want:raise ValueError('Frozen stage probe changed: '+path)
    os.sched_setaffinity(0,{4});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=reg['case'];scene=dict(read(ROOT/case['receipt'])['scene'],reference_substeps=1,forcing_control_substeps=1);specs={s['step']:s for s in case['states']}
    with np.load(ROOT/specs[0]['path'],allow_pickle=False) as a:initial={name:torch.from_numpy(a[name].copy()) for name in FIELDS}
    with np.load(ROOT/specs[1]['path'],allow_pickle=False) as a:truth={name:a[name].copy() for name in (*FIELDS,'rhs')}
    source=read(ROOT/reg['observed_snapshot']);expected=next(r for r in source['rows'] if r['K']==1 and r['step']==1)['output_sha256']
    model=CompleteCudaGraphGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model);core=model.model
    args.output.mkdir(parents=True)
    try:
        with torch.inference_mode():
            counts=tuple(scene['counts']);shape=tuple(reversed(counts));sizes=dimensions(counts);dt=1/60;h=1/scene['cells_per_metre'];rho=1000.
            core.prepare_geometry(counts);state=upload_state(initial);current=[state[name].reshape(size) for name,size in zip(FIELDS[1:],sizes)];previous=state['pressure'].reshape(1,1,*shape)
            external=model.controls.generate(scene,0,1,1)[0];advected=core.transport(current,counts,dt/h)
            entering=[(v.double()+f).float() for v,f in zip(advected[:3],external)]
            provisional=[core.wall((entering[c].double()+core.linear(entering,size,'viscosity',c)*(.001/rho*dt/h**2)).float(),c) for c,size in enumerate(sizes[:3])]
            rhs=torch.zeros(shape,dtype=torch.float64,device='cuda')
            for channel in range(3):rhs=rhs+(h*h)*core.linear(provisional,shape,'flux',0,channel)
            unscaled_rhs=rhs;rhs=rhs/dt;history=[core.wall(core.linear([previous[0,0]],size,'momentum',c)*((h*h)*(1/(rho*h))),c) for c,size in enumerate(sizes[:3])]
            action=core.linear(history,shape,'flux',0,sign=-1);residual=rhs-action
            predicted=previous+core.pressure((residual*(rho/h)).reshape(1,1,*shape),counts);predicted=predicted-predicted[:,:,:1,:1,:1]
            oracle=torch.from_numpy(truth['pressure']).to('cuda').reshape_as(predicted)
            native_rhs=torch.from_numpy(truth['rhs']).to('cuda').reshape(shape)
            tensor_div_rhs=unscaled_rhs/torch.full((),dt,dtype=torch.float64,device='cuda')
            numpy_div_rhs=torch.from_numpy(unscaled_rhs.cpu().numpy()/np.float64(dt)).to('cuda')
            def pressure_from_rhs(value):
                p=previous+core.pressure(((value-action)*(rho/h)).reshape(1,1,*shape),counts)
                return p-p[:,:,:1,:1,:1]
            rhs_comparisons={name:difference(value,truth['rhs']) for name,value in [('old_scalar_division',rhs),('gpu_tensor_division',tensor_div_rhs),('numpy_reference_division',numpy_div_rhs)]}
            rows=[]
            for mode,pressure in [('unchanged_real_path',predicted),('gpu_tensor_division_only',pressure_from_rhs(tensor_div_rhs)),('numpy_reference_division_only',pressure_from_rhs(numpy_div_rhs)),('native_rhs_oracle_only',pressure_from_rhs(native_rhs)),('native_pressure_oracle_only',oracle)]:
                velocity=[core.wall((provisional[c].double()+core.linear([pressure[0,0]],size,'momentum',c)*(dt*(1/(rho*h)))).float(),c) for c,size in enumerate(sizes[:3])]
                output=dict(pressure=pressure.contiguous().flatten(),vx=velocity[0].flatten(),vy=velocity[1].flatten(),vz=velocity[2].flatten(),tracer=advected[3].flatten())
                copied=download_outputs([output])[0];identity={name:hashlib.sha256(v.numpy().tobytes()).hexdigest() for name,v in copied.items()}
                if mode=='unchanged_real_path' and identity!=expected:raise ValueError('Stage diagnostic does not exactly match actual real first-step fields')
                rows.append(dict(mode=mode,metrics=measure(copied,truth,scene,read(ROOT/reg['normalization'])),field_sha256=identity,
                    velocity_fields_vs_native={name:difference(copied[name],truth[name]) for name in ('vx','vy','vz')}))
            if inventory(model)!=before:raise ValueError('Probe changed learned weights')
            write(args.output/'results.json',dict(status='completed_first_step_rhs_division_and_pressure_isolation',rows=rows,rhs_comparisons=rhs_comparisons,
                candidate_rhs_vs_actual_native_rhs=difference(rhs,truth['rhs']),predicted_pressure_vs_native=difference(predicted,truth['pressure']),
                oracle_uses_native_future_pressure_for_diagnosis_only=True,candidate_or_deployment_change=False,latency_claim=False,
                optimizer_updates=0,cpu_model_forward=False,inference_device='cuda',protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256']))
            print('RHS precision comparisons',rhs_comparisons,flush=True)
            for row in rows:print(row['mode'],'velocity error',row['metrics']['relative']['velocity'],flush=True)
    finally:model.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
