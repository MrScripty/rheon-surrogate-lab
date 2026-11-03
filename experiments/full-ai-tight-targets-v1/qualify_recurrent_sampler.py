"""Check untrained native/tensor neural graphs and inherited quantized tensors."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT,read,write,digest
from recurrent_sampler_model import RecurrentSamplerPrototype
from recurrent_sampler_native import transport,spatial,forcing,dimensions
import recurrent_sampler_reference as reference
from double_prescribed_force import force_features
from diagnose_causal_composition import inventory
import numpy as np
import torch


def relative(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))


def reference_predict(model,fields,scene,start,horizons):
    counts=tuple(scene['counts']);current=[torch.as_tensor(fields[name]).reshape(shape)for name,shape in zip(('vx','vy','vz','tracer'),dimensions(counts))]
    external=force_features(scene,start,horizons);dt=1./60.;h=1./scene['cells_per_metre'];rho=1000.;outputs=[]
    momentum=-model.flux.flip(dims=(2,3,4)).transpose(0,1).contiguous()
    for lead in range(horizons):
        advected=reference.transport(current,counts,model.point,model.mid_gate,model.mix,dt/h)
        entered=[(value.double()+external[0,lead,c,:value.shape[0],:value.shape[1],:value.shape[2]]).float()for c,value in enumerate(advected[:3])]
        provisional=reference.sparse_stage(0,entered,counts,model.viscosity,.001/rho*dt/h**2,base=entered)
        q=reference.sparse_stage(1,provisional,counts,model.flux,rho*h/dt);q=q-q.mean()
        pressure=model.pressure_core(q.reshape(1,1,*q.shape))
        velocity=reference.sparse_stage(2,[pressure[0,0]],counts,momentum,dt/(rho*h),base=provisional)
        outputs.append(dict(pressure=pressure.flatten(),vx=velocity[0].flatten(),vy=velocity[1].flatten(),vz=velocity[2].flatten(),tracer=advected[3].flatten()))
        current=[*velocity,advected[3]]
    return outputs


def run(args):
    if args.output.exists():raise ValueError('Fresh untrained graph qualification required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered sampler source changed: '+path)
    if digest(reg['library_path'])!=reg['library_sha256']:raise ValueError('Registered pressure backend changed')
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=[];sources={};models=[]
    with torch.inference_mode():
        for initializer in reg['initializers']:
            model=RecurrentSamplerPrototype(reg['library_path'],initializer,reg['seed']);before=inventory(model)
            models.append(dict(initializer=initializer,inventory=before,assets=model.asset_identity,
                               learned_tensor_count=sum(p.numel()for p in model.parameters()),new_trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad)))
            try:
                for trajectory in read(ROOT/reg['bank'])['trajectories']:
                    path=ROOT/trajectory['receipt']
                    if digest(path)!=trajectory['receipt_sha256']:raise ValueError('Consumed development receipt changed')
                    receipt=read(path);scene=receipt['scene'];counts=tuple(scene['counts']);sources[str(path.relative_to(ROOT))]=digest(path)
                    for start in reg['starts']:
                        spec=receipt['states'][start];state=path.parent/spec['path']
                        if digest(state)!=spec['sha256']:raise ValueError('Current physical state changed')
                        sources[str(state.relative_to(ROOT))]=digest(state)
                        with np.load(state,allow_pickle=False)as archive:fields={name:archive[name].copy()for name in ('pressure','vx','vy','vz','tracer')}
                        carriers=[torch.from_numpy(fields[name]).reshape(shape)for name,shape in zip(('vx','vy','vz','tracer'),dimensions(counts))]
                        expected=reference.transport(carriers,counts,model.point,model.mid_gate,model.mix,scene['cells_per_metre']/60.)
                        actual=transport(carriers,counts,model.coefficients['point'],model.mid_gate,model.mix,scene['cells_per_metre']/60.)
                        transport_error=max(relative(a,b)for a,b in zip(actual,expected))
                        # Quantization is tested as an explicit arithmetic
                        # policy, never assumed exact from coefficient size.
                        with np.load(ROOT/model.asset_identity['parent_pair']/'restore/weights.npz',allow_pickle=False)as archive:
                            raw=torch.from_numpy(archive['displacement.point_weight'].copy()).double()
                        raw_mix=raw.sum(dim=(2,3,4)).reshape(4,3,3)
                        if initializer=='seeded':raw_mix=raw_mix+(model.mix-model.point.sum(dim=(2,3,4)).reshape(4,3,3))
                        unquantized=reference.transport(carriers,counts,raw,model.mid_gate,raw_mix,scene['cells_per_metre']/60.)
                        quantization_error=max(relative(a,b)for a,b in zip(expected,unquantized))
                        force=forcing(scene,start,4);force_reference=force_features(scene,start,4)
                        force_error=max(relative(value,force_reference[0,k,c,:value.shape[0],:value.shape[1],:value.shape[2]])for k,group in enumerate(force)for c,value in enumerate(group))
                        native_outputs=model.predict(fields,scene,start,4)
                        tensor_outputs=reference_predict(model,fields,scene,start,4)
                        complete={name:max(relative(a[name],b[name])for a,b in zip(native_outputs,tensor_outputs))for name in fields}
                        finite=all(bool(torch.isfinite(value).all())for output in native_outputs for value in output.values())
                        if not finite or transport_error>reg['maximum_transport_relative_error']or quantization_error>reg['maximum_quantization_relative_error']or force_error>reg['maximum_force_relative_error']or max(complete.values())>reg['maximum_complete_execution_relative_error']:
                            raise ValueError('Neural graph execution qualification failed: '+str((initializer,trajectory['identifier'],start,transport_error,quantization_error,force_error,complete)))
                        rows.append(dict(initializer=initializer,identifier=trajectory['identifier'],start=start,counts_xyz=scene['counts'],
                            transport_relative_l2=transport_error,quantization_relative_l2=quantization_error,known_force_relative_l2=force_error,
                            complete_four_step_execution_relative_l2=complete,complete_states=4,all_outputs_finite=True,
                            transport_bytes_exact=all(torch.equal(a,b)for a,b in zip(actual,expected))))
                    print(initializer,trajectory['identifier'],'neural graph comparisons passed',flush=True)
                if inventory(model)!=before:raise ValueError('Untrained graph qualification changed parameters')
            finally:model.close()
    if len(rows)!=reg['expected_cases']or torch.cuda.is_initialized():raise ValueError('Finite qualification population or device changed')
    write(args.output,dict(status='passed_untrained_neural_graph_execution',rows=rows,models=models,source_evidence=sources,
        source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),optimizer_updates=0,new_parameters_fitted=False,
        native_backtrace_called_in_candidate=False,analytic_inverse_constructed=False,all_plan_owners_closed=True,cuda_initialized=False,
        truth_future_states_used=False,fresh_holdout_consumed=False,model_accuracy_qualified=False,candidate_frontier_point=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
