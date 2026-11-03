"""Offline teacher-injection/gradient qualification, never model inference.

Fine endpoint pressure and the LAST fine provisional velocity define a coarse
provisional target via a learned momentum map and timestep conversion. This
checks that local residual supervision can express compatible complete states.
No network is fitted and teacher-derived corrections cannot be deployed.
"""
import argparse
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from coupled_macrostep_v2 import CoupledMacroBackbone
from coupled_macrostep_v1 import wall
from diagnose_group_transition_v1 import Transitions, hashes, comparison, vector
from diagnose_causal_composition import inventory
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS


def provisional(output, increments, counts):
    # Both endpoints of the stored pressure increment were F32carriers, so its
    # F64 subtraction is exact in the qualified ordinary dynamic range. Check
    # reconstruction against the independent actual preprojection path below.
    return [output[n].reshape(s).double()-v for n,s,v in zip(('vx','vy','vz'),dimensions(counts)[:3],increments['pressure'])]


def loss(output, target, norm):
    values=[]
    for name in ('pressure','velocity','tracer'):
        a=vector(output,name);b=vector(target,name)
        denominator=b.norm().clamp_min(norm['relative_denominator_rms_floor'][name]*math.sqrt(b.numel()))
        values.append(((a-b)/denominator).square().sum())
    return sum(values)


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh offline CUDA teacher-target qualification required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen residual target dependency changed: '+p)
    gate=read(ROOT/reg['transition_qualification'])
    if gate['status']!='passed_group_transition_exact_fine_coarse_and_increment_controls':raise ValueError('Qualified actual transition helper required')
    bank=read(ROOT/reg['bank']);cases=[c for c in bank['trajectories'] if c['reference_substeps']==8]
    if bank['split']!='train' or len(bank['trajectories'])!=148 or len(cases)!=29:raise ValueError('Original active training population required')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True);began=time.monotonic();model=None
    result=dict(status='running',active=None,rows=[],source_evidence={},
        inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        cpu_model_forward=False,optimizer_updates=0,coefficient_fitting=False,new_parameter_fits=0,
        teacher_derived_oracle_injection=True,deployable_model=False,native_accuracy_claim=False,latency_claim=False,
        future_native_labels_read=False,development_or_reserved_used=False,expanded_reserved52_references_opened=0,
        full_generator_target_pass=False,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    def save():write(args.output/'progress.json',result)
    def bounded():
        if time.monotonic()-began>reg['maximum_wall_seconds']:raise TimeoutError('Finite residual target qualification deadline')
        if torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']:
            raise ValueError('Residual target qualification resource ceiling exceeded')
    try:
        model=CoupledMacroBackbone(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build'])
        before=inventory(model);runner=Transitions(model,reg);norm=read(ROOT/reg['normalization'])
        for case in cases:
            if case['split']!='train' or digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Consumed training receipt required')
            scene=read(ROOT/case['receipt'])['scene'];counts=tuple(scene['counts']);shape=tuple(reversed(counts));sizes=dimensions(counts)
            h=1./scene['cells_per_metre'];rho=1000.
            for start in reg['starts']:
                with torch.no_grad():
                    spec=next(s for s in case['states'] if s['step']==start)
                    if digest(ROOT/spec['path'])!=spec['sha256']:raise ValueError('Current training state changed')
                    result['source_evidence'][spec['path']]=spec['sha256']
                    with np.load(ROOT/spec['path'],allow_pickle=False) as archive:
                        fields={n:torch.from_numpy(archive[n].copy()).to('cuda') for n in FIELDS}
                    inputs=hashes(fields);external=model.controls.generate_from_packet(counts,model.packet(scene,start,1,8).to('cuda'))
                    fine=fields;offset=0
                for group,width in enumerate((2,2,3)):
                    result['active']=dict(identifier=case['identifier'],start=start,group_index=group);save();bounded()
                    with torch.no_grad():
                        initial=fine
                        for j in range(width):
                            last_initial=fine;fine,events=runner.advance(fine,scene,external,offset+j,1)
                        last_provisional=provisional(fine,events,counts)
                        # Independent recomputation of last-fine provisional fields
                        # qualifies the lossless reconstruction, not a new target.
                        core=model.model;core.configure_transport(h,1./480.)
                        current=[last_initial[n].reshape(s) for n,s in zip(FIELDS[1:],sizes)]
                        transported=core.transport(current,counts,1./(480.*h))
                        forced=core.affine_ops.three(transported[:3],external[offset+width-1],counts)
                        expected=core.gather_ops.three(core,forced,forced,counts,'viscosity',(.001/rho*(1./480.)/h**2))
                        if any(not torch.equal(a,b.double()) for a,b in zip(last_provisional,expected)):
                            raise ValueError('Last-fine provisional reconstruction not exact')
                        coarse,coarse_events=runner.advance(initial,scene,external,offset,width)
                        base=provisional(coarse,coarse_events,counts);dt=width/480.
                        context=dict(counts=counts,shape=shape,h=h,rho=rho,dt=dt,
                            previous=initial['pressure'].reshape(1,1,*shape),provisional=base,tracer=coarse['tracer'].reshape(shape))
                        zero=model.emit(context)
                        if hashes(zero)!=hashes(coarse):raise ValueError('Local trainable emitter zero path changed coarse complete state')
                        p=fine['pressure'].reshape(shape)
                        momentum=[wall(model.linear([p],s,'momentum',c)*(1./(rho*h)),c) for c,s in enumerate(sizes[:3])]
                        required=[v+(1./480.-dt)*m for v,m in zip(last_provisional,momentum)]
                        velocity_delta=[target-v for target,v in zip(required,base)]
                        tracer_delta=fine['tracer'].reshape(shape).double()-context['tracer'].double()
                        injected=model.emit(context,velocity_delta,tracer_delta)
                        metrics=comparison(injected,fine,fine,norm)
                        if any(v['relative']>reg['maximum_teacher_injected_relative_error'] for v in metrics.values()):
                            raise ValueError('Teacher-derived local target cannot reproduce all P/V/T within registered tolerance')
                    # One scalar probes all local residual fields without fitting.
                    alpha=torch.zeros((),device='cuda',dtype=torch.float64,requires_grad=True)
                    prediction=model.emit(context,[alpha*d for d in velocity_delta],alpha*tracer_delta,float_carriers=False)
                    objective=loss(prediction,fine,norm);objective.backward();gradient=float(alpha.grad)
                    epsilon=reg['gradient_control_epsilon']
                    with torch.no_grad():
                        sides=[loss(model.emit(context,[s*epsilon*d for d in velocity_delta],s*epsilon*tracer_delta,float_carriers=False),fine,norm) for s in (-1.,1.)]
                        numerical=float((sides[1]-sides[0])/(2.*epsilon))
                    relative=abs(gradient-numerical)/max(abs(gradient),abs(numerical),1e-24)
                    difference=abs(gradient-numerical)
                    allowed=reg['gradient_absolute_tolerance']+reg['maximum_gradient_relative_error']*max(abs(gradient),abs(numerical))
                    if not math.isfinite(gradient) or not math.isfinite(relative) or difference>allowed or (float(objective)>reg['gradient_nonzero_loss_floor'] and gradient>=0.):
                        raise ValueError('Actual local coupled residual training gradient control failed')
                    result['rows'].append(dict(identifier=case['identifier'],start=start,group_index=group,width=width,
                        counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],zero_complete_state_bits_exact=True,
                        last_fine_provisional_reconstruction_exact=True,teacher_injected_fields=metrics,
                        directional_gradient=gradient,central_difference_gradient=numerical,gradient_relative_error=relative,
                        teacher_correction_not_deployable=True))
                    offset+=width;bounded();save()
                    del prediction,objective,alpha,injected,zero
                if hashes(fields)!=inputs:raise ValueError('Teacher qualification mutated current caller fields')
                print('Local residual targets',case['identifier'],start,'zero/complete-target/gradient controls passed',flush=True)
        if len(result['rows'])!=174 or runner.calls!=580 or inventory(model)!=before:
            raise ValueError('All174target/gradient pairs and580actual learned advances required')
        for p,h in reg['source_sha256'].items():
            if digest(ROOT/p)!=h:raise ValueError('Frozen target qualification source changed during run')
        result.update(status='passed_all174_local_coupled_residual_teacher_target_and_gradient_controls',active=None,
            actual_learned_complete_advances=runner.calls,independent_provisional_recomputations=174,
            coupled_emitter_forward_calls=870,teacher_injected_or_gradient_forward_calls=696,learned_assets_unchanged=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_diagnostic_seconds=time.monotonic()-began,
            limitations='Teacher-injected targets and gradients only; future learned fields used solely as offline labels. No learned correction network is implemented/trained/deployed and no native equation/accuracy/timing gate passed. All148training retention and managedCUDA/exactCPUrestore/seed/split/nativeallK/history/independent/scaling/600gates remain.')
        write(args.output/'results.json',result)
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        if model is not None:model.close()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
