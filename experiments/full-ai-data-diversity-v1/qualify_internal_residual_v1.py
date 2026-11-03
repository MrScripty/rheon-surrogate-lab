"""CUDA architecture/representation/zero/gradient gates, no parameter fitting."""
import argparse
import copy
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT,read,write,digest
from internal_group_residual_v1 import InternalGroupGenerator,InternalGroupResidual,encode,face_pairs,PARAMETERS
from internal_residual_network_v1 import _ClampedPad,wall
from diagnose_group_transition_v1 import Transitions,hashes,comparison
from qualify_group_residual_targets_v2 import loss,provisional
from diagnose_causal_composition import inventory
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS


def gradients(model):
    result={}
    for n,p in model.named_parameters():
        if p.grad is None or not bool(p.grad.isfinite().all()):raise ValueError('Missing/nonfinite actual network gradient: '+n)
        result[n]=float(p.grad.double().norm())
    return result


def clone_phase(phase):
    result=dict(phase)
    result['current']={n:v.clone() for n,v in phase['current'].items()}
    for key in ('advected','provisional'):result[key]=[v.clone() for v in phase[key]]
    result['forcing']=[[v.clone() for v in step] for step in phase['forcing']]
    return result


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh GPU-only architecture qualification required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen internal residual dependency changed: '+p)
    target_gate=read(ROOT/reg['target_qualification'])
    if target_gate['status']!='passed_all174_local_coupled_residual_teacher_target_and_gradient_controls':raise ValueError('All174compatible target/gradient controls required')
    bank=read(ROOT/reg['bank']);cases={c['identifier']:c for c in bank['trajectories']}
    if bank['split']!='train' or len(cases)!=148:raise ValueError('Full original training bank required')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    args.output.mkdir(parents=True);began=time.monotonic();model=None
    result=dict(status='running',active=None,padding_rows=[],rows=[],graph_rows=[],zero_block_rows=[],N1_control_rows=[],
        input_representation_rows=[],source_evidence={},inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        parameters=PARAMETERS,initialization_seed=reg['seed'],normalization='Unit input/output RMS for architecture qualification only; actual train-only scales require separate fit qualification.',
        optimizer_updates=0,coefficient_fitting=False,new_parameter_fits=0,cpu_model_forward=False,
        development_or_reserved_used=False,expanded_reserved52_references_opened=0,future_native_labels_read=False,
        native_accuracy_claim=False,latency_claim=False,full_generator_target_pass=False,
        source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    def save():write(args.output/'progress.json',result)
    def bounded():
        if time.monotonic()-began>reg['maximum_wall_seconds']:raise TimeoutError('Finite internal architecture deadline')
        if torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']:
            raise ValueError('Internal architecture resource ceiling exceeded')
    def load(identifier):
        case=cases[identifier]
        if case['split']!='train' or digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Consumed training receipt required')
        spec=next(s for s in case['states'] if s['step']==reg['start'])
        if digest(ROOT/spec['path'])!=spec['sha256']:raise ValueError('Current training state changed')
        result['source_evidence'][spec['path']]=spec['sha256']
        with np.load(ROOT/spec['path'],allow_pickle=False) as archive:
            fields={n:torch.from_numpy(archive[n].copy()).to('cuda') for n in FIELDS}
        return read(ROOT/case['receipt'])['scene'],fields
    try:
        for shape in ((3,4,5),(1,3,5)):
            x=torch.arange(math.prod(shape),device='cuda',dtype=torch.float64).reshape(1,1,*shape).requires_grad_()
            value=_ClampedPad.apply(x);value.sum().backward()
            axes=[]
            for n in shape:
                a=torch.arange(n,device='cuda');axes.append(1.+(a==0).double()+(a==n-1).double())
            expected=axes[0][:,None,None]*axes[1][None,:,None]*axes[2][None,None,:]
            if not torch.equal(x.grad[0,0],expected):raise ValueError('Deterministic clamped-padding boundary multiplicity gradient failed')
            result['padding_rows'].append(dict(shape=list(shape),boundary_gradient_multiplicities_exact=True))
        model=InternalGroupGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build'])
        before=inventory(model);network=InternalGroupResidual(seed=reg['seed']);initial=inventory(network)
        probe=copy.deepcopy(network)
        # A fixed unfitted probe is separate from the saved zero-head network.
        with torch.no_grad():
            for layer in (*probe.velocity,probe.tracer):
                layer.weight.copy_(torch.linspace(-.001,.001,layer.weight.numel(),device='cuda').reshape_as(layer.weight));layer.bias.fill_(.0001)
        probe_before=inventory(probe);runner=Transitions(model,reg);norm=read(ROOT/reg['normalization'])
        for identifier in reg['identifiers']:
            if cases[identifier]['reference_substeps']!=8:raise ValueError('Active training scene required')
            scene,fields=load(identifier);inputs=hashes(fields);counts=tuple(scene['counts']);sizes=dimensions(counts)
            with torch.no_grad():
                fine_states=model.feature_states_grouped(fields,scene,reg['start'],(1,)*8)[0]
                external=model.controls.generate_from_packet(counts,model.packet(scene,reg['start'],1,8).to('cuda'))
            for offset,width in ((0,2),(4,3)):
                result['active']=dict(identifier=identifier,offset=offset,width=width);save();bounded()
                with torch.no_grad():
                    phase=model.phase(fine_states[offset],scene,external,offset,width)
                    features=encode(phase)
                    if features.shape!=(1,42,*reversed(counts)) or not bool(features.isfinite().all()):raise ValueError('Complete finite42channel features required')
                    values=network(features)
                    if any(bool(v.abs().max()!=0.) for v in values):raise ValueError('Initial residual heads must be exactly zero')
                    for c,n in enumerate(('vx','vy','vz')):
                        v=phase['current'][n].reshape(sizes[c]);lo,hi=face_pairs(v,c);axis=2-c
                        restored=torch.cat((lo,hi.narrow(axis,hi.shape[axis]-1,1)),dim=axis)
                        if not torch.equal(restored,v):raise ValueError('Face packing loses current velocity information')
                    shifted=clone_phase(phase);shifted['current']['pressure'].add_(1.25)
                    gauge=float((encode(shifted)-features).double().norm()/features.double().norm().clamp_min(1.))
                    if gauge>reg['maximum_feature_control_error']:raise ValueError('Current pressure feature gauge control failed')
                    fine,events=runner.advance(fine_states[offset+width-1],scene,external,offset+width-1,1)
                    if hashes(fine)!=hashes(fine_states[offset+width]):raise ValueError('Teacher lastfine endpoint changed')
                    last=provisional(fine,events,counts);p=fine['pressure'].reshape(phase['shape'])
                    momentum=[wall(model.linear([p],s,'momentum',c)*(1./(phase['rho']*phase['h'])),c) for c,s in enumerate(sizes[:3])]
                    target=[a+(1./480.-phase['dt'])*m-v.double() for a,m,v in zip(last,momentum,phase['provisional'])]
                    quantized=[(d*(phase['dt']/phase['h'])).float().double()*(phase['h']/phase['dt']) for d in target]
                    tracer_delta=(fine['tracer'].reshape(phase['shape']).double()-phase['tracer'].double()).float().double()
                    injected=model.emit(phase,quantized,tracer_delta);precision=comparison(injected,fine,fine,norm)
                    if any(v['relative']>1e-5 for v in precision.values()):raise ValueError('F32head output precision cannot reproduce qualified targets')
                network.zero_grad(set_to_none=True)
                prediction=model.corrected(phase,network,float_carriers=False);objective=loss(prediction,fine,norm);objective.backward()
                zero_gradients=gradients(network)
                if any(v!=0. for k,v in zero_gradients.items() if k.startswith(('conv1.','conv2.'))) or not any(v>0. for k,v in zero_gradients.items() if k.startswith(('velocity.','tracer.'))):
                    raise ValueError('Zero-head initialization gradient structure failed')
                probe.zero_grad(set_to_none=True);probe_output=model.corrected(phase,probe,float_carriers=False);probe_loss=loss(probe_output,fine,norm);probe_loss.backward()
                probe_gradients=gradients(probe)
                if not all(probe_gradients[k]>0. for k in ('conv1.weight','conv2.weight')):raise ValueError('Actual trunk did not receive nonzero finite gradients')
                analytic=float(probe.velocity[0].bias.grad[0]);epsilon=reg['gradient_epsilon'];sides=[]
                with torch.no_grad():
                    original=probe.velocity[0].bias.clone()
                    for sign in (-1.,1.):
                        probe.velocity[0].bias.copy_(original+sign*epsilon)
                        sides.append(float(loss(model.corrected(phase,probe,float_carriers=False),fine,norm)))
                    probe.velocity[0].bias.copy_(original)
                numerical=(sides[1]-sides[0])/(2.*epsilon)
                relative=abs(analytic-numerical)/max(abs(analytic),abs(numerical),1e-20)
                if not math.isfinite(relative) or relative>reg['maximum_F32_parameter_gradient_error']:raise ValueError('Actual F32network parameter gradient disagrees with finite difference')
                result['rows'].append(dict(identifier=identifier,offset=offset,width=width,counts_xyz=list(counts),cells_per_metre=scene['cells_per_metre'],
                    face_packing_exact=True,zero_outputs_exact=True,pressure_gauge_feature_error=gauge,
                    F32_teacher_output_precision_only=precision,zero_head_gradient_norms=zero_gradients,probe_gradient_norms=probe_gradients,
                    parameter_gradient=analytic,finite_difference_parameter_gradient=numerical,parameter_gradient_relative_error=relative))
                if width==2:
                    static=clone_phase(phase)
                    with torch.no_grad():
                        for _ in range(2):probe(encode(static))
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):captured=probe(encode(static))
                        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();old=[v.clone() for v in captured]
                        for v in static['forcing'][0]:v.add_(torch.linspace(-.001,.002,v.numel(),device='cuda',dtype=v.dtype).reshape_as(v))
                        graph.replay();torch.cuda.synchronize();expected=probe(encode(static))
                        errors=[float((a-b).double().norm()/b.double().norm().clamp_min(1.)) for a,b in zip(captured,expected)]
                        if any(e>reg['maximum_graph_error'] for e in errors) or any(torch.equal(a,b) for a,b in zip(old,expected)):
                            raise ValueError('Changed-force captured CNN graph stale or incorrect')
                        result['graph_rows'].append(dict(identifier=identifier,all_four_outputs_change=True,maximum_relative_error=max(errors)))
                        del graph,captured,old,expected,static
                if identifier==reg['collision_identifier'] and width==3:
                    with torch.no_grad():
                        collision=read(ROOT/reg['collision_evidence']);source=collision['pattern_source']
                        raw=model.controls.generate_from_packet(counts,model.packet(scene,source['known_control_start'],1,8).to('cuda'))[source['internal_index']]
                        pulse=[v/source['original_maximum']*.005 for v in raw];encoded=[]
                        for sign in (1.,-1.):
                            forces=[[v*sign for v in pulse],[-v*sign for v in pulse],[torch.zeros_like(v) for v in pulse]]
                            context=model.phase(phase['current'],scene,forces,0,3);encoded.append(encode(context))
                        if not torch.equal(encoded[0][:,:21],encoded[1][:,:21]) or torch.equal(encoded[0][:,21:39],encoded[1][:,21:39]):
                            raise ValueError('Ordered local force slots still alias controlled pulse reversal')
                        result['input_representation_rows'].append(dict(control='prior_pulse_order_collision',amplitude=.005,old_current_and_coalesced_channels_exact=True,ordered_force_channels_different=True))
                del prediction,objective,probe_output,probe_loss,features,values
                bounded();save()
            with torch.no_grad():
                for K in (1,2,3,4):
                    actual=model.predict(fields,scene,reg['start'],K,network,8)
                    expected=[block[-1] for block in model.feature_states_grouped(fields,scene,reg['start'],(2,2,3,1),horizons=K)]
                    if len(actual)!=K or any(hashes(a)!=hashes(b) for a,b in zip(actual,expected)):raise ValueError('Zero residual changed complete own-state multi-output baseline')
                    result['zero_block_rows'].append(dict(identifier=identifier,K=K,complete_physical_outputs_exact=K))
            if hashes(fields)!=inputs:raise ValueError('Architecture qualification mutated caller fields')
            print('Internal residual',identifier,'features/precision/gradients/graph/allKzero passed',flush=True)
        class ForbiddenNetwork:
            def __call__(self,*args):raise ValueError('N1control invoked learned correction network')
        with torch.no_grad():
            for identifier in reg['N1_identifiers']:
                if cases[identifier]['reference_substeps']!=1:raise ValueError('Frozen N1control required')
                scene,fields=load(identifier)
                for K in (1,2,3,4):
                    actual=model.predict(fields,scene,reg['start'],K,ForbiddenNetwork(),1)
                    expected=[block[-1] for block in model.feature_states(fields,scene,reg['start'],1,horizons=K)]
                    if any(hashes(a)!=hashes(b) for a,b in zip(actual,expected)):raise ValueError('Frozen N1regime baseline changed')
                    result['N1_control_rows'].append(dict(identifier=identifier,K=K,complete_physical_outputs_exact=K,neural_correction_not_called=True))
        if inventory(network)!=initial or inventory(probe)!=probe_before or inventory(model)!=before:raise ValueError('Unfitted network/probe/frozen assets changed')
        if len(result['rows'])!=6 or len(result['graph_rows'])!=3 or len(result['zero_block_rows'])!=12 or len(result['N1_control_rows'])!=12 or len(result['input_representation_rows'])!=1:
            raise ValueError('Complete registered architecture controls required')
        for p,h in reg['source_sha256'].items():
            if digest(ROOT/p)!=h:raise ValueError('Architecture source changed during qualification')
        initial_path=args.output/'untrained-initial-weights.npz'
        np.savez_compressed(initial_path,**{n:v.detach().cpu().numpy().copy() for n,v in network.state_dict().items()})
        result.update(status='passed_internal_width8_network_features_zero_precision_gradient_graph_controls',active=None,
            initial_untrained_weights=dict(path=initial_path.name,sha256=digest(initial_path),bytes=initial_path.stat().st_size),
            frozen_assets_and_initial_weights_unchanged=True,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_diagnostic_seconds=time.monotonic()-began,
            limitations='Initial architecture only. Unit RMS scales, six local target examples and three N1controls are not an all148scene fit objective gate. Probe head assignments are fixed unfitted diagnostics, not trained/deployable weights. Actual training-only normalization/full-grid data/objective/managedCUDAfit/exactCPUrestore/nativeallK/history/seed/independent/timing/scaling/600gates remain.')
        write(args.output/'results.json',result)
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        if model is not None:model.close()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
