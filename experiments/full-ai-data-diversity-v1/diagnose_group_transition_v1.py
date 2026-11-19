"""Training-only local grouping/inherited error diagnostic, never a candidate.

References are restored learned fine-stage trajectories, not new native solver
labels. Velocity increments telescope over the same physical group duration;
last-fine and coarse provisional fields are not interchangeable targets.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from grouped_macrostep_backbone_v1 import GroupedMacroBackbone
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory


TERMS = ('transport', 'forcing', 'viscosity', 'pressure')


def hashes(fields):
    return {n: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n, v in fields.items()}


def vector(fields, name):
    values = [fields[n].double().flatten() for n in (('vx', 'vy', 'vz') if name == 'velocity' else (name,))]
    value = torch.cat(values)
    return value-value.mean() if name == 'pressure' else value


def metric(delta, reference, name, norm):
    floor = norm['relative_denominator_rms_floor'][name]
    denominator = reference.norm().clamp_min(floor*math.sqrt(reference.numel()))
    value = dict(relative=float(delta.norm()/denominator), absolute_rms=float(delta.square().mean().sqrt()),
                 maximum_absolute=float(delta.abs().max()), reference_rms=float(reference.square().mean().sqrt()))
    if not all(math.isfinite(v) for v in value.values()): raise ValueError('Nonfinite transition diagnostic')
    return value


def comparison(a, b, reference, norm):
    return {n: metric(vector(a,n)-vector(b,n),vector(reference,n),n,norm) for n in ('pressure','velocity','tracer')}


class Transitions:
    def __init__(self, backbone, reg):
        self.backbone=backbone; self.reg=reg; self.calls=0; self.maximum_closure=0.

    @torch.no_grad()
    def advance(self, fields, scene, external, offset, width):
        counts=tuple(scene['counts']); shape=tuple(reversed(counts)); core=self.backbone.model
        h=1./scene['cells_per_metre']; rho=1000.; dt=width/(60.*8.)
        core.prepare_geometry(counts); core.configure_transport(h,dt)
        current=[fields[n].reshape(s) for n,s in zip(FIELDS[1:],dimensions(counts))]
        previous=fields['pressure'].reshape(1,1,*shape)
        addition=external[offset]
        for j in range(1,width): addition=[a+b for a,b in zip(addition,external[offset+j])]
        advected=core.transport(current,counts,dt/h)
        entering=core.affine_ops.three(advected[:3],addition,counts)
        provisional=core.gather_ops.three(core,entering,entering,counts,'viscosity',(.001/rho*dt/h**2))
        rhs=core.gather_ops.ordered_rhs(core,provisional,shape,'flux',0,h*h,dt)
        history=core.gather_ops.three(core,[previous[0,0]],None,counts,'momentum',(h*h)*(1/(rho*h)),multiply_only=True,as_double=True)
        request=core.gather_ops.difference(core,history,rhs,shape,'flux',0,-1.,rho/h)
        correction=core.pressure(request.reshape(1,1,*shape),counts)
        pressure=core.affine_ops.binary('sum_gauge',previous,correction)
        velocity=core.gather_ops.three(core,[pressure[0,0]],provisional,counts,'momentum',dt*(1/(rho*h)))
        output=dict(pressure=pressure.contiguous().flatten(),**{n:v.flatten() for n,v in zip(FIELDS[1:],[*velocity,advected[3]])})
        if any(not bool(v.isfinite().all()) for v in output.values()):raise ValueError('Nonfinite complete learned state')
        for c,v in enumerate(velocity):
            axis=2-c
            if bool(v.select(axis,0).abs().max()!=0.) or bool(v.select(axis,v.shape[axis]-1).abs().max()!=0.):
                raise ValueError('Learned transition violated normal velocity walls')
        steps=(current[:3],advected[:3],entering,provisional,velocity)
        increments={name:[b.double()-a.double() for a,b in zip(steps[i],steps[i+1])] for i,name in enumerate(TERMS)}
        for c in range(3):
            total=sum((increments[n][c] for n in TERMS),torch.zeros_like(increments['transport'][c]))
            target=velocity[c].double()-current[c].double()
            denominator=target.norm()+sum(increments[n][c].norm() for n in TERMS)
            closure=float((total-target).norm()/denominator.clamp_min(1e-30))
            self.maximum_closure=max(self.maximum_closure,closure)
            if closure>self.reg['maximum_additive_closure_error']: raise ValueError('Stage increment decomposition failed to telescope')
        self.calls+=1
        return output,increments


def sum_terms(accumulator, increments):
    if accumulator is None: return {n:[v.clone() for v in increments[n]] for n in TERMS}
    return {n:[a+b for a,b in zip(accumulator[n],increments[n])] for n in TERMS}


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda': raise ValueError('Fresh CUDA-only transition diagnostic required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h: raise ValueError('Frozen transition source changed: '+p)
    bank=read(ROOT/reg['bank']); cases=[c for c in bank['trajectories'] if c['reference_substeps']==8]
    if bank['split']!='train' or len(bank['trajectories'])!=148 or len(cases)!=29 or any(c['split']!='train' for c in cases):
        raise ValueError('Original all148training bank with29active scenes required')
    qualification=reg['mode']=='qualification'
    if qualification:
        selected={c['identifier']:c for c in cases};cases=[selected[n] for n in reg['qualification_identifiers']]
        starts=[32]
    else:
        gate=read(ROOT/reg['qualification'])
        if gate['status']!='passed_group_transition_exact_fine_coarse_and_increment_controls' or len(gate['qualification_rows'])!=3:
            raise ValueError('Exact independent fine/coarse path qualification required')
        starts=reg['starts']
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True);began=time.monotonic();backbone=None
    result=dict(status='running',active=None,rows=[],windows=[],qualification_rows=[],parts=[],source_evidence={},
        inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        cpu_model_forward=False,optimizer_updates=0,coefficient_fitting=False,new_parameter_fits=0,
        development_or_reserved_used=False,expanded_reserved52_references_opened=0,future_native_labels_read=False,
        reference='Restored learned eight-stage trajectories, no fresh native reference or deployment accuracy claim.',
        native_accuracy_claim=False,latency_claim=False,full_generator_target_pass=False,
        source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    def save(): write(args.output/'progress.json',result)
    def bounded():
        if time.monotonic()-began>reg['maximum_wall_seconds']: raise TimeoutError('Finite group-transition deadline')
        if torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes'] or sum(p['bytes'] for p in result['parts'])>reg['maximum_output_bytes']:
            raise ValueError('Transition diagnostic resource ceiling exceeded')
    try:
        with torch.no_grad():
            backbone=GroupedMacroBackbone(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build'])
            before=inventory(backbone);runner=Transitions(backbone,reg);norm=read(ROOT/reg['normalization'])
            for case in cases:
                if digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Training scene receipt changed')
                scene=read(ROOT/case['receipt'])['scene'];counts=tuple(scene['counts'])
                for start in starts:
                    result['active']=dict(identifier=case['identifier'],start=start);save();bounded()
                    spec=next(s for s in case['states'] if s['step']==start)
                    if digest(ROOT/spec['path'])!=spec['sha256']:raise ValueError('Training current-state bytes changed')
                    result['source_evidence'][spec['path']]=spec['sha256']
                    with np.load(ROOT/spec['path'],allow_pickle=False) as archive:
                        fields={n:torch.from_numpy(archive[n].copy()).to('cuda') for n in FIELDS}
                    inputs=hashes(fields)
                    forces=backbone.controls.generate_from_packet(counts,backbone.packet(scene,start,1,8).to('cuda'))
                    force_inputs=hashes({str(i*3+c):v for i,step in enumerate(forces) for c,v in enumerate(step)})
                    if qualification:
                        comparisons=0
                        for widths in ((1,)*8,(2,2,3,1)):
                            actual=[fields];offset=0
                            for width in widths:
                                state,_=runner.advance(actual[-1],scene,forces,offset,width);actual.append(state);offset+=width
                            expected=backbone.feature_states_grouped(fields,scene,start,widths)[0]
                            if any(hashes(a)!=hashes(b) for a,b in zip(actual,expected)) or len(actual)!=len(expected):
                                raise ValueError('Diagnostic advance changed original complete internal physical outputs')
                            comparisons+=len(actual)
                        result['qualification_rows'].append(dict(identifier=case['identifier'],start=start,counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],complete_internal_state_comparisons=comparisons,
                            all_pressure_velocity_tracer_bits_exact=True,width1_path_exact=True,additive_increment_closure=runner.maximum_closure))
                    else:
                        fine=fields;rolling=fields;offset=0
                        for group,width in enumerate((2,2,3)):
                            initial=fine;increments=None
                            for j in range(width):
                                fine,event=runner.advance(fine,scene,forces,offset+j,1);increments=sum_terms(increments,event)
                            isolated,coarse_terms=runner.advance(initial,scene,forces,offset,width)
                            rolling,_=runner.advance(rolling,scene,forces,offset,width)
                            reference=vector(fine,'velocity');term_rows={};total=None;term_norm=reference.new_zeros(())
                            for name in TERMS:
                                difference=torch.cat([(a-b).flatten() for a,b in zip(coarse_terms[name],increments[name])])
                                term_rows[name]=metric(difference,reference,'velocity',norm)
                                term_norm=term_norm+difference.norm()
                                total=difference if total is None else total+difference
                            defect=vector(isolated,'velocity')-reference
                            closure=float((total-defect).norm()/(term_norm+defect.norm()).clamp_min(1e-30))
                            if closure>reg['maximum_additive_closure_error']:raise ValueError('Group contribution differences do not close')
                            temporal_variation=max(float((v-forces[offset][c]).norm()) for j in range(width) for c,v in enumerate(forces[offset+j]))
                            max_displacement=max(float(initial[n].abs().max()) for n in ('vx','vy','vz'))*(width/(60.*8.))*scene['cells_per_metre']
                            result['rows'].append(dict(identifier=case['identifier'],start=start,group_index=group,fine_offset=offset,width=width,
                                counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],physical_group_seconds=width/(60.*8.),
                                local_coarse_from_identical_start=comparison(isolated,fine,fine,norm),inherited_through_coarse_transition=comparison(rolling,isolated,fine,norm),total_grouped_error=comparison(rolling,fine,fine,norm),
                                accumulated_velocity_contribution_differences=term_rows,contribution_closure_error=closure,
                                within_group_force_variation_l2=temporal_variation,max_component_velocity_group_dt_over_h=max_displacement,
                                learned_reference_end_sha256=hashes(fine),isolated_coarse_end_sha256=hashes(isolated),rolling_coarse_end_sha256=hashes(rolling)))
                            offset+=width;bounded();save()
                        fine,_=runner.advance(fine,scene,forces,7,1);rolling,_=runner.advance(rolling,scene,forces,7,1)
                        arrays={f'{prefix}_{n}':v.cpu().numpy().copy() for prefix,state in [('fine8',fine),('grouped4',rolling)] for n,v in state.items()}
                        path=args.output/(case['identifier']+f'-start{start:03d}-endpoints.npz');np.savez_compressed(path,**arrays)
                        result['parts'].append(dict(path=path.name,sha256=digest(path),bytes=path.stat().st_size))
                        result['windows'].append(dict(identifier=case['identifier'],start=start,counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],endpoint=comparison(rolling,fine,fine,norm),endpoint_part=path.name))
                        del fine,rolling,isolated,initial,increments,coarse_terms,arrays
                    if hashes(fields)!=inputs or hashes({str(i*3+c):v for i,step in enumerate(forces) for c,v in enumerate(step)})!=force_inputs:
                        raise ValueError('Diagnostic changed caller fields or prescribed forces')
                    bounded();save()
                    print('Group-transition',case['identifier'],start,'exact paths' if qualification else 'three isolated/inherited pairs and final complete endpoints',flush=True)
            if inventory(backbone)!=before:raise ValueError('Diagnostic changed restored learned coefficients')
            for p,h in reg['source_sha256'].items():
                if digest(ROOT/p)!=h:raise ValueError('Frozen transition source changed during run')
            if qualification:
                if len(result['qualification_rows'])!=3 or runner.calls!=36:raise ValueError('All3geometry exact-path controls required')
                status='passed_group_transition_exact_fine_coarse_and_increment_controls'
            else:
                if len(result['rows'])!=174 or len(result['windows'])!=58 or runner.calls!=870:raise ValueError('All174transition pairs/58windows/870actual advances required')
                status='completed_all_training_group_transition_defect_diagnostic'
            result.update(status=status,active=None,actual_diagnostic_learned_advances=runner.calls,
                reference_control_additional_advances=36 if qualification else 0,
                maximum_stage_increment_closure_error=runner.maximum_closure,learned_assets_unchanged=True,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                elapsed_diagnostic_seconds=time.monotonic()-began,
                interpretation='Velocity increments include actual F32carrier rounding and telescope. Fine transport increments include flow changes from intervening forcing/projection; these are additive bookkeeping, not independent causal attributions. Inherited component is rolling-coarse minus isolated-coarse, with the same learned fine-end normalization. Pressure is mean-free. All references learned; no native accuracy or latency admission.')
            write(args.output/'results.json',result)
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        if backbone is not None:backbone.close()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
