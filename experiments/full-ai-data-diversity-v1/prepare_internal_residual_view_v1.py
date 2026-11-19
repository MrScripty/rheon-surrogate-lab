"""Training-only full-grid local labels/assets, stored as bounded NPZ-in-ZIP.

No optimizer. Fine future fields are offline labels only. Candidate features
come solely from the initial state and its own coarse preprojection phase.
"""
import argparse
import collections
import io
import math
import os
from pathlib import Path
import resource
import time
import zipfile
import numpy as np
import torch
from support import ROOT,read,write,digest
from internal_group_residual_v1 import InternalGroupGenerator,encode
from internal_residual_network_v1 import wall
from internal_residual_loss_v1 import FrozenCoupledTraining,metrics,objective
from diagnose_group_transition_v1 import Transitions,hashes
from qualify_group_residual_targets_v2 import provisional
from diagnose_causal_composition import inventory
from recurrent_sampler_native import dimensions
from cuda_native_transfer_v1 import FIELDS
from numerics import fine_action


def export_sparse(core):
    result={}
    for name in ('flux','momentum'):
        rows,channels,offsets,weights=core.sparse[name]
        for key,value in [('rows',rows),('channels',channels),('offsets',offsets)]:result[name+'_'+key]=np.asarray(value,dtype='<i8')
        result[name+'_weights']=weights.detach().cpu().numpy().astype('<f8',copy=True)
    return result


def export_geometry(core,counts):
    core.prepare_geometry(counts);shape=tuple(reversed(counts))
    return dict(**{'axis'+str(a):b.detach().cpu().numpy().copy() for a,b in enumerate(core.geometry_cache[counts][1])},
                response=core.divisors['g'+'_'.join(map(str,shape))].detach().cpu().numpy().copy())


def field_vectors(fields):
    pressure=fields['pressure'].double().flatten();pressure=pressure-pressure.mean()
    return dict(pressure=pressure,velocity=torch.cat([fields[n].double().flatten() for n in ('vx','vy','vz')]),tracer=fields['tracer'].double().flatten())


def field_statistics(output,target,rhs,h,rho):
    values={};a=field_vectors(output);b=field_vectors(target)
    for name in a:values[name]=dict(reference_rms=float(b[name].square().mean().sqrt()),absolute_error_rms=float((a[name]-b[name]).square().mean().sqrt()))
    action=fine_action(output['pressure'].reshape(1,1,*rhs.shape),rhs.new_tensor([h]),rhs.new_tensor([rho]))[0,0]
    values['equation']=dict(reference_rms=float(rhs.square().mean().sqrt()),absolute_error_rms=float((action-rhs).square().mean().sqrt()))
    return values


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh CUDA-only view extraction required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen local view dependency changed: '+p)
    gate=read(ROOT/reg['architecture_qualification'])
    if gate['status']!='passed_internal_width8_network_features_zero_precision_gradient_graph_controls':raise ValueError('Qualified actual compact network required')
    bank=read(ROOT/reg['bank']);cases=bank['trajectories']
    if bank['split']!='train' or len(cases)!=148 or any(c['split']!='train' for c in cases):raise ValueError('Every original148training scene required')
    pilot=reg['mode']=='pilot'
    if pilot:cases=[c for c in cases if c['identifier'] in reg['pilot_identifiers']]
    else:
        preflight=read(ROOT/reg['pilot_manifest'])
        if preflight['status']!='verified_internal_residual_training_view_pilot':raise ValueError('Source-bound finite data/resource pilot required')
    starts=reg['starts'];os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.statvfs(args.output.parent).f_bavail*os.statvfs(args.output.parent).f_frsize<reg['minimum_free_disk_bytes']:raise ValueError('Finite disk reserve unavailable')
    args.output.mkdir(parents=True);began=time.monotonic();model=None;archive_path=args.output/'training-view.zip'
    result=dict(status='running',active=None,parts=[],geometry_parts=[],control_rows=[],scene_rows=[],source_evidence={},
        inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        optimizer_updates=0,coefficient_fitting=False,cpu_model_forward=False,development_or_reserved_used=False,
        expanded_reserved52_references_opened=0,future_native_labels_used_for_features=False,
        native_accuracy_claim=False,latency_claim=False,full_generator_target_pass=False,
        source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    def save():write(args.output/'progress.json',result)
    def bounded():
        if time.monotonic()-began>reg['maximum_wall_seconds']:raise TimeoutError('Finite local view deadline')
        if torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']:
            raise ValueError('Local view memory ceiling exceeded')
        if archive_path.exists() and archive_path.stat().st_size>reg['maximum_output_bytes']:raise ValueError('Local view archive ceiling exceeded')
    statistics=[];field_rms=[]
    try:
        with torch.no_grad(),zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            def store(member,arrays):
                buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);blob=buffer.getvalue()
                import hashlib
                archive.writestr(member,blob)
                return dict(member=member,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),arrays={n:dict(shape=list(a.shape),dtype=a.dtype.str) for n,a in arrays.items()})
            model=InternalGroupGenerator(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build'])
            before=inventory(model);runner=Transitions(model,reg);norm=read(ROOT/reg['normalization'])
            sparse=export_sparse(model.model);sparse_part=store('assets/restored-spatial-weights.npz',sparse);geometries={};loss_maps={}
            for index,case in enumerate(cases):
                if digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Training receipt changed')
                scene=read(ROOT/case['receipt'])['scene'];counts=tuple(scene['counts']);shape=tuple(reversed(counts));sizes=dimensions(counts)
                specs={s['step']:s for s in case['states']};h=1./scene['cells_per_metre'];rho=1000.;local_rows=0;scene_stats=[]
                def load(step):
                    spec=specs[step]
                    if digest(ROOT/spec['path'])!=spec['sha256']:raise ValueError('Training state bytes changed')
                    result['source_evidence'][spec['path']]=spec['sha256']
                    with np.load(ROOT/spec['path'],allow_pickle=False) as arrays:return {n:torch.from_numpy(arrays[n].copy()).to('cuda') for n in (*FIELDS,'rhs')}
                for start in starts:
                    result['active']=dict(identifier=case['identifier'],start=start);save();bounded()
                    loaded=load(start);fields={n:loaded[n] for n in FIELDS};inputs=hashes(fields)
                    if case['reference_substeps']==1:
                        truth=load(start+1);prediction=model.predict(fields,scene,start,1,None,1)[0]
                        references={n:truth[n] for n in FIELDS};rhs=truth['rhs'].reshape(shape)
                        scored=field_statistics(prediction,references,rhs,h,rho);scene_stats.append(scored)
                        relative={n:v['absolute_error_rms']/max(v['reference_rms'],norm['relative_denominator_rms_floor'][n]) for n,v in scored.items() if n!='equation'}
                        relative['equation']=float((fine_action(prediction['pressure'].reshape(1,1,*shape),rhs.new_tensor([h]),rhs.new_tensor([rho]))[0,0]-rhs).norm()/rhs.norm().clamp_min(1e-30))
                        result['control_rows'].append(dict(identifier=case['identifier'],start=start,counts_xyz=list(counts),cells_per_metre=scene['cells_per_metre'],statistics=scored,original_eval_relative=relative,trainable_branch=False))
                    elif case['reference_substeps']==8:
                        if counts not in geometries:
                            asset=export_geometry(model.model,counts);part=store('assets/geometry-'+'x'.join(map(str,counts))+'.npz',asset)
                            part.update(counts_xyz=list(counts));result['geometry_parts'].append(part);geometries[counts]=part['member'];loss_maps[counts]=FrozenCoupledTraining(sparse,asset)
                        external=model.controls.generate_from_packet(counts,model.packet(scene,start,1,8).to('cuda'));fine=fields;offset=0
                        for group,width in enumerate((2,2,3)):
                            initial=fine;events=None
                            for j in range(width):fine,events=runner.advance(fine,scene,external,offset+j,1)
                            phase=model.phase(initial,scene,external,offset,width);features=encode(phase);base=model.emit(phase,float_carriers=False)
                            last=provisional(fine,events,counts);p=fine['pressure'].reshape(shape)
                            momentum=[wall(model.linear([p],s,'momentum',c)*(1./(rho*h)),c) for c,s in enumerate(sizes[:3])]
                            target_delta=[(a+(1./480.-phase['dt'])*m-v.double())*(phase['dt']/h) for a,m,v in zip(last,momentum,phase['provisional'])]
                            target_delta.append(fine['tracer'].reshape(shape).double()-phase['tracer'].double())
                            sample=dict(h=h,dt=phase['dt'],rho=rho)
                            for n,s in zip(FIELDS,(shape,*sizes)):
                                sample['base_'+n]=base[n].reshape(s);sample['target_'+n]=fine[n].reshape(s)
                            sample['target_rhs']=fine_action(p.reshape(1,1,*shape),p.new_tensor([h]),p.new_tensor([rho]))[0,0]
                            map_errors=[];loss_errors=[]
                            for alpha in (0.,.5,1.):
                                delta=[(d*alpha).float() for d in target_delta]
                                portable=loss_maps[counts].emit(sample,delta)
                                actual=model.emit(phase,[d.double()*(h/phase['dt']) for d in delta[:3]],delta[3].double(),float_carriers=False)
                                actual={n:actual[n].reshape(sample['target_'+n].shape) for n in FIELDS}
                                for n in FIELDS:
                                    a=portable[n].flatten();b=actual[n].flatten()
                                    error=float((a-b).norm()/b.norm().clamp_min(1.))
                                    if error>reg['maximum_portable_field_error']:raise ValueError('Portable frozen response changed actual complete map')
                                    map_errors.append(error)
                                expected=metrics(actual,sample,norm);observed=metrics(portable,sample,norm)
                                for name in observed:
                                    error=abs(float(observed[name]-expected[name]));allowed=reg['portable_loss_absolute_tolerance']+reg['portable_loss_relative_tolerance']*abs(float(expected[name]))
                                    if error>allowed:raise ValueError('Portable full-grid four-field loss differs from actual coupled map')
                                    loss_errors.append(error)
                            precision={n:float(v.sqrt()) for n,v in metrics(actual,sample,norm).items()}
                            with torch.enable_grad():
                                alpha=torch.tensor(.5,device='cuda',dtype=torch.float64,requires_grad=True)
                                direction=[d.float().double()*alpha for d in target_delta]
                                portable_gradient_loss=objective(loss_maps[counts].emit(sample,direction),sample,norm)
                                actual_gradient_fields=model.emit(phase,[d*(h/phase['dt']) for d in direction[:3]],direction[3],float_carriers=False)
                                actual_gradient_fields={n:actual_gradient_fields[n].reshape(sample['target_'+n].shape) for n in FIELDS}
                                actual_gradient_loss=objective(actual_gradient_fields,sample,norm)
                                ga=float(torch.autograd.grad(actual_gradient_loss,alpha,retain_graph=True)[0])
                                gp=float(torch.autograd.grad(portable_gradient_loss,alpha)[0])
                                gradient_error=abs(ga-gp)
                                if gradient_error>reg['portable_gradient_absolute_tolerance']+reg['portable_gradient_relative_tolerance']*max(abs(ga),abs(gp)):
                                    raise ValueError('Portable full-grid four-field gradient differs from actual coupled map')
                                del alpha,direction,portable_gradient_loss,actual_gradient_fields,actual_gradient_loss
                            arrays={n:np.asarray(v,dtype='<f8') for n,v in [('h',h),('dt',phase['dt']),('rho',rho)]}
                            arrays['features']=features.cpu().numpy().copy()
                            for n,v in sample.items():
                                if isinstance(v,torch.Tensor):arrays[n]=v.cpu().numpy().copy()
                            for n,v in zip(('vx','vy','vz','tracer'),target_delta):arrays['target_delta_'+n]=v.cpu().numpy().copy()
                            part=store(f'samples/scene{index:03d}-start{start:03d}-group{group}.npz',arrays)
                            part.update(identifier=case['identifier'],start=start,group_index=group,width=width,counts_xyz=list(counts),cells_per_metre=scene['cells_per_metre'],geometry_member=geometries[counts],maximum_portable_field_error=max(map_errors),maximum_portable_loss_error=max(loss_errors),actual_directional_gradient=ga,portable_directional_gradient=gp,portable_gradient_absolute_error=gradient_error,F32_injected_original_relative_errors=precision)
                            result['parts'].append(part);local_rows+=1
                            statistics.append(dict(input_square_means=features.double().square().mean(dim=(0,2,3,4)).cpu().tolist(),output_square_means=[float(v.square().mean()) for v in target_delta]))
                            target_fields={n:sample['target_'+n] for n in FIELDS};scored=field_statistics(target_fields,target_fields,sample['target_rhs'],h,rho);scene_stats.append(scored)
                            del arrays,features,sample,base,portable,actual,target_delta
                            offset+=width;bounded();save()
                    else:raise ValueError('Original1/8reference regimes required')
                    if hashes(fields)!=inputs:raise ValueError('View extraction changed caller physical fields')
                    bounded();save()
                field_rms.append({n:sum(x[n]['reference_rms']**2 for x in scene_stats)/len(scene_stats) for n in ('equation','pressure','velocity','tracer')})
                result['scene_rows'].append(dict(identifier=case['identifier'],teacher_substeps=case['reference_substeps'],current_starts=starts,local_training_samples=local_rows,fixed_N1_control_windows=len(starts) if case['reference_substeps']==1 else 0))
                print('Internal full-grid view',case['identifier'],'local samples',local_rows,flush=True)
            if inventory(model)!=before:raise ValueError('Frozen learned assets changed during view extraction')
            if not pilot and (len(result['scene_rows'])!=148 or len(result['parts'])!=783 or len(result['control_rows'])!=1071):raise ValueError('All148scenes/783active samples/1071fixed controls required')
            if pilot and (len(result['parts'])!=9 or len(result['scene_rows'])!=7):raise ValueError('Finite3active/4fixed resource pilot required')
            input_rms=np.sqrt(np.mean([x['input_square_means'] for x in statistics],axis=0));output_rms=np.sqrt(np.mean([x['output_square_means'] for x in statistics],axis=0))
            normalization=dict(input_rms=np.maximum(input_rms,1e-8).tolist(),output_rms=np.maximum(output_rms,1e-12).tolist(),
                input_floor=1e-8,output_floor=1e-12,source='Only active training full-grid samples, equal per local sample (allscenes/starts/groups balanced). No label influences feature construction; output scales are training-label RMS only.')
            training_norm=dict(norm);training_norm['relative_denominator_rms_floor']=dict(norm['relative_denominator_rms_floor'])
            for name in ('pressure','velocity','tracer'):training_norm['relative_denominator_rms_floor'][name]=max(norm['relative_denominator_rms_floor'][name],.01*math.sqrt(sum(x[name] for x in field_rms)/len(field_rms)))
            training_norm['optimization_equation_rms_floor']=.01*math.sqrt(sum(x['equation'] for x in field_rms)/len(field_rms))
            training_norm['optimization_normalization_note']='Training-only optimization floors1percent of scene-balanced teacher/current reference RMS. Original evaluation floors/1e-5criteria unchanged; this file cannot qualify accuracy.'
            write(args.output/'network-normalization.json',normalization);write(args.output/'optimization-normalization.json',training_norm)
        import hashlib
        with zipfile.ZipFile(archive_path) as verify:
            for p in [sparse_part,*result['geometry_parts'],*result['parts']]:
                if hashlib.sha256(verify.read(p['member'])).hexdigest()!=p['sha256']:raise ValueError('Final archive member changed')
        for p,h in reg['source_sha256'].items():
            if digest(ROOT/p)!=h:raise ValueError('View source changed during extraction')
        status='verified_internal_residual_training_view_pilot' if pilot else 'verified_all148_internal_residual_training_view'
        manifest=dict(result,status=status,active=None,mode=reg['mode'],starts=starts,archive=dict(path=archive_path.name,sha256=digest(archive_path),bytes=archive_path.stat().st_size),
            sparse_part=sparse_part,network_normalization_sha256=digest(args.output/'network-normalization.json'),optimization_normalization_sha256=digest(args.output/'optimization-normalization.json'),
            original_eval_normalization=reg['normalization'],all148training_scenes_in_objective=not pilot,trainable_N8_scenes=29 if not pilot else 3,fixed_N1_scenes=119 if not pilot else 4,
            scene_weighting='Mean over148scenes.29active scenes each9starts x3groups equal weighted,119fixed controls each9starts. Fixed losses remain in objective withzero parameter derivative; optimization total=(29*meanactive+sum119fixed)/148.',
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_preparation_seconds=time.monotonic()-began,
            label_scope='Active local future labels are restored learned fine trajectories. Equation target is independent fine_action(learnedfineP), not a fresh native internalRHS. FixedN1control future fields/RHS are existing native TRAINING labels only. Native allK/history/held-out/timing qualification still required.')
        write(args.output/'manifest.json',manifest);result.update(status='completed',active=None,manifest_sha256=digest(args.output/'manifest.json'))
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        if model is not None:model.close()
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
