"""Fixed train-only modal features from all expanded native reference scenes.

CPU execution is offline teacher preprocessing only. No learned model, fitted
coefficient, physical eigenvalue or pressure inverse is constructed here.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from support import ROOT, read, write, digest
from fftw_pressure_transform import DCTMultiplier
from numerics import fine_action


def fixed_transform(value, owner):
    output=torch.empty_like(value)
    owner._execute(0,value,output)
    # FFTW's raw DCT-II -> orthonormal DCT-II. Transform normalization only.
    for axis,size in enumerate(value.shape[2:]):
        scale=torch.full((size,),math.sqrt(2/size)/2,dtype=torch.float64)
        scale[0]=math.sqrt(1/size)/2
        shape=[1]*5;shape[axis+2]=size
        output.mul_(scale.reshape(shape))
    return output


def run(args):
    if args.output.exists():raise ValueError('Fresh expanded fixed-feature view required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered fixed view dependency changed: '+path)
    if digest(reg['library_path'])!=reg['library_sha256']:raise ValueError('Fixed transform library changed')
    bank=read(ROOT/reg['bank'])
    if bank['scene_count']!=148 or bank['split']!='train' or any(row['split']!='train' for row in bank['trajectories']):
        raise ValueError('All148training scenes and no development/reserved data required')
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    args.output=args.output.resolve();args.output.mkdir(parents=True)
    began=time.monotonic();sources={};rows=[];arrays={name:[] for name in ('features','q','p','normalizer','example_weight','scene_index','step','mode_zyx')}
    bytes_read=0;owners={}
    progress=dict(status='running',active=None,scenes_completed=0,requests_considered=0,requests_with_signal=0,mode_rows=0,
        optimizer_updates=0,learned_model_forward_executed=False,source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol))
    write(args.output/'progress.json',progress)
    try:
        with torch.inference_mode():
            for scene_index,selected in enumerate(bank['trajectories']):
                path=ROOT/selected['receipt']
                if digest(path)!=selected['receipt_sha256']:raise ValueError('Train reference receipt changed')
                receipt=read(path);scene=receipt['scene'];shape=tuple(reversed(scene['counts']))
                if receipt['split']!='train' or selected['reference_substeps']!=receipt.get('reference_substeps',1):
                    raise ValueError('One registered training lineage required')
                if shape not in owners:owners[shape]=DCTMultiplier((1,1,*shape),reg['library_path'])
                owner=owners[shape];states={entry['step']:entry for entry in selected['states']};samples=[]
                progress['active']=dict(identifier=selected['identifier']);write(args.output/'progress.json',progress)
                for step in reg['states']:
                    if time.monotonic()-began>reg['maximum_wall_seconds']:raise TimeoutError('Finite fixed-view preprocessing budget reached')
                    entry=states[step];source=ROOT/entry['path']
                    if digest(source)!=entry['sha256']:raise ValueError('Actual train state changed before preprocessing')
                    bytes_read+=source.stat().st_size
                    if bytes_read>reg['maximum_unique_archive_bytes']:raise ValueError('Finite archive population byte cap reached')
                    with np.load(source,allow_pickle=False) as archive:
                        raw_p=archive['pressure'];raw_rhs=archive['rhs']
                        if raw_p.dtype.str!='<f8' or raw_rhs.dtype.str!='<f8' or raw_p.shape!=(math.prod(shape),) or raw_rhs.shape!=raw_p.shape:
                            raise ValueError('Complete native double pressure/request labels required')
                        for name,raw in [('pressure',raw_p),('rhs',raw_rhs)]:
                            if not np.isfinite(raw).all() or hashlib.sha256(raw.tobytes()).hexdigest()!=entry['fields'][name]['sha256']:
                                raise ValueError('Actual decoded training field changed')
                        p=torch.empty((1,1,*shape),dtype=torch.float64);p.copy_(torch.from_numpy(raw_p).reshape_as(p))
                        rhs=torch.empty_like(p);rhs.copy_(torch.from_numpy(raw_rhs).reshape_as(rhs))
                    h=torch.tensor([1/scene['cells_per_metre']],dtype=torch.float64);rho=torch.tensor([1000.],dtype=torch.float64)
                    error=float((fine_action(p,h,rho)-rhs).norm()/rhs.norm().clamp_min(1e-30))
                    if error>reg['maximum_native_control_residual']:raise ValueError('Offline native request/pressure identity failed')
                    q=torch.empty_like(rhs);q.copy_(rhs*(1000*scene['cells_per_metre']))
                    qc=fixed_transform(q,owner).numpy().reshape(-1);pc=fixed_transform(p,owner).numpy().reshape(-1)
                    progress['requests_considered']+=1
                    maximum_q=float(np.abs(qc[1:]).max());maximum_p=float(np.abs(pc[1:]).max())
                    admitted=np.flatnonzero((np.abs(qc)>=reg['relative_signal_floor']*maximum_q)&(np.abs(pc)>=reg['relative_signal_floor']*maximum_p))
                    admitted=admitted[admitted!=0] if maximum_q>0 and maximum_p>0 else np.empty(0,dtype=np.int64)
                    sample=dict(identifier=selected['identifier'],scene_index=scene_index,step=step,source=entry['path'],source_sha256=entry['sha256'],
                        reference_substeps=selected['reference_substeps'],output_dt_s=selected['output_dt_s'],
                        cells_per_metre=scene['cells_per_metre'],counts_xyz=scene['counts'],native_control_residual=error,
                        admitted_modes=int(len(admitted)),sampled_modes=0,skipped_zero_signal=len(admitted)==0)
                    if len(admitted):
                        count=min(reg['maximum_modes_per_request'],len(admitted));top_count=count//2
                        # Half strongest request coefficients, half seeded signal-admitted modes.
                        strongest=admitted[np.argsort(-np.abs(qc[admitted]),kind='stable')[:top_count]]
                        remaining=np.setdiff1d(admitted,strongest,assume_unique=True)
                        rng=np.random.default_rng(reg['sampling_seed']+scene_index*10007+step)
                        chosen=np.sort(np.concatenate([strongest,rng.choice(remaining,size=count-top_count,replace=False)]))
                        mode=np.stack(np.unravel_index(chosen,shape),axis=1).astype('<i4')
                        features=np.stack([np.ones(count),*(np.cos(math.pi*mode[:,axis]/shape[axis]) for axis in (2,1,0))],axis=1).astype('<f8')
                        normalizer=float(np.sqrt(np.mean(qc[1:]**2)))
                        if not math.isfinite(normalizer) or normalizer<=0:raise ValueError('Finite positive train-only request RMS required')
                        sample.update(sampled_modes=count,request_q_coefficients_rms=normalizer)
                        samples.append((sample,features,qc[chosen].copy(),pc[chosen].copy(),mode,normalizer))
                        progress['requests_with_signal']+=1;progress['mode_rows']+=count
                    rows.append(sample);sources[entry['path']]=entry['sha256']
                    if digest(source)!=entry['sha256']:raise ValueError('Actual train archive changed during preprocessing')
                if not samples:raise ValueError('Every registered training scene must contribute actual signal; no silent scene omission')
                for sample,features,q_values,p_values,mode,normalizer in samples:
                    count=len(q_values)
                    for name,value in dict(features=features,q=q_values,p=p_values,normalizer=np.full(count,normalizer,dtype='<f8'),
                        example_weight=np.full(count,1/(148*len(samples)*count),dtype='<f8'),
                        scene_index=np.full(count,scene_index,dtype='<i4'),step=np.full(count,sample['step'],dtype='<i4'),mode_zyx=mode).items():arrays[name].append(value)
                progress['scenes_completed']+=1;progress['active']=None;write(args.output/'progress.json',progress)
                print('Prepared train-only native spectral features',selected['identifier'],'nonzero requests',len(samples),flush=True)
        combined={name:np.concatenate(values,axis=0) for name,values in arrays.items()}
        if len(np.unique(combined['scene_index']))!=148 or not all(np.isfinite(value).all() for value in combined.values()):
            raise ValueError('Complete finite all-scene contribution required')
        if not np.isclose(combined['example_weight'].sum(),1,rtol=1e-12,atol=1e-12):raise ValueError('Equal-scene objective weights required')
        part=args.output/'part-00.npz';np.savez_compressed(part,**combined)
        if part.stat().st_size>reg['maximum_output_bytes']:raise ValueError('Finite feature artifact cap reached')
        normalization=dict(schema='rheon.shared-learned-spectral-view.v1',basis='Orthonormal real DCT-II',
            features=['constant_one','cos(pi*kx/Nx)','cos(pi*ky/Ny)','cos(pi*kz/Nz)'],
            q_units='native full same-request rhs *1000*cells_per_metre',pressure_dtype='<f8',
            supervision='Actual native same-request pressure and q coefficients; not macro future-transition supervision.',
            loss='Equal-scene weighted squared residual (D_learned(features)*P_coefficient-Q_coefficient)/request_Q_RMS',
            feature_coefficients_are_not_fitted=True,pressure_over_q_ratios_computed=False,
            fitted_normalization=False,development_or_reserved_used=False)
        write(args.output/'normalization.json',normalization)
        write(args.output/'manifest.json',dict(status='verified_all148train_fixed_shared_spectral_view',
            training_scenes=148,training_groups=len({r['group'] for r in bank['trajectories']}),geometries=len(owners),
            source_scene_identifiers=[r['identifier'] for r in bank['trajectories']],source_samples=rows,
            requests_considered=progress['requests_considered'],requests_with_signal=progress['requests_with_signal'],mode_rows=progress['mode_rows'],
            source_evidence=sources,logical_unique_archive_bytes=bytes_read,
            io_scope='Logical source population size, counted once per archive; hash-before/decode/hash-after reads are additional I/O.',
            arrays={name:dict(dtype=value.dtype.str,shape=list(value.shape)) for name,value in combined.items()},
            parts=[dict(path=part.name,sha256=digest(part),bytes=part.stat().st_size)],
            normalization_sha256=digest(args.output/'normalization.json'),source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
            optimizer_updates=0,learned_model_forward_executed=False,analytic_pressure_inverse_constructed=False,
            physical_eigenvalues_constructed=False,development_or_reserved_used=False,
            future_transition_lineages_pooled=False,reference_temporal_convergence_claim=False))
        for path,expected in reg['source_sha256'].items():
            if digest(ROOT/path)!=expected:raise ValueError('Fixed-view source changed during preprocessing: '+path)
        progress.update(status='completed_all148train_shared_spectral_view',active=None,elapsed_seconds=time.monotonic()-began)
    except BaseException as error:
        progress.update(status='failed',error=repr(error));raise
    finally:
        for owner in owners.values():owner.close()
        progress['active']=None;write(args.output/'progress.json',progress)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
