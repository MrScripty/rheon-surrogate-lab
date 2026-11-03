"""All148train-only CUDA stage features and small full-grid QR loss factors."""
import argparse
import collections
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np
import torch
from support import ROOT,read,write,digest
from macrostep_backbone_v1 import FrozenMacroBackbone
from macrostep_loss_design_v1 import design
from cuda_native_transfer_v1 import FIELDS
from diagnose_causal_composition import inventory


def run(args):
    if args.output.exists() or os.environ.get('TULDOK_TORCH_DEVICE')!='cuda':raise ValueError('Fresh GPU-only supervised view required')
    reg=read(args.protocol)
    for p,h in reg['source_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen view dependency changed: '+p)
    gate=read(ROOT/reg['backbone_qualification'])
    if gate['status']!='passed_macrostep_frozen_backbone_and_zero_readouts' or gate['existing_complete_outputs_exact']!=196:
        raise ValueError('Actual frozen-backbone qualification required')
    bank=read(ROOT/reg['bank']);cases=bank['trajectories']
    if bank['split']!='train' or len(cases)!=148 or any(c['split']!='train' for c in cases):raise ValueError('Every148training scene and no other split required')
    os.sched_setaffinity(0,{reg['host_cpu']});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True);started=time.monotonic()
    model=FrozenMacroBackbone(reg['transport_pair'],reg['shared_pair'],reg['build'],reg['control_build']);before=inventory(model)
    norm=read(ROOT/reg['normalization']);factors={s:[] for s in reg['student_stages']}
    progress=dict(status='running',active=None,rows=[],source_evidence={},scenes_completed=0,
        inference_device='cuda',display_label='GPU (CUDA)',gpu_model=torch.cuda.get_device_name(),
        cpu_model_forward=False,optimizer_updates=0,local_coefficient_fitting=False,
        expanded_reserved52_references_opened=0,latency_claim=False,development_or_reserved_used=False,
        protocol_sha256=digest(args.protocol),source_sha256=reg['source_sha256'])
    def save():write(args.output/'progress.json',progress)
    try:
        with torch.no_grad():
            for index,case in enumerate(cases):
                if digest(ROOT/case['receipt'])!=case['receipt_sha256']:raise ValueError('Training receipt changed')
                scene=read(ROOT/case['receipt'])['scene'];specs={s['step']:s for s in case['states']}
                def load(step):
                    spec=specs[step];p=ROOT/spec['path']
                    if digest(p)!=spec['sha256']:raise ValueError('Native training fields changed')
                    progress['source_evidence'][str(p.relative_to(ROOT))]=spec['sha256']
                    with np.load(p,allow_pickle=False) as a:return {n:a[n].copy() for n in (*FIELDS,'rhs')}
                per_stage={s:[] for s in reg['student_stages']}
                for start in reg['starts']:
                    initial=load(start);fields={n:torch.from_numpy(initial[n]).to('cuda') for n in FIELDS}
                    # No target is supplied to the causal feature generator.
                    feature_sets={}
                    if case['reference_substeps']==1:
                        frozen=model.feature_states(fields,scene,start,1)[0]
                        feature_sets={s:frozen for s in reg['student_stages']}
                    elif case['reference_substeps']==8:
                        for s in reg['student_stages']:feature_sets[s]=model.feature_states(fields,scene,start,s)[0]
                    else:raise ValueError('Original1/8teacher integration schemes required')
                    truth=load(start+1)
                    for s,features in feature_sets.items():
                        if any(not bool(v.isfinite().all()) for f in features for v in f.values()):raise ValueError('Nonfinite frozen student features')
                        qr,stats=design(features,truth,scene,norm,s,case['reference_substeps']==8,
                                        reg['compression_relative_tolerance'],reg['compression_absolute_tolerance'])
                        per_stage[s].append(qr.cpu().numpy().copy())
                        progress['rows'].append(dict(identifier=case['identifier'],scene_index=index,group=case['group'],
                            counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],start=start,student_stages=s,
                            teacher_substeps=case['reference_substeps'],trainable_readout_active=case['reference_substeps']==8,metrics=stats))
                    del feature_sets,features,fields,truth,initial
                    progress['active']=dict(identifier=case['identifier'],start=start);save()
                    if time.monotonic()-started>reg['maximum_wall_seconds']:raise TimeoutError('Finite supervised view deadline')
                for s in reg['student_stages']:factors[s].append(np.stack(per_stage[s]))
                progress['scenes_completed']+=1
                if (torch.cuda.max_memory_allocated()>reg['maximum_gpu_allocated_bytes']
                        or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>reg['maximum_peak_rss_bytes']):
                    raise ValueError('Supervised view memory ceiling exceeded')
                save();print('Macrostep train-only full-grid loss view',case['identifier'],'S3/S4contributed',flush=True)
        if progress['scenes_completed']!=148 or len(progress['rows'])!=592 or inventory(model)!=before:
            raise ValueError('All148scenes, two starts/two architectures and unchanged frozen assets required')
        parts=[]
        for s in reg['student_stages']:
            arrays=dict(design=np.stack(factors[s]).astype('<f8'),scene_index=np.arange(148,dtype='<i8'),
                        teacher_substeps=np.array([c['reference_substeps'] for c in cases],dtype='<i8'),starts=np.array(reg['starts'],dtype='<i8'))
            if arrays['design'].shape!=(148,2,4,s+1,s+1) or not np.isfinite(arrays['design']).all():raise ValueError('Exact finite full-population loss design required')
            path=args.output/f'readout-s{s}.npz';np.savez_compressed(path,**arrays)
            parts.append(dict(stages=s,path=path.name,sha256=digest(path),bytes=path.stat().st_size,
                              arrays={n:dict(dtype=a.dtype.str,shape=list(a.shape)) for n,a in arrays.items()}))
        shutil.copyfile(ROOT/reg['normalization'],args.output/'normalization.json')
        for p,h in reg['source_sha256'].items():
            if digest(ROOT/p)!=h:raise ValueError('View dependency changed during extraction: '+p)
        manifest=dict(status='verified_all148_train_macrostep_loss_view',training_scenes=148,training_N8_scenes=29,training_N1_scenes=119,
            development_or_reserved_used=False,starts=reg['starts'],metric_order=['equation','pressure','velocity','tracer'],
            metric_to_parameter_group=[0,0,1,2],parts=parts,scene_identifiers=[c['identifier'] for c in cases],
            normalization_sha256=digest(args.output/'normalization.json'),source_evidence=progress['source_evidence'],
            source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
            compression='Reduced QR of full normalized affine residual designs; no local coefficient solve or optimizer.',
            loss_precision_scope='Full F64 affine readout residuals before finalF32carrier casts; actual deployed-field accuracy must be independently scored after fitting.',
            inference_device='cuda',display_label='GPU (CUDA)',cpu_model_forward=False,optimizer_updates=0,expanded_reserved52_references_opened=0)
        write(args.output/'manifest.json',manifest)
        progress.update(status='completed_all148_train_macrostep_loss_view',active=None,frozen_assets_unchanged=True,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),observed_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_diagnostic_seconds=time.monotonic()-started,manifest_sha256=digest(args.output/'manifest.json'))
    except BaseException as e:progress.update(status='failed',error=repr(e));raise
    finally:model.close();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
