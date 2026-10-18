"""CPU-only fixed-pass snapshots over the exact consumed validation population."""
import argparse
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

from common import ROOT, read, sha, write


def material(config_path, projects, output, include_candidates):
    config=read(config_path)
    captured=read(ROOT/config['baseline_material'])
    result={**captured,'models':[m for m in captured['models'] if m['id'] in config['baselines']]}
    result['config']={**captured['config'],'passes':config['validation_passes'],'timing_repetitions':3,'warmup_each_case':True}
    if include_candidates:
        for number,arm in enumerate(config['arms'],57):
            directory=projects/(arm['id']+'-study-01')
            receipt_path=directory/'receipt.json'
            if not receipt_path.exists():
                continue
            receipt=read(receipt_path)
            if receipt['status']!='completed' or receipt['optimizer_updates']!=config['updates'] or not all(receipt['artifact_parity'].values()):
                raise ValueError('Completed CUDA/fresh CPU restore required')
            project=projects/(arm['id']+'-project-01')
            restore=directory/'restore'
            identifier=arm.get('model_id','C'+str(number))
            if any(m['id']==identifier for m in result['models']):
                raise ValueError('Candidate ID already belongs to another checkpoint: '+identifier)
            result['models'].append(dict(id=identifier,name=arm['id'],project=str(project.relative_to(ROOT)),
                   restore=str(restore.relative_to(ROOT)),weights_sha256=sha(restore/'weights.npz'),tags=[config['schema']],
                   identities={str(p.relative_to(ROOT)):sha(p)for p in [project/'config.json',project/'source-witness.json',
                       restore/'weights.npz',restore/'normalization.json',restore/'receipt.json',restore/'history.json']},
                   training_receipt_source=str(receipt_path.relative_to(ROOT)),training_receipt_sha256=sha(receipt_path)))
    write(output,result)
    print('Frozen CPU material',[(m['id'],m['name'])for m in result['models']],flush=True)


def snapshots(material_path,identifier,cpu,output):
    started=time.perf_counter();os.sched_setaffinity(0,{cpu})
    sys.path.insert(0,str(ROOT/'experiments/pressure-pass-sweep-v1'))
    import inference
    selected=read(material_path);state=inference.load(selected,identifier);torch=state.torch
    inputs=ROOT/selected['config']['validation_inputs']
    for name,identity in selected['validation_sources'].items():
        if sha(inputs/name)!=identity:
            raise ValueError('Consumed validation population changed')
    _,parents=state.data.load(tuple((inputs/name).read_bytes()for name in ['validation_dataset.npz','validation_dataset_extra.npz']),read(inputs/'validation_splits.json'))
    cases=[(parent,slot,dict(trajectory_id=parent['entry']['trajectory_id'],native_frame=frame['index'],
             counts_xyz=parent['counts_xyz'],cell_size_m=parent['cell_size_m']))for parent in parents for slot,frame in enumerate(parent['frames'])]
    expected=sorted((c['trajectory_id'],c.get('native_frame',c.get('frame')),tuple(c['counts_xyz']),c['cell_size_m'])for c in selected['validation_identities'])
    actual=sorted((c['trajectory_id'],c['native_frame'],tuple(c['counts_xyz']),c['cell_size_m'])for _,_,c in cases)
    if expected!=actual or len(cases)!=128:
        raise ValueError('Exact consumed 128-case membership required')
    output.mkdir(parents=True,exist_ok=False);rows=[]
    with torch.no_grad():
        for index,(parent,slot,identity)in enumerate(cases):
            sample=state.data.sample(parent,slot)
            inference.fixed(state,sample,1)
            values={n:[]for n in selected['config']['passes']}
            for repetition in range(selected['config']['timing_repetitions']):
                pressure=sample['base'];elapsed=0.
                for n in range(1,max(values)+1):
                    tick=time.perf_counter();pressure=inference.step(state,sample,pressure);elapsed+=time.perf_counter()-tick
                    if n not in values:
                        continue
                    tick=time.perf_counter()
                    error=sample['rhs']-state.model.full_operator(pressure,sample['cell_size_m'],sample['rho'])
                    residual=float(torch.linalg.vector_norm(error))/max(float(torch.linalg.vector_norm(sample['rhs'])),1e-300)
                    quality_s=time.perf_counter()-tick
                    delta=pressure-sample['reference'];centered=delta-delta.mean();reference=sample['reference']-sample['reference'].mean()
                    values[n].append(dict(relative_residual=residual,inference_s=elapsed+quality_s,
                        same_request_pressure_rms_pa=float(delta.square().mean().sqrt()),
                        same_request_mean_free_pressure_relative_l2=float(torch.linalg.vector_norm(centered))/max(float(torch.linalg.vector_norm(reference)),1e-300)))
            for n,measurements in values.items():
                rows.append(dict(**identity,passes=n,relative_residual=measurements[-1]['relative_residual'],
                            same_request_pressure_rms_pa=measurements[-1]['same_request_pressure_rms_pa'],
                            same_request_mean_free_pressure_relative_l2=measurements[-1]['same_request_mean_free_pressure_relative_l2'],
                            inference_s=statistics.median(v['inference_s']for v in measurements),timing_samples_s=[v['inference_s']for v in measurements]))
            if index%16==0:
                write(output/'progress.json',dict(completed=index+1,total=128))
    if torch.cuda.is_initialized() or any(not math.isfinite(r['relative_residual'])for r in rows):
        raise ValueError('CPU-only finite pressure evaluation required')
    summary=[]
    for n in selected['config']['passes']:
        group=[r for r in rows if r['passes']==n]
        summary.append(dict(passes=n,cases=128,equation_passes=sum(r['relative_residual']<=1e-6 for r in group),
             worst_equation_residual=max(r['relative_residual']for r in group),
             median_equation_residual=statistics.median(r['relative_residual']for r in group),
             median_inference_ms=1000*statistics.median(r['inference_s']for r in group),
             worst_same_request_pressure_rms_pa=max(r['same_request_pressure_rms_pa']for r in group)))
    write(output/'snapshots.json',dict(status='passed',model=identifier,summary=summary,rows=rows,cuda_initialized=False,
           torch_threads=torch.get_num_threads(),cpu=cpu,elapsed_s=time.perf_counter()-started,
           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
           source_sha256=sha(__file__),material_sha256=sha(material_path),weights_sha256=state.entry['weights_sha256'],
           timing_scope='CPU fixed-pass recurrence plus final equation check; excludes input assembly and pressure-reference comparison. Concurrent and isolated phases are recorded separately.'))
    print(identifier,summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    create=modes.add_parser('material')
    create.add_argument('--config',type=Path,required=True);create.add_argument('--projects',type=Path,required=True)
    create.add_argument('--output',type=Path,required=True);create.add_argument('--include-candidates',action='store_true')
    evaluate=modes.add_parser('snapshots')
    evaluate.add_argument('--material',type=Path,required=True);evaluate.add_argument('--id',required=True)
    evaluate.add_argument('--cpu',type=int,required=True);evaluate.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='material':
        material(args.config.resolve(),args.projects.resolve(),args.output.resolve(),args.include_candidates)
    else:
        snapshots(args.material.resolve(),args.id,args.cpu,args.output.resolve())
