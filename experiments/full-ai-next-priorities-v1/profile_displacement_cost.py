"""Instrument the restored displacement model without changing its prediction."""
import argparse
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import bundle_model_displacement as model_module
from displacement_evaluation_support import load_model
from temporal_train_displacement import inventory
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh displacement stage profile required')
    protocol_path=DOCS/'displacement-CPU-stage-profile-protocol-01.json';reg=read(protocol_path)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered profile/inference source changed')
    benchmark=read(BASE/'temporal-displacement-timing-01/execution.json')
    diagnosis=read(BASE/'displacement-component-diagnostic-batch-02/execution.json')
    if benchmark['status']!='completed'or not all(w['reaped']for w in benchmark['workers'])or diagnosis['status']=='running'or not diagnosis['owned_child_reaped']:
        raise ValueError('Prior numerical workers must finish and reap before isolated profiling')
    os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    network,norm,identity=load_model(BASE/'b2-displacement-pair-01');before=inventory(network);cases=[];sources={}
    for row in read(BASE/'temporal-development-240-01/bank.json')['trajectories']:
        path=ROOT/row['receipt']
        if sha(path)!=row['receipt_sha256']:raise ValueError('Development receipt changed')
        receipt=read(path);scene=receipt['scene'];sources[str(path.relative_to(ROOT))]=sha(path)
        for step in reg['starts']:
            spec=receipt['states'][step];source=path.parent/spec['path']
            if sha(source)!=spec['sha256']:raise ValueError('Current state changed')
            sources[str(source.relative_to(ROOT))]=sha(source)
            with np.load(source,allow_pickle=False)as a:fields={n:a[n].copy()for n in ('pressure','vx','vy','vz','tracer')}
            cases.append(dict(identifier=row['identifier'],scene=scene,step=step,fields=fields))
    def predict(case):return network.predict(case['fields'],case['scene'],case['step'],norm)
    names=('fluid','local_update','displacement','displacement.context','displacement.curvature','viscous_head','flux','pressure_core','momentum','network')
    clocks={};elapsed={};counts={};handles=[];stages=[]
    def begin(name):
        def hook(*args):clocks[name]=time.perf_counter()
        return hook
    def end(name):
        def hook(*args):
            elapsed[name]=elapsed.get(name,0.)+time.perf_counter()-clocks[name];counts[name]=counts.get(name,0)+1
        return hook
    original_readout=model_module.convex_field_readout
    def observed_readout(*args,**kwargs):
        tick=time.perf_counter()
        try:return original_readout(*args,**kwargs)
        finally:
            elapsed['field_readout']=elapsed.get('field_readout',0.)+time.perf_counter()-tick
            counts['field_readout']=counts.get('field_readout',0)+1
    with torch.inference_mode():
        reference=[predict(c)for c in cases]
        for name in names:
            module=network if name=='network'else network.get_submodule(name)
            handles.extend((module.register_forward_pre_hook(begin(name)),module.register_forward_hook(end(name))))
        model_module.convex_field_readout=observed_readout
        try:
            for case,expected in zip(cases,reference):
                for repeat in range(reg['repeats']):
                    clocks.clear();elapsed.clear();counts.clear();tick=time.perf_counter();out=predict(case);full=time.perf_counter()-tick
                    if any(not torch.equal(a[n],b[n])for a,b in zip(out,expected)for n in a):raise ValueError('Instrumentation changed predictions')
                    if counts['network']!=1 or counts['fluid']!=1 or any(counts[n]!=2 for n in ('local_update','displacement','viscous_head','flux','pressure_core','momentum','field_readout')):
                        raise ValueError('Actual shared encoder/field-readout count differs')
                    stages.append(dict(identifier=case['identifier'],step=case['step'],counts_xyz=case['scene']['counts'],repeat=repeat,
                        complete_profiled_seconds=full,module_inclusive_seconds=dict(elapsed),module_calls=dict(counts),
                        preparation_and_reconstruction_seconds=full-elapsed['network']))
        finally:
            model_module.convex_field_readout=original_readout
            for handle in handles:handle.remove()
    if inventory(network)!=before or torch.cuda.is_initialized():raise ValueError('Profile changed weights or initialized CUDA')
    summaries=[]
    for regime in ('small','32cubed'):
        selected=[s for s in stages if(max(s['counts_xyz'])<=16)==(regime=='small')]
        for name in (*names,'field_readout'):
            summaries.append(dict(regime=regime,module=name,median_inclusive_block_ms=statistics.median(s['module_inclusive_seconds'][name]for s in selected)*1000))
        summaries.append(dict(regime=regime,module='physical_preparation_and_reconstruction',
            median_inclusive_block_ms=statistics.median(s['preparation_and_reconstruction_seconds']for s in selected)*1000))
    write(args.output,dict(status='completed_diagnostic_only',model=identity,stages=stages,stage_summaries=summaries,
        source_evidence=sources,source_sha256=reg['source_sha256'],protocol_sha256=sha(protocol_path),
        predictions_unchanged=True,weights_unchanged=True,optimizer_updates=0,CPU=23,threads=1,
        inclusive_times_not_additive=True,profile_overhead_is_not_benchmark_latency=True,
        candidate_frontier_point=False,reserved_scenes_consumed=False))
    print('Profiled actual displacement field readout and neural stages; predictions/weights unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
