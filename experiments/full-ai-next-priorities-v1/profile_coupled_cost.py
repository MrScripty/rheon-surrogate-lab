"""CPU stage profile and preliminary denormal-policy screen; no weight changes."""
import argparse
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from coupled_evaluation_support import load_model
from temporal_train_coupled import inventory
from temporal_metrics import metrics,FIELDS
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    if args.output.exists():raise ValueError('Fresh profiling receipt required')
    os.sched_setaffinity(0,{23});torch.set_num_threads(1)
    if not torch.set_flush_denormal(False):raise ValueError('Registered denormal policy unavailable')
    protocol=DOCS/'coupled-CPU-stage-profile-protocol-01.json';reg=read(protocol)
    for name,digest in reg['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered profiler/inference source changed')
    benchmark=read(BASE/'temporal-provisional-timing-02/execution.json')
    if benchmark['status']!='completed'or not all(w['reaped']for w in benchmark['workers']):raise ValueError('Serial benchmark must finish before profiling')
    network,norm,identity=load_model(BASE/'b2-provisional-aux-pair-01');before=inventory(network);cases=[];sources={}
    bank=read(BASE/'temporal-development-240-01/bank.json')
    for row in bank['trajectories']:
        path=ROOT/row['receipt']
        if sha(path)!=row['receipt_sha256']:raise ValueError('Reference receipt changed')
        receipt=read(path);scene=receipt['scene'];sources[str(path.relative_to(ROOT))]=sha(path)
        for step in (0,63):
            values=[]
            for actual in (step,step+1,step+2):
                spec=receipt['states'][actual];source=path.parent/spec['path']
                if sha(source)!=spec['sha256']:raise ValueError('Current/reference field changed')
                sources[str(source.relative_to(ROOT))]=sha(source)
                with np.load(source,allow_pickle=False)as arrays:values.append({n:arrays[n].copy()for n in (*FIELDS,'rhs')})
            cases.append(dict(identifier=row['identifier'],scene=scene,step=step,current={n:values[0][n]for n in FIELDS},targets=values[1:]))
    def predict(case):return network.predict(case['current'],case['scene'],case['step'],norm)
    timings=[];outputs={False:[],True:[]};parity=[];stages=[]
    try:
        with torch.inference_mode():
            for repeat in range(3):
                for mode in ((False,True)if repeat%2==0 else(True,False)):
                    torch.set_flush_denormal(mode)
                    for case in cases:predict(case)
                    for case in cases:
                        tick=time.perf_counter();out=predict(case);elapsed=time.perf_counter()-tick
                        timings.append(dict(flush_denormal=mode,repeat=repeat,identifier=case['identifier'],step=case['step'],counts_xyz=case['scene']['counts'],seconds=elapsed))
                        if repeat==0:outputs[mode].append(out)
            torch.set_flush_denormal(False)
            for case,baseline,changed in zip(cases,outputs[False],outputs[True]):
                for horizon,(a,b,target)in enumerate(zip(baseline,changed,case['targets']),1):
                    differences={n:float((a[n]-b[n]).double().norm())/max(float(a[n].double().norm()),1e-30)for n in FIELDS}
                    original=metrics(a,target,target['rhs'],case['scene'],norm,0.)
                    variant=metrics(b,target,target['rhs'],case['scene'],norm,0.)
                    parity.append(dict(identifier=case['identifier'],step=case['step'],horizon=horizon,
                        field_relative_difference=differences,accuracy_classification_unchanged=original['all_accuracy_criteria_pass']==variant['all_accuracy_criteria_pass']))
            names=('fluid','left','right','local_update','transport_context','transport_head','viscous_head','tracer_readout','flux','pressure_core','momentum')
            clocks={};elapsed={};counts={};handles=[]
            def begin(name):
                def f(module,args):clocks[name]=time.perf_counter()
                return f
            def end(name):
                def f(module,args,out):
                    elapsed[name]=elapsed.get(name,0.)+time.perf_counter()-clocks[name];counts[name]=counts.get(name,0)+1
                return f
            for name in (*names,'network'):
                module=network if name=='network'else getattr(network,name)
                handles.extend([module.register_forward_pre_hook(begin(name)),module.register_forward_hook(end(name))])
            try:
                for case in cases:
                    for repeat in range(3):
                        clocks.clear();elapsed.clear();counts.clear();tick=time.perf_counter();predict(case);full=time.perf_counter()-tick
                        stages.append(dict(identifier=case['identifier'],step=case['step'],counts_xyz=case['scene']['counts'],
                            repeat=repeat,complete_profiled_seconds=full,module_inclusive_seconds=dict(elapsed),module_calls=dict(counts),
                            preparation_and_reconstruction_seconds=full-elapsed['network']))
            finally:
                for handle in handles:handle.remove()
    finally:torch.set_flush_denormal(False)
    if inventory(network)!=before or torch.cuda.is_initialized():raise ValueError('Profiling modified model or initialized CUDA')
    summaries=[];stage_summaries=[]
    for regime in ('small','32cubed'):
        for mode in (False,True):
            selected=[r for r in timings if(max(r['counts_xyz'])<=16)==(regime=='small')and r['flush_denormal']==mode]
            summaries.append(dict(regime=regime,flush_denormal=mode,median_block_ms=statistics.median(r['seconds']for r in selected)*1000))
        selected=[r for r in stages if(max(r['counts_xyz'])<=16)==(regime=='small')]
        for name in (*names,'network'):
            stage_summaries.append(dict(regime=regime,module=name,median_inclusive_block_ms=statistics.median(r['module_inclusive_seconds'][name]for r in selected)*1000))
    write(args.output,dict(status='completed_diagnostic_only',model=identity,preliminary_variant_timings=summaries,stage_summaries=stage_summaries,
        parity=parity,timings=timings,stages=stages,source_evidence=sources,source_sha256=reg['source_sha256'],protocol_sha256=sha(protocol),
        weights_unchanged=True,optimizer_updates=0,CPU=23,threads=1,
        flush_variant_fully_qualified=False,own_history_variant_qualification=False,profile_overhead_is_not_benchmark_latency=True,
        candidate_frontier_point=False,reserved_scenes_consumed=False,original_goal_complete=False))
    print('Profiled coupled CPU stages and preliminary denormal execution policy; weights unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
