"""Separate fitting error, unseen-scene error and cold-start error against persistence."""
import argparse
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from evaluate_temporal import load_model
from temporal_metrics import metrics,FIELDS


def run(args):
    if args.output.exists():raise ValueError('Fresh temporal accuracy diagnosis required')
    os.sched_setaffinity(0,{22});torch.set_num_threads(1)
    network,norm,identity=load_model(args.pair.resolve());bank=read(BASE/'temporal-data-01/bank.json');rows=[]
    with torch.inference_mode():
        for row in bank['trajectories']:
            if row['split']not in ('train','development'):raise ValueError('Reserved scenes excluded from diagnosis')
            path=ROOT/row['receipt']
            if sha(path)!=row['receipt_sha256']:raise ValueError('Native trajectory receipt changed')
            receipt=read(path);scene=receipt['scene']
            def state(step):
                spec=receipt['states'][step];source=path.parent/spec['path']
                if sha(source)!=spec['sha256']:raise ValueError('Reference state changed')
                with np.load(source,allow_pickle=False)as arrays:return {name:arrays[name].copy()for name in (*FIELDS,'rhs')}
            for start in (0,8,63,120):
                current=state(start);outputs=network.predict({name:current[name]for name in FIELDS},scene,start,norm)
                for horizon,predicted in enumerate(outputs,1):
                    target=state(start+horizon)
                    learned=metrics(predicted,target,target['rhs'],scene,norm,0.)
                    persistence=metrics({name:current[name]for name in FIELDS},target,target['rhs'],scene,norm,0.)
                    rows.append(dict(identifier=row['identifier'],split=row['split'],counts_xyz=scene['counts'],
                        start=start,horizon=horizon,learned=learned['relative'],persistence=persistence['relative']))
    summaries=[]
    for split in ('train','development'):
        for phase in ('cold','warm'):
            selected=[r for r in rows if r['split']==split and (r['start']==0)==(phase=='cold')]
            for name in ('pressure','equation','velocity','tracer'):
                summaries.append(dict(split=split,phase=phase,field=name,outputs=len(selected),
                    learned_median=statistics.median(r['learned'][name]for r in selected),
                    learned_worst=max(r['learned'][name]for r in selected),
                    persistence_median=statistics.median(r['persistence'][name]for r in selected),
                    persistence_worst=max(r['persistence'][name]for r in selected)))
    write(args.output,dict(status='completed',model=identity,rows=rows,summaries=summaries,
        source_sha256=sha(__file__),source_bank_sha256=sha(BASE/'temporal-data-01/bank.json'),
        reserved_scenes_consumed=False,optimizer_updates=0,
        caveat='Four fixed starting times per scene; fitting diagnosis rather than a replacement for all-window or240-step evaluation'))
    print('Diagnosed cold/warm fitting and unseen-scene errors against persistence',identity['horizons'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
