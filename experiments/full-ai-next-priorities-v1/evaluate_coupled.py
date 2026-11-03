"""Evaluate every joint output and autonomous 240-step continuation on native scenes."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from study_support import ROOT,HERE,BASE,read,write,sha
from temporal_metrics import metrics,FIELDS,physical_arrays


from coupled_evaluation_support import load_model


def references(row):
    path=ROOT/row['receipt']
    if sha(path)!=row['receipt_sha256']:raise ValueError('Reference receipt changed')
    receipt=read(path);states=[]
    for index,spec in enumerate(receipt['states']):
        source=path.parent/spec['path']
        if sha(source)!=spec['sha256']or spec['step']!=index or spec['time_s']!=index/60:
            raise ValueError('Full reference state or timestamp changed')
        with np.load(source,allow_pickle=False)as arrays:
            values={name:arrays[name].copy()for name in (*FIELDS,'rhs')}
        for name,value in values.items():
            if hashlib.sha256(value.tobytes()).hexdigest()!=spec['arrays'][name]['sha256']:
                raise ValueError('Full reference field changed')
        states.append(values)
    return receipt['scene'],states


def run(args):
    if args.output.exists():raise ValueError('Fresh temporal evaluation required')
    os.sched_setaffinity(0,{23});torch.set_num_threads(1);args.output.mkdir(parents=True)
    network,norm,identity=load_model(args.pair.resolve());k=network.horizons
    bank=read(args.bank);rows=[]
    if bank['status']!='completed':raise ValueError('Complete native reference bank required')
    for row in bank['trajectories']:
        if row['split']not in ('development','reserved'):raise ValueError('Evaluation-only scene bank required')
        scene,truth=references(row);initial={name:truth[0][name]for name in FIELDS}
        initial_mass=float(truth[0]['tracer'].astype('f8').sum()/scene['cells_per_metre']**3)
        qualification=metrics({name:truth[8][name]for name in FIELDS},truth[8],truth[8]['rhs'],scene,norm,initial_mass)
        if any(v!=0 for v in qualification['absolute_rms'].values())or qualification['relative']['equation']>1e-6:
            raise ValueError('Reference identity or pressure/RHS alignment qualification failed')
        scene_rows=[];started=time.perf_counter()
        with torch.inference_mode():
            for start in range(121):
                predictions=network.predict({name:truth[start][name]for name in FIELDS},scene,start,norm)
                if len(predictions)!=k:raise ValueError('Every intermediate horizon required')
                for horizon,predicted in enumerate(predictions,1):
                    target=truth[start+horizon]
                    result=metrics(predicted,target,target['rhs'],scene,norm,initial_mass)
                    scene_rows.append(dict(mode='single_block',start=start,horizon=horizon,step=start+horizon,metrics=result))
            current=initial;first_invalid=None;inference_failures=[];thresholds={name:{str(level):None for level in (1e-6,1e-5,1e-4,1e-3,1e-2)}for name in ('pressure','velocity','tracer','equation')}
            for start in range(0,240,k):
                try:
                    predictions=network.predict(current,scene,start,norm)
                except (ValueError,FloatingPointError)as failure:
                    known={'Finite input and colocated float64 neural map required','Finite request on the component device required'}
                    if not isinstance(failure,FloatingPointError)and str(failure)not in known:raise
                    first_invalid=start+1
                    inference_failures.append(dict(start=start,first_unavailable_step=first_invalid,
                        requested_states=k,error_type=type(failure).__name__,detail=str(failure),
                        returned_states=0,fallback_used=False))
                    break
                for horizon,predicted in enumerate(predictions,1):
                    step=start+horizon;target=truth[step]
                    if not all(bool(torch.isfinite(v).all())for v in predicted.values()):
                        first_invalid=step;break
                    result=metrics(predicted,target,target['rhs'],scene,norm,initial_mass)
                    scene_rows.append(dict(mode='own_history',start=start,horizon=horizon,step=step,metrics=result))
                    for name,levels in thresholds.items():
                        for level,first in levels.items():
                            if first is None and result['relative'][name]>float(level):levels[level]=dict(step=step,time_s=step/60)
                if first_invalid is not None:break
                current=predictions[-1]
        def summarize(mode,limit):
            selected=[r for r in scene_rows if r['mode']==mode and r['step']<=limit]
            return dict(mode=mode,interval_steps=limit,outputs=len(selected),
                complete=mode=='single_block'or(first_invalid is None or first_invalid>limit),
                worst={name:max(r['metrics']['relative'][name]for r in selected)for name in thresholds},
                all_accuracy_criteria_pass=all(r['metrics']['all_accuracy_criteria_pass']for r in selected)and(first_invalid is None or first_invalid>limit))
        summary=dict(identifier=row['identifier'],counts_xyz=scene['counts'],cells_per_metre=scene['cells_per_metre'],
            split=row['split'],first_nonfinite_step=first_invalid if not inference_failures else None,
            first_unavailable_step=first_invalid,inference_failures=inference_failures,
            worst_scope='All outputs'if first_invalid is None else'Observed finite prefix only; complete accuracy fails',diagnostic_drift_thresholds=thresholds,
            application_acceptance_limits=False,reference_identity_qualification=qualification,
            summaries=[summarize('single_block',124)]+[summarize('own_history',limit)for limit in (12,48,240)],
            elapsed_s=time.perf_counter()-started)
        write(args.output/(row['identifier']+'.json'),dict(**summary,rows=scene_rows))
        rows.append(summary);write(args.output/'progress.json',dict(status='running',model=identity,scenes=rows))
        print('Evaluated temporal K',k,row['identifier'],'single-block and own-history through240',flush=True)
    write(args.output/'summary.json',dict(status='completed',model=identity,bank_sha256=sha(args.bank),scenes=rows,
        source_sha256=sha(__file__),raw_failure_accounting=True,loader_sha256=sha(HERE/'coupled_evaluation_support.py'),metrics_sha256=sha(HERE/'temporal_metrics.py'),cuda_initialized=torch.cuda.is_initialized(),
        candidate_numerical_pressure_solve=False,optimizer_updates=0,model_promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','bank','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
