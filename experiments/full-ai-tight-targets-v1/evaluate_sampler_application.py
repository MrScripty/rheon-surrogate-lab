"""Deferred full-field accuracy after all matched application clocks finish."""
import argparse
import os
from pathlib import Path
from profile_frozen_components import ROOT,read,write,digest
from diagnose_causal_composition import measure,FIELDS
import numpy as np
import torch


def run(args):
    if args.output.exists():raise ValueError('Fresh deferred application evaluation required')
    reg=read(args.protocol)
    for path,expected in reg['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Registered deferred accuracy source changed: '+path)
    result=read(args.application)
    if result['status']!='completed_matched_application_comparison':raise ValueError('Every application clock must finish first')
    os.sched_setaffinity(0,{reg['cpu']});torch.set_num_threads(1)
    norm=read(ROOT/reg['normalization']);references={};cache={};rows=[];sources={str(args.application.relative_to(ROOT)):digest(args.application)}
    for scenario,bank_path in reg['reference_banks'].items():
        for trajectory in read(ROOT/bank_path)['trajectories']:
            identifier=trajectory.get('original_identifier',trajectory['identifier'])
            if identifier not in reg['identifiers']:continue
            path=ROOT/trajectory['receipt']
            if digest(path)!=trajectory['receipt_sha256']:raise ValueError('Deferred native reference receipt changed')
            references[scenario,identifier]=(path,read(path))
    for row in result['rows']:
        if row['warmup']:continue
        spec=row['physical_output_evidence'];path=ROOT/spec['path'];key=(spec['sha256'],row['scenario'],row['identifier'])
        if key not in cache:
            if digest(path)!=spec['sha256']:raise ValueError('Actual post-clock application outputs changed')
            ref_path,receipt=references[row['scenario'],row['identifier']];scored=[]
            with np.load(path,allow_pickle=False)as archive:
                for step in spec['steps']:
                    native=receipt['states'][step];truth_path=ref_path.parent/native['path']
                    if digest(truth_path)!=native['sha256']:raise ValueError('Deferred native timestamp/field reference changed')
                    with np.load(truth_path,allow_pickle=False)as source:truth={name:source[name].copy()for name in (*FIELDS,'rhs')}
                    predicted={name:torch.from_numpy(archive[f'step{step}_{name}'].copy())for name in FIELDS}
                    score=measure(predicted,truth,receipt['scene'],norm)
                    scored.append(dict(step=step,metrics=score))
            cache[key]=scored;sources[spec['path']]=spec['sha256']
        scored=cache[key]
        if row['method'].startswith('sampler-')and len(scored)!=row['result']['physical_steps']:
            raise ValueError('Every consumed candidate state needs deferred physical scoring')
        rows.append(dict(identifier=row['identifier'],method=row['method'],scenario=row['scenario'],repeat=row['repeat'],
            physical_outputs_scored=len(scored),every_consumed_neural_state_scored=row['method'].startswith('sampler-'),
            rows=scored,worst={k:max((v['metrics']['relative'][k]for v in scored),default=None)for k in ('equation','pressure','velocity','tracer')}))
    summaries=[]
    for scenario in reg['scenarios']:
        for method in reg['methods']:
            chosen=[row for row in rows if row['scenario']==scenario and row['method']==method]
            worst={k:max((row['worst'][k]for row in chosen if row['worst'][k]is not None),default=None)for k in ('equation','pressure','velocity','tracer')}
            summaries.append(dict(method=method,scenario=scenario,worst=worst,all_fields_below_1e5=all(v is not None and v<=1e-5 for v in worst.values()),
                accuracy_scope='Every consumed forecast state'if method.startswith('sampler-')else'Native control diagnostic checkpoints'))
    write(args.output,dict(status='completed_deferred_application_accuracy',rows=rows,summaries=summaries,source_evidence=sources,
        application_results_sha256=digest(args.application),source_sha256=reg['source_sha256'],protocol_sha256=digest(args.protocol),
        optimizer_updates=0,all_metrics_outside_application_clocks=True,fresh_holdout_consumed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--application',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.application=args.application.resolve();run(args)
