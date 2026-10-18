"""Project immutable CPU measurements and choose the next bounded proposal."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def report(directory,execution):
    e=read(execution)
    if e['status']!='completed'or any(not w['reaped']or w['returncode']!=0 for w in e['workers']):raise ValueError('Completed and reaped CPU workers required')
    models={};sources={};summary=[];limits=0
    for identifier in ['C53','C66','C67']:
        p=directory/'reports'/(identifier+'-components.json');d=read(p)
        if d['status']!='passed'or not d['inventory_unchanged']or d['cuda_initialized']or d['optimizer_updates']!=0:raise ValueError('Unchanged CPU-only diagnostic required')
        if not all(r['joint_gradient_exact']for r in d['cases']):raise ValueError('Exact joint-gradient equivalence required')
        limited=sum(not r['separate_gradient_precision_check_passed']for r in d['cases']);limits+=limited
        models[identifier]={k:v for k,v in d.items()if k not in ['vectors','sources']}
        source=str(p.relative_to(ROOT));sources[source]=sha(p)
        summary.append(dict(model=identifier,opposing_pairs={t:len(g['negative_pairs'])for t,g in d['across'].items()},
            strong_opposing_pairs={t:sum(r['cosine']<-.5 for r in g['negative_pairs'])for t,g in d['across'].items()},
            early_norm_exceeds_final=sum(r['geometry']['norms'][2]>r['geometry']['norms'][1]for r in d['cases']),
            precision_limited_cases=limited,worst_separate_gradient_sum_relative_l2=max(r['gradient_sum_relative_l2']for r in d['cases']),elapsed_s=d['elapsed_s']))
    sources[str(execution.relative_to(ROOT))]=sha(execution)
    decision=dict(status='diagnosis_completed_with_precision_limits',baseline_retained='C53',model_promoted=False,training_started=False,
        measured_cases=24,joint_gradient_equivalence_passes=24,separate_gradient_precision_limited_cases=limits,
        finding='Removing early feedback reduces opposing pairs but leaves strong final-solve disagreement across shapes. Early feedback also dominates several already-fitted cases; this is not a causal or Adam-dynamics result.',
        next_action='Propose a balanced eight-shape aggregate CUDA screen with matched example work; distinguish 200 aggregate optimizer updates from 1600 per-example updates. Retain early-pass guards and C53 deployment.')
    write(directory/'results.json',dict(schema='rheon-loss-components-results-v1',status='completed',models=models,summary=summary,execution=e,decision=decision,sources=sources))
    write(directory/'decision.json',decision);print('Recorded component diagnosis; precision-limited cases',limits,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--execution',type=Path,required=True)
    a=p.parse_args();report(a.directory.resolve(),a.execution.resolve())
