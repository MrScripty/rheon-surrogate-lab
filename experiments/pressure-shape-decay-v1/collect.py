"""Project immutable scalar receipts into the current offline comparison report."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def collect(directory,allow_partial=False):
    arms={};sources={}
    for name in ['shape-later-decay','joint-later-decay']:
        path=directory/'reports'/name/'result.json'
        if not path.exists():continue
        full=read(path);t=full['training'][0];identifier=t['id']
        control=full['matched_comparison'];source=ROOT/control['control_result_source'];previous=read(source)
        if sha(source)!=control['control_result_sha256']:raise ValueError('Retained control result changed')
        previous_id=control['control']
        quality=full['fitted'];quality={**quality,previous_id:previous['fitted'][previous_id]}
        summaries={i:{'summary':v['summary']}for i,v in full['snapshots'].items()}
        summaries[previous_id]={'summary':previous['snapshots'][previous_id]['summary']}
        manufactured={}
        for i,rows in [(i,d['rows'])for i,d in full['diagnostics'].items()]+[(previous_id,previous['diagnostics'][previous_id]['rows'])]:
            manufactured[i]=[dict(passes=n,cases=sum(r['passes']==n for r in rows),qualified=sum(r['passes']==n and r['relative_equation_residual']<=1e-6 for r in rows),worst=max(r['relative_equation_residual']for r in rows if r['passes']==n))for n in [1,2,4,8]]
        arms[name]=dict(id=identifier,control=previous_id,training=t,fitted=quality,snapshots=summaries,manufactured=manufactured,
            control_training=previous['training'][0],decision=full['decision'],cpu_execution=full['cpu_execution'],matched=control,
            control_timing_scope='Retained control timings come from a previous CPU phase; no inference-speed claim from that difference.')
        sources[str(path.relative_to(ROOT))]=sha(path)
    if not allow_partial:
        if 'shape-later-decay'not in arms:raise ValueError('Completed C67 required')
        if arms['shape-later-decay']['decision']['capacity_gate_passed']and'joint-later-decay'not in arms:
            raise ValueError('Passing C67 requires the conditional completed C68 test')
    gradients={i:read(directory/'reports'/(i+'-gradients.json'))for i in ['C53','C66']}
    for i in gradients:
        p=directory/'reports'/(i+'-gradients.json');sources[str(p.relative_to(ROOT))]=sha(p)
    decision=dict(status='running'if allow_partial else'study_completed',baseline_retained='C53',model_promoted=False,
        original_untouched_holdouts_consumed=0,next_action='Await restored CPU qualification.'if allow_partial else
            'C67 failed the full fitting gate: skip C68, retain C53, and diagnose loss-term gradient conflict before further CUDA training.'if not arms['shape-later-decay']['decision']['capacity_gate_passed']else
            'Use the completed 24-case result to plan a separate real-frame retention experiment; retain C53.')
    write(directory/'results.json',dict(schema='rheon-shape-decay-study-v1',status='running'if allow_partial else'completed',arms=arms,gradients=gradients,decision=decision,sources=sources))
    write(directory/'decision.json',decision)
    print('Projected',list(arms),decision['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();collect(args.directory.resolve(),args.allow_partial)
