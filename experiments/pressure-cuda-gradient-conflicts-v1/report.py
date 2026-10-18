"""Retain scalar diagnostics and render case/pass conflicts from measured vectors."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,sha,write


def summarize(measurement,execution,directory):
    if (directory/'results.json').exists():
        raise ValueError('Fresh diagnostic summary required')
    result=read(measurement)
    controller=read(execution)
    assert result['status']=='completed' and controller['status']=='completed'
    assert controller['all_owned_children_reaped'] and result['optimizer_updates']==0
    summary=[]
    for snapshot in result['snapshots']:
        chosen=[r for r in snapshot['rows']if r['case']in result['config']['layer_cases']and r['term']in ('1','2')]
        values=[r['vs_weak_final']['cosine']for r in chosen]
        assert len(chosen)==6
        summary.append(dict(update=snapshot['update'],opposed_short_gradient_pairs=sum(v<0 for v in values),
                            compared_pairs=len(values),minimum_cosine=min(values),maximum_cosine=max(values),
                            raw_cuda_gradient_exact=snapshot['cuda_replay']['raw_exact'],
                            clipped_cuda_gradient_exact=snapshot['cuda_replay']['clipped_exact']))
    opposed=sum(r['opposed_short_gradient_pairs']for r in summary)
    compared=sum(r['compared_pairs']for r in summary)
    result.update(execution=controller,summary=summary,
        decision=dict(model_promoted=False,baseline_retained='C53',training_started=False,
            conclusion=f'The weak four-pass equation gradient opposes {opposed}/{compared} selected case 0/1/4 one/two-pass gradients. This is local derivative evidence, not a qualified new training policy.'),
        scope_limits=['Three consumed pre-update training states; eight analytical fitting cases, not fluid rollouts.',
            'Exact CUDA historical replay qualifies measured gradients at these states only.',
            'Gradient cosine describes local first-order descent conflict; finite Adam steps also depend on state, magnitude and nonlinear response.',
            'Separate unscaled per-case gradients have a disclosed tiny aggregation gap; they are not substituted for the exact original backward accumulation.',
            'No new CUDA training trajectory, CPU inference qualification or model promotion.'])
    result['summary_sources']={str(p.relative_to(ROOT)):sha(p)for p in [measurement,execution,HERE/'report.py',HERE/'viewer.html']}
    write(directory/'results.json',result)
    template=(HERE/'viewer.html').read_text()
    (directory/'index.html').write_text(template.replace('STUDY_DATA',json.dumps(result,allow_nan=False).replace('</','<\\/')))
    print('Conflicting selected pairs:',[(r['update'],r['opposed_short_gradient_pairs'],r['compared_pairs'])for r in summary],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('measurement','execution','directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    summarize(args.measurement.resolve(),args.execution.resolve(),args.directory.resolve())
