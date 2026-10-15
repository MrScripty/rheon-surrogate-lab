"""Retain one immutable scientific record per completed managed experiment.

Small records belong in Git; weights and numeric inputs remain Tuldok artifacts.
Re-recording identical evidence is harmless; differing evidence needs a new ID.
"""
import argparse
import hashlib
import json
from pathlib import Path

LAB=Path(__file__).resolve().parents[2]
REPORTS=LAB/'docs/plans/water-residual-learning-v1/reports'


def load(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',required=True)
    args=parser.parse_args()
    assert args.trial and all(c.isalnum() or c=='-' for c in args.trial)
    root=LAB/'runs/water-residual-learning-v1'/args.trial;project=root/'project'
    config=load(project/'config.json');source=load(project/'source-witness.json')
    stages=[];train=None;history=None
    for path in sorted(root.glob('managed-*/receipt.json')):
        receipt=load(path);assert receipt['status']=='completed_and_drained'
        for index,stage in enumerate(receipt['stages']):
            assert stage['state']=='completed' and stage['custody_phase']=='drained'
            stages.append({'catalog_id':receipt['catalog_id'],'project_id':receipt['project_id'],**stage})
            if stage['name']=='train':
                artifacts=path.parent/f'stage-{index:02d}'/'artifacts'
                train=load(artifacts/'receipt.json');history=load(artifacts/'history.json')
    assert train is not None and any(s['name']=='restore' for s in stages)
    assert len({s['run_id'] for s in stages})==len(stages)
    assert train['closure']['source_witness_sha256']==hashlib.sha256((project/'source-witness.json').read_bytes()).hexdigest()
    evaluations={name:load(root/name/'evaluation.json') for name in ('fitted-evaluation','validation-evaluation') if (root/name/'evaluation.json').exists()}
    audits={name:load(root/name/'summary.json') for name in ('fitted-audit','validation-audit') if (root/name/'summary.json').exists()}
    record={'schema':'rheon-residual-experiment-record-v1','trial':args.trial,
        'method':{'configuration':config,'source_commit':source['source_commit'],'architecture':source['architecture'],
            'source_witness_sha256':train['closure']['source_witness_sha256'],'source_hashes':source['sources'],
            'inputs':source['inputs'],'normalization_sha256':source['normalization_sha256'],
            'sample_order_sha256':train['closure']['sample_order_sha256'],
            'selection':'terminal update fixed in advance; validation is development, not untouched test',
            'native_checks':'unchanged float32 profile plus independently checked native equation/physical fields',
            'scope':'teacher-state fitting/development; no autonomous rollout or deployment claim'},
        'managed_stages':stages,'cuda_optimizer_updates':train['optimizer_updates'],'execution':train['execution'],
        'runtime':train['runtime'],'weights_sha256':train['outputs']['weights.npz']['sha256'],
        'checkpoints':{name:{key:spec[key] for key in ('update','sha256','bytes')} for name,spec in history['checkpoints'].items()},
        'training_trace':history['trace'],'validation_loss':history['validation'],
        'optimizer_updates_per_s':history['optimizer_updates_per_s'],'native_evaluations':evaluations,'cpu_audits':audits,
        'retained_artifacts':str(root.relative_to(LAB)),'new_reserved_test_used':False}
    REPORTS.mkdir(exist_ok=True,parents=True);path=REPORTS/(args.trial+'.json')
    raw=json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==raw,'immutable record differs; retain a separately named revision'
    else:path.write_text(raw)
    print(json.dumps({'trial':args.trial,'record':str(path.relative_to(LAB)),'stages':len(stages),'updates':train['optimizer_updates'],
        'native_evaluations':evaluations,'cpu_audits':audits}),flush=True)


if __name__=='__main__':main()
