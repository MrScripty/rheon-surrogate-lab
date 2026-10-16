"""Freeze matched completed managed artifacts before CPU regression."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import read,write,utc


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True);args=p.parse_args()
    config=read(args.config);root=LAB/config['root'];root.mkdir(exist_ok=False)
    models=[];trials={};intervals=[]
    for model in config['models']:
        artifacts=LAB/model['artifacts'];managed=artifacts.parent.parent
        assert read(managed/'receipt.json')['status']=='completed_and_drained'
        source=LAB/model['project'];witness=read(source/'source-witness.json')
        train=managed/'stage-00/artifacts'
        trained,restored=read(train/'receipt.json'),read(artifacts/'receipt.json')
        assert trained['mode']=='train' and trained['optimizer_updates']==5000
        assert restored['mode']=='restore' and not restored['execution']['cuda_initialized']
        assert trained['closure']==restored['closure']
        assert restored['closure']['source_witness_sha256']==digest(source/'source-witness.json')
        assert trained['execution']['device']=='cuda:0' and trained['execution']['model_dtype']=='float32'
        assert trained['execution']['parameters_gradients_adam_moments_cuda']
        assert not trained['execution']['amp'] and not trained['execution']['tf32']
        for name in ('weights.npz','normalization.json','history.json'):
            assert digest(train/name)==digest(artifacts/name)==restored['outputs'][name]['sha256']
        history=read(artifacts/'history.json');start,end=history['training_started_unix'],history['training_finished_unix']
        intervals.append((start,end))
        trials[model['name']]={'weights_sha256':digest(artifacts/'weights.npz'),
            'normalization_sha256':digest(artifacts/'normalization.json'),
            'initial_weights_sha256':history['checkpoints']['checkpoint-000.npz']['sha256'],
            'sample_order_sha256':trained['closure']['sample_order_sha256'],
            'source_commit':witness['source_commit'],'architecture':trained['architecture'],
            'training_seconds':end-start,'updates_per_second':history['optimizer_updates_per_s'],
            'managed_elapsed_seconds':read(source.parent.parent/('managed-'+model['name']+'-timing.json'))['elapsed_seconds'],
            'train_recipe_elapsed_seconds':trained['elapsed_s'],
            'restore_recipe_elapsed_seconds':restored['elapsed_s'],
            'weights_file_bytes':(artifacts/'weights.npz').stat().st_size,
            'cuda_peak_allocated_bytes':trained['execution']['peak_allocated_bytes'],
            'cuda_cpu_pressure_max_abs_difference':trained['execution']['cuda_cpu_pressure_max_abs_difference'],
            'history':{k:history[k] for k in ('analytical_feedback','supplemental_feedback','direction_feedback','history','validation')},
            'managed_stages':[{k:s[k] for k in ('name','run_id','state','custody_phase')} for s in read(managed/'receipt.json')['stages']],
            'configuration':read(source/'config.json')}
        models.append({**model,'weights_sha256':digest(artifacts/'weights.npz'),
                       'normalization_sha256':digest(artifacts/'normalization.json')})
    a,b=[trials[m['name']] for m in config['models']]
    for key in ('initial_weights_sha256','sample_order_sha256','normalization_sha256'):
        assert a[key]==b[key],key
    for key in ('analytical_feedback','supplemental_feedback','direction_feedback'):
        assert a['history'][key]==b['history'][key],key
    assert intervals[0][1]<=intervals[1][0]
    ca,cb=[dict(t['configuration']) for t in (a,b)]
    for c in (ca,cb):c.pop('id');c.pop('first_pressure_weight')
    assert ca==cb
    baseline='docs/plans/water-pressure-bound-regression-v1/reports/regression-summary.json'
    sources=[baseline,'docs/plans/water-pressure-bound-regression-v1/reports/population-bindings.json',
        'runs/water-pressure-bound-regression-v1/development/report.json',
        'experiments/water-stability-learning-v1/learned_iteration.py',
        'experiments/water-residual-learning-v1/rollout.py','experiments/water-stopping-floor-v1/episode.py',
        'experiments/water-stopping-floor-v1/run.py','experiments/water-stopping-floor-v1/configs/pressure-bound.json']
    for m in config['models']:
        sources.extend([m['project']+'/source-witness.json',m['artifacts']+'/weights.npz',m['artifacts']+'/normalization.json'])
    record={'frozen_at_utc':utc(),'config_sha256':digest(args.config),'models':models,'trials':trials,
        'baseline_summary':baseline,'binary_sha256':digest(LAB/config['binary']),
        'sources':[{'path':s,'sha256':digest(LAB/s)} for s in sources],
        'matched_initialization_sampling_and_cells':True,'optimizer_intervals_disjoint':True,
        'independent_scenes_used':0,'source_sha256':digest(__file__)}
    write(root/'admission.json',record);write(args.reports/'training-and-admission.json',record)
    print(json.dumps({'trials':{n:{k:t[k] for k in ('weights_sha256','training_seconds','updates_per_second','cuda_peak_allocated_bytes')} for n,t in trials.items()},'matched':True},indent=2))


if __name__=='__main__':main()
