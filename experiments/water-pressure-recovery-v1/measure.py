"""Bind frozen artifacts and reuse the existing full-population comparison."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'experiments/water-shape-repeatability-v1'))
from execution import read, write, utc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit(config_path, reports):
    config = read(config_path)
    root = LAB/config['root']
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'admission.json').exists()
    models = []
    sources = ['experiments/water-stability-learning-v1/learned_iteration.py',
               'experiments/water-residual-learning-v1/rollout.py',
               'experiments/water-stopping-floor-v1/episode.py',
               'experiments/water-stopping-floor-v1/run.py']
    trials = {}
    baseline_root = LAB/'runs/water-early-pressure-v1/control'
    baseline_history = read(baseline_root/'managed-full/stage-01/artifacts/history.json')
    baseline_receipt = read(baseline_root/'managed-full/stage-01/artifacts/receipt.json')
    baseline_config = read(baseline_root/'project/config.json')
    for model in config['models']:
        project, artifacts = LAB/model['project'],LAB/model['artifacts']
        managed = artifacts.parent.parent
        assert read(managed/'receipt.json')['status']=='completed_and_drained'
        trained = read(managed/'stage-00/artifacts/receipt.json')
        restored = read(artifacts/'receipt.json')
        assert trained['optimizer_updates']==5000 and trained['execution']['device']=='cuda:0'
        assert restored['mode']=='restore' and not restored['execution']['cuda_initialized']
        assert trained['closure']==restored['closure']
        assert restored['closure']['source_witness_sha256']==digest(project/'source-witness.json')
        for name in ('weights.npz','normalization.json','history.json'):
            assert digest(artifacts/name)==restored['outputs'][name]['sha256']
            assert digest(artifacts/name)==digest(managed/'stage-00/artifacts'/name)
        history = read(artifacts/'history.json')
        assert history['checkpoints']['checkpoint-000.npz']['sha256']==baseline_history['checkpoints']['checkpoint-000.npz']['sha256']
        assert restored['closure']['sample_order_sha256']==baseline_receipt['closure']['sample_order_sha256']
        assert digest(artifacts/'normalization.json')==digest(baseline_root/'managed-full/stage-01/artifacts/normalization.json')
        for key in ('analytical_feedback','supplemental_feedback','direction_feedback'):
            assert history[key]==baseline_history[key]
        selected=read(project/'config.json')
        a,b=dict(selected),dict(baseline_config)
        for c in (a,b):
            c.pop('id');c.pop('first_pressure_weight')
        assert a==b
        for name, sha in read(project/'source-witness.json')['sources'].items():
            assert digest(project/name)==sha
            if name not in ('config.json','qualification.json'):
                assert sha==read(baseline_root/'project/source-witness.json')['sources'][name]
        models.append({**model,'weights_sha256':digest(artifacts/'weights.npz'),
                       'normalization_sha256':digest(artifacts/'normalization.json')})
        sources.extend([model['project']+'/source-witness.json',model['artifacts']+'/weights.npz',
                        model['artifacts']+'/normalization.json',model['stopping_profile']])
        trials[model['name']]={'configuration':selected,'weights_sha256':digest(artifacts/'weights.npz'),
            'source_commit':read(project/'source-witness.json')['source_commit'],
            'optimizer_seconds':history['training_finished_unix']-history['training_started_unix'],
            'peak_cuda_allocated_bytes':trained['execution']['peak_allocated_bytes'],
            'matched_control_initialization_sampling_and_feedback':True}
    record={'frozen_at_utc':utc(),'config_sha256':digest(config_path),'models':models,'trials':trials,
            'binary_sha256':digest(LAB/config['binary']),
            'baseline_summary':'docs/plans/water-pressure-bound-regression-v1/reports/regression-summary.json',
            'sources':[{'path':p,'sha256':digest(LAB/p)} for p in sorted(set(sources))]}
    write(root/'admission.json',record)
    write(reports/'admission.json',record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('admit','analyze'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--reports',type=Path,required=True)
    args=parser.parse_args()
    args.reports.mkdir(parents=True,exist_ok=True)
    if args.mode=='admit':
        admit(args.config,args.reports)
    else:
        spec=importlib.util.spec_from_file_location('consumed_comparison',LAB/'experiments/water-early-pressure-v1/compare.py')
        owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
        sys.argv=[str(spec.origin),'--config',str(args.config),'--reports',str(args.reports)]
        owner.main()


if __name__=='__main__':
    main()
