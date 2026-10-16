"""Seal completed matched CUDA/CPU evidence without activating a checkpoint."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

LAB=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--reports',type=Path,required=True)
    args=p.parse_args();config=read(args.config);root=LAB/'runs/water-early-pressure-v1';reports=args.reports
    regression=read(reports/'regression-summary.json');timing=read(reports/'serial-timing.json')
    admission=read(reports/'training-and-admission.json');paired=read(root/'pair-execution.json')
    assert paired['state']=='completed' and timing['state']=='completed'
    cpu=read(root/'cpu-sequence-execution.json');assert cpu['state']=='consumed_measurements_completed'
    for item in admission['sources']:assert digest(LAB/item['path'])==item['sha256']
    assert digest(LAB/config['binary'])==admission['binary_sha256']
    eligible=regression['all_consumed_gates_passed']
    if eligible:
        fresh=read(reports/'fresh-validation-summary.json')
        fresh_state='completed';last=read(LAB/config['root']/'validation/report.json')['finished_at_utc']
    else:
        assert not (LAB/config['root']/'validation').exists()
        assert not (LAB/config['root']/'fresh-preflight').exists()
        fresh=None;fresh_state='deferred: consumed replacement gates failed';last=timing['finished_at_utc']
    controllers=[];native=[];peaks=[]
    for path in sorted(root.rglob('*-timing.json')):
        value=read(path);assert value['child_reaped']
        controllers.append({'path':str(path.relative_to(LAB)),'sha256':digest(path),
            'status':value['status'],'elapsed_seconds':value['elapsed_seconds'],'child_reaped':True})
    for path in sorted((LAB/config['root']).rglob('receipt.json')):
        value=read(path)
        if 'native_process' in value:
            assert value['native_process']['reaped'] and not value['cuda_initialized']
            assert value['binary_sha256']==admission['binary_sha256']
            native.append({'path':str(path.relative_to(LAB)),'sha256':digest(path),
                'status':value['status'],'native_process':value['native_process']})
    for path in (LAB/config['root']).rglob('diagnostics.json'):
        value=read(path)
        if 'peak_rss_bytes' in value:peaks.append(value['peak_rss_bytes'])
    managed=[]
    for trial,folder in [('smoke','managed-smoke'),('control','managed-full'),('first-pressure','managed-full')]:
        value=read(root/trial/folder/'receipt.json');assert value['status']=='completed_and_drained'
        for stage in value['stages']:
            assert stage['state']=='completed' and stage['custody_phase']=='drained'
            managed.append({'trial':trial,**{k:stage[k] for k in ('name','run_id','state','custody_phase')},
                'catalog_id':value['catalog_id'],'project_id':value['project_id']})
    assert len({s['run_id'] for s in managed})==6 and len({s['catalog_id'] for s in managed})==1
    first=read(root/'preparation-qualified-source-timing.json')['started_at_utc']
    elapsed=(datetime.fromisoformat(last)-datetime.fromisoformat(first)).total_seconds()
    timings={name:read(root/(name+'-timing.json'))['elapsed_seconds'] for name in
        ('preparation-qualified-source','managed-smoke','managed-control','managed-first-pressure',
         'regression-fixture','regression-development','current-serial-timing')}
    result={'schema':'rheon-earlier-pressure-completed-experiment-v1','state':'completed',
        'consumed_gates_passed':eligible,'regression':regression['groups'],
        'eligible_models':regression['eligible_for_fresh_validation'],
        'frozen_challenger':regression['frozen_challenger'],'fresh_validation_state':fresh_state,
        'fresh_validation':fresh,'fresh_scenes_consumed':8 if eligible else 0,
        'optimizer_updates':10001,'full_training_runs':2,'smoke_training_runs':1,
        'candidate_promoted':False,'deployment_changed':False,
        'training':admission['trials'],'phase_wall_seconds':timings,
        'serial_timing':timing['summaries'],'serial_timing_all_qualified':timing['all_qualified'],
        'first_timed_preparation_to_last_measurement_seconds':elapsed,
        'elapsed_scope':'Qualified preparation command start through last scientific measurement, including source/analysis/managed setup gaps; excludes initial untimed failed homogeneity preparation and subsequent documentation.',
        'managed_runs':managed,'all_managed_runs_completed_and_drained':True,
        'native_receipts':native,'all_recorded_native_children_reaped':True,
        'controller_receipts':controllers,'all_recorded_controller_children_reaped':True,
        'maximum_individual_completed_worker_rss_bytes':max(peaks,default=None),
        'memory_scope':'Individual process high-water mark, not simultaneous system memory.',
        'raw_retained_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'numerical_runtime_sources_and_frozen_artifacts_unchanged':True,
        'source_sha256':digest(__file__),
        'sources':{name:digest(reports/name) for name in ('regression-summary.json','serial-timing.json',
            'training-and-admission.json','amplitude-summary.json','fixed-call-summary.json','input-rounding.json')},
        'limitations':['Consumed and manufactured diagnostics are not independent CFD validation.',
            'Temporal frequency is distinct from spatial mode index;32-frame windows cannot resolve cycles below1.875Hz.',
            'Weak modes can be obscured by model leakage and input rounding in mixtures; quantization oracle is not an iterative precision lower bound.',
            'Same-RHS pressure bounds do not guarantee a universal autonomous trajectory error envelope.',
            'Serial timing covers three declared workloads with desktop apps running; archived or concurrent timings are not current isolated latency evidence.',
            'Initial failed homogeneity preparation lacks a durable full elapsed/controller receipt; retained separately, with terminal process scan required.']}
    write(reports/'summary.json',result)
    write(reports/'regression-episodes.json',read(LAB/config['root']/'development/report.json'))
    (reports/'current.json').write_text(json.dumps({'state':'completed','consumed_gates_passed':eligible,
        'fresh_validation':fresh_state,'candidate_promoted':False,'summary':'summary.json'},indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('state','consumed_gates_passed','fresh_validation_state',
        'first_timed_preparation_to_last_measurement_seconds')},indent=2))


if __name__=='__main__':main()
