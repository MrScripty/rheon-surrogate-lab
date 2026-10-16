"""Index actual training, native trajectories, final callbacks and elapsed times."""
import argparse
from datetime import datetime
import os
import json
from pathlib import Path
import shutil
import sys

from run import LAB,digest,read,utc,write


def verify_commands(commands,reconciled):
    exceptions={r['failed_timing']:r for r in reconciled}
    assert len(exceptions)==len(reconciled)
    failed=set()
    for name,c in commands.items():
        assert c['child_reaped']
        if c['status']=='completed':assert c['returncode']==0
        else:
            assert name in exceptions and c['status']=='failed' and c['returncode']!=0
            assert digest(LAB/name)==exceptions[name]['timing_sha256']
            recovery=commands[exceptions[name]['recovery_timing']]
            assert recovery['status']=='completed' and recovery['returncode']==0 and recovery['child_reaped']
            failed.add(name)
    assert failed==set(exceptions)
    return failed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('sequence','reports'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    sequence=read(args.sequence)
    root=LAB/sequence['root']
    status=read(root/'study-status.json')
    if status['state']!='completed':
        raise ValueError('complete the scientific workflow before indexing')
    comparison=read(args.reports/'comparison.json')
    selection=read(args.reports/'final-selection.json')
    decision=read(args.reports/'decision.json')
    if decision['candidate_promoted']:
        replication=read(args.reports/'replication.json')
        assert replication['promotion_confirmed'] and replication['candidate_weights_sha256']==decision['retained_weights_sha256']
    trials=[read(args.reports/('trial-'+t['name']+'.json')) for t in sequence['trials']]
    assert sum(t['cuda_optimizer_updates'] for t in trials)==comparison['new_cuda_updates']
    stages=[s for t in trials for s in t['managed_stages']]
    assert len({s['run_id'] for s in stages})==len(stages)==3*len(trials)
    assert all(s['state']=='completed' and s['custody_phase']=='drained' for s in stages)
    commands={str(p.relative_to(LAB)):read(p) for p in root.rglob('*-timing.json')}
    reconciled=read(args.reports/'reconciliations.json')['accepted_failed_commands']
    failed=verify_commands(commands,reconciled)
    preparation=read(args.reports/'cpu-preparation.json')
    teachers=[preparation['fixture'],*preparation['parents']]
    assert len(teachers)==17 and all(t['status']=='qualified_cpu_teacher_parent'
        and t['native_process']['reaped'] and t['native_process']['returncode']==0
        and t['network_calls']==t['optimizer_updates']==0 and not t['cuda_initialized'] for t in teachers)
    assert len(preparation['late_labels'])==32
    old=LAB/'runs/water-coverage-feedback-v1/prepared-late-residuals'
    assert digest(old/'predictions.npz')==read(old/'receipt.json')['predictions_sha256']==preparation['old_predictions_sha256']
    assert digest(Path(__file__).with_name('prepare_cpu.py'))==preparation['source_sha256']
    for row in preparation['late_labels']:assert digest(LAB/row['path'])==row['sha256']
    for i,teacher in enumerate(preparation['parents']):
        assert teacher['frames']==teacher['scene']['steps']==64 and len(teacher['samples'])==49
        assert digest(LAB/preparation['output_root']/f'parent-{i:02d}/fields.npz')==teacher['fields_sha256']
    collection=read(args.reports/'future-collection.json')
    assert collection['reopened_exact_records']==len(collection['records'])==16
    assert all(t['catalog_id']==collection['catalog_id'] for t in trials)
    native=[read(p) for p in root.rglob('receipt.json') if p.parent.parent.name.startswith('scene-')]
    assert native and all(r['native_process']['reaped'] and r['native_process']['returncode']==0 for r in native)
    populations=[]
    for trial in trials:
        directory=root/trial['trial']
        values=[trial[k]['native'] for k in ('adaptive_fit','adaptive_development')]
        values.append(read(directory/'fixed-eight/report.json')['native'])
        values.extend(trial['raw_native'].values())
        assert len(values)==5 and all(v['all_native_children_drained'] for v in values)
        populations.extend({'trial':trial['trial'],'cases':v['cases']} for v in values)
    final=read(root/'final-holdout/report.json')
    assert datetime.fromisoformat(final['started_at_utc'])>datetime.fromisoformat(selection['frozen_at_utc'])
    assert digest(LAB/selection['final_config'])==selection['final_config_sha256']
    for model in final['models']:
        key=selection['challenger'] if model['name']==selection['challenger'] else 'reference'
        assert model['weights_sha256']==selection['scores'][key]['weights_sha256']
    counts={'reference':0,'candidate':0}
    qualified={'reference':0,'candidate':0}
    for row in final['records']:
        if row['model']=='native':
            continue
        directory=root/'final-holdout'/f"scene-{row['scene']:02d}"/f"{row['model']}-{row['repetition']}"
        receipt=read(directory/'receipt.json')
        frames=[read_line for line in (directory/'frames.jsonl').read_text().splitlines()
                if (read_line:=json.loads(line)).get('ai_quality')]
        callbacks=[json.loads(line) for line in (directory/'callbacks.jsonl').read_text().splitlines()]
        assert [f['step'] for f in frames]==[c['step'] for c in callbacks]
        assert [c['network_passes'] for c in callbacks]==row['network_passes']
        assert all(f['step']>receipt['scene']['warmup_steps'] for f in frames)
        key='candidate' if row['model']==selection['challenger'] else 'reference'
        counts[key]+=len(frames)
        qualified[key]+=sum(f['ai_quality']['qualified'] for f in frames)
    assert all(counts[k]==decision['final'][k]['attempted_ai_steps']
               and qualified[k]==decision['final'][k]['qualified_ai_steps'] for k in counts)
    alive=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():
            continue
        try:
            argv=[a.decode(errors='surrogateescape') for a in (process/'cmdline').read_bytes().split(b'\0') if a]
        except (OSError,PermissionError):
            continue
        if len(argv)>1 and Path(argv[0]).name.startswith('python') and any('/'+Path(sequence['root']).name+'/' in a for a in argv[1:]):
            alive.append(int(process.name))
    assert not alive,alive
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert size<2*1024**3 and shutil.disk_usage(LAB).free>=1024**3
    timings={'recorded_at_utc':utc(),'commands':commands,'managed_stages':comparison['managed_stages'],
        'optimizer_seconds_sum':comparison['new_optimizer_seconds'],
        'managed_sequence_seconds':comparison['sequence_elapsed_seconds'],
        'scientific_controller_seconds':status['elapsed_seconds'],
        'capture_to_final_scientific_evaluation_seconds':(datetime.fromisoformat(status['finished_at_utc'])
            -datetime.fromisoformat(read(root/'capture.json')['started_at_utc'])).total_seconds(),
        'scope':'elapsed workflow and overlapping throughput observations; sums are not makespan or isolated inference latency'}
    write(args.reports/'timings.json',timings)
    summary={'recorded_at_utc':utc(),'method_argv':[sys.executable,*sys.argv],
        'cuda_updates':comparison['new_cuda_updates'],'managed_stages_completed_drained':len(stages),
        'optimizer_intervals_disjoint':comparison['optimizer_intervals_disjoint'],
        'native_trajectory_children_exit_zero_reaped':len(native),
        'native_diagnostic_populations':len(populations),'native_diagnostic_cases':sum(p['cases'] for p in populations),
        'final_frames_and_callbacks':counts,'qualified_final_frames':qualified,'frozen_selection_precedes_final':True,
        'owned_controllers_alive':alive,'retained_bytes':size,'retention_budget_bytes':2*1024**3,
        'reconciled_failed_commands':sorted(failed),'cpu_teacher_children_exit_zero_reaped':len(teachers),
        'future_training_parents':len(preparation['parents']),'late_labels':len(preparation['late_labels']),
        'published_future_records':collection['reopened_exact_records'],
        'decision':decision,'timings_sha256':digest(args.reports/'timings.json'),
        'source_sha256':{str(p.relative_to(LAB)):digest(p) for p in Path(__file__).parent.glob('*.py')},
        'deployment_changed':False}
    write(args.reports/'summary.json',summary)
    print({k:summary[k] for k in ('cuda_updates','managed_stages_completed_drained',
                                'native_trajectory_children_exit_zero_reaped','retained_bytes')})


if __name__=='__main__':
    main()
