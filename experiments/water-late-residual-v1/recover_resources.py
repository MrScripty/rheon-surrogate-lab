"""Reconcile only resource-deadline final episodes; keep all original attempts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import time

LAB = Path(__file__).resolve().parents[2]
spec = spec_from_file_location('resource_final_owner', LAB/'experiments/water-coverage-feedback-v1/run.py')
owner = module_from_spec(spec)
spec.loader.exec_module(owner)
read, write, digest, command, utc = owner.read, owner.write, owner.digest, owner.command, owner.utc
sys.path.insert(0, str(LAB/'experiments/water-inference-budget-v1'))
from scaling import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--torch-python', required=True)
    args = parser.parse_args()
    sequence = read(args.sequence)
    root = LAB/sequence['root']
    original = root/'final-holdout-amended'
    report = read(original/'report.json')
    selected = read(args.reports/'final-selection.json')
    assert not (args.reports/'decision.json').exists()
    assert read(root/'final-validation-status.json')['state'] == 'failed'
    failures = []
    for row in report['records']:
        if row['returncode'] == 0:
            continue
        directory = original/f'scene-{row["scene"]:02d}'/f'{row["model"]}-{row["repetition"]}'
        receipt = read(directory/'receipt.json')
        assert receipt['status'] == 'failed' and receipt['error'].startswith('TimeoutError:')
        assert receipt['native_process']['reaped'] and receipt['native_process']['terminated']
        failures.append((row, directory))
    assert failures and len(report['records']) == 24
    status_path = root/'final-resource-recovery-status.json'
    assert not status_path.exists()
    retries = root/'final-resource-retries'
    retries.mkdir(exist_ok=False)
    status = {'started_at_utc': utc(), 'state': 'running', 'workers': 2,
        'original_report_sha256': digest(original/'report.json'),
        'selection_sha256': digest(args.reports/'final-selection.json'), 'source_sha256': digest(__file__),
        'deadline_or_physics_changed': False, 'model_or_scene_changed': False}
    started = time.monotonic()
    write(status_path, status)
    owner.Observer(LAB/'runs/live-project').publish('Final resource recovery',
        'Only timed-out episodes, two at a time; original 120-second deadline and CPU affinity retained')

    def retry(item):
        row, previous = item
        label = f'scene-{row["scene"]:02d}-{row["model"]}-{row["repetition"]}'
        output = retries/label
        argv = list(row['command'])
        index = argv.index('--output')+1
        argv[index] = str(output)
        timing = command(argv, retries, label)
        receipt = read(output/'receipt.json')
        old_receipt = read(previous/'receipt.json')
        assert receipt['scene'] == old_receipt['scene'] and receipt['command'] == old_receipt['command']
        assert receipt['binary_sha256'] == old_receipt['binary_sha256']
        assert receipt['cpu_restore'] == old_receipt['cpu_restore']
        before = [json.loads(line) for line in (previous/'frames.jsonl').read_text().splitlines()]
        after = [json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
        for a, b in zip(before, after):
            assert {k: v for k, v in a.items() if k not in ('ai_seconds', 'reference_seconds')} == \
                   {k: v for k, v in b.items() if k not in ('ai_seconds', 'reference_seconds')}
        return (row['scene'], row['model'], row['repetition']), output, timing, len(before)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            completed = list(pool.map(retry, failures))
        recovered = {key: (output, timing, frames) for key, output, timing, frames in completed}
        reconciled = root/'final-holdout-reconciled'
        reconciled.mkdir(exist_ok=False)
        final_report = json.loads(json.dumps(report))
        for row in final_report['records']:
            key = row['scene'], row['model'], row['repetition']
            source = original/f'scene-{row["scene"]:02d}'/f'{row["model"]}-{row["repetition"]}'
            if key in recovered:
                source, timing, frames = recovered[key]
                row.update(returncode=0, controller_child={'pid': timing['pid'], 'reaped': timing['child_reaped']})
                row.update(summarize(source))
                row['resource_recovery'] = {'original_receipt_sha256': digest(original/f'scene-{row["scene"]:02d}'/
                    f'{row["model"]}-{row["repetition"]}'/'receipt.json'), 'matching_original_physical_frame_prefix': frames,
                    'retry_timing': timing, 'output': str(source)}
            destination = reconciled/f'scene-{row["scene"]:02d}'
            destination.mkdir(exist_ok=True)
            (destination/f'{row["model"]}-{row["repetition"]}').symlink_to(source.absolute(), target_is_directory=True)
        final_report['reconciliation'] = {'original_report_sha256': status['original_report_sha256'],
            'resource_retries': len(completed), 'workers': 2, 'partial_failed_attempts_retained': True,
            'scope': 'same physical cases/checkpoints; successful first attempts retained, only resource failures retried; not isolated timing'}
        write(reconciled/'report.json', final_report)
        drift = args.reports/'final-amended-reconciled-drift.json'
        command([args.torch_python, LAB/'experiments/water-multigrid-learning-v1/trajectory_summary.py',
            '--studies', reconciled, '--output', drift], root, 'final-amended-reconciled-drift')
        weights = {'reference': selected['scores']['reference']['weights_sha256'],
            'candidate': selected['scores'][selected['challenger']]['weights_sha256']}
        final = owner.retained.population(drift, [reconciled], weights)
        ref, candidate = final['reference'], final['candidate']
        gate = candidate['qualified_scenes'] == candidate['scenes'] == 8 and all(candidate[k] <= ref[k]
            for k in ('maximum_network_evaluations', 'mean_network_evaluations', 'worst_pressure_relative_l2'))
        write(args.reports/'decision.json', {'recorded_at_utc': utc(), 'candidate_promoted': False,
            'retained_weights_sha256': weights['reference'], 'challenger': selected['challenger'],
            'development': selected['populations'], 'final': final, 'amended_final_numeric_gates_passed': gate,
            'original_final_protocol_satisfied': False, 'selection_sha256': status['selection_sha256'],
            'resource_retries': len(completed), 'scope': 'post-fit setup amendment and transparent resource recovery; original protocol unsatisfied; no promotion, refit, reselection or deployment change'})
        status.update(state='completed', retries=[{'scene': key[0], 'model': key[1], 'output': str(output),
            'matching_prefix_frames': frames, 'timing': timing} for key, output, timing, frames in completed])
    except BaseException as error:
        status.update(state='failed', error_type=type(error).__name__)
        raise
    finally:
        status.update(finished_at_utc=utc(), elapsed_seconds=time.monotonic()-started)
        write(status_path, status)
        owner.Observer(LAB/'runs/live-project').publish('Final recovery '+status['state'],
            'Original attempts and reconciled outcomes retained; no promotion',
            state='completed' if status['state']=='completed' else 'failed')


if __name__ == '__main__':
    main()
