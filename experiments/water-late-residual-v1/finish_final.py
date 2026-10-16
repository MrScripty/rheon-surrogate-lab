"""Evaluate a declared setup amendment with the already frozen model choice.

No fitting, reselection, physics relaxation or original-attempt overwrite. The
post-fit amendment is reported explicitly and cannot promote a checkpoint under
the original pre-fit final protocol.
"""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parents[2]
spec = spec_from_file_location('direction_final_owner', LAB/'experiments/water-coverage-feedback-v1/run.py')
owner = module_from_spec(spec)
spec.loader.exec_module(owner)
read, write, digest, command, utc = owner.read, owner.write, owner.digest, owner.command, owner.utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--amendment', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--torch-python', required=True)
    args = parser.parse_args()
    sequence, amendment = read(args.sequence), read(args.amendment)
    root = LAB/sequence['root']
    status_path = root/'final-validation-status.json'
    assert not status_path.exists() and not (args.reports/'decision.json').exists()
    selection_path = args.reports/'final-selection.json'
    selection = read(selection_path)
    assert digest(selection_path) == amendment['selection_sha256']
    assert read(root/'study-recovery-status.json')['state'] == 'failed'
    for item in amendment['original_controls']:
        assert digest(LAB/item['path']) == item['sha256']
        receipt = read(LAB/item['path'])
        assert receipt['kind'] == 'native-algorithm-control' and receipt['native_process']['reaped']
    assert not list((root/'final-holdout').glob('scene-*/*/callbacks.jsonl'))
    original = LAB/amendment['original_config']
    amended = LAB/amendment['amended_config']
    assert digest(original) == amendment['original_sha256'] == selection['final_config_sha256']
    assert digest(amended) == amendment['amended_sha256']
    assert digest(LAB/amendment['native_validator_source']) == amendment['native_validator_sha256']
    assert all(read(LAB/item['path'])['binary_sha256'] == digest(LAB/read(amended)['binary'])
        for item in amendment['original_controls'])
    assert subprocess.check_output(['git', 'show', 'HEAD:'+amendment['amended_config']], cwd=LAB) == amended.read_bytes()
    before, after = read(original), read(amended)
    expected = json.loads(json.dumps(before))
    for change in amendment['changes']:
        index = change['scene_index']
        assert abs(expected['scenes'][index]['swirl']) > 1 and change['field'] == 'swirl'
        expected['scenes'][index]['swirl'] = max(-1., min(1., expected['scenes'][index]['swirl']))
    expected['models'] = [sequence['reference'], selection['candidate']]
    assert expected == after
    for label, model in [('reference', sequence['reference']), (selection['challenger'], selection['candidate'])]:
        assert digest(LAB/model['artifacts']/'weights.npz') == selection['scores'][label]['weights_sha256']
    status = {'started_at_utc': utc(), 'state': 'running', 'amendment_sha256': digest(args.amendment),
        'selection_sha256': digest(selection_path), 'source_sha256': digest(__file__),
        'original_protocol_amended_after_fitting': True, 'training_repeated': False, 'selection_changed': False}
    started = time.monotonic()
    write(status_path, status)
    observer = owner.Observer(LAB/'runs/live-project')
    observer.publish('Amended final validation', 'Frozen late-direction candidate; two native setup repairs, no fitting or reselection')
    try:
        # Qualify one actual post-warm-up algorithm step for both repaired setups.
        for change in amendment['changes']:
            index = change['scene_index']
            scene = {**after['scenes'][index], 'steps': after['scenes'][index]['warmup_steps']+1}
            path = root/f'final-amended-fixture-{index:02d}.json'
            write(path, scene)
            output = root/f'final-amended-fixture-{index:02d}'
            command([args.torch_python, LAB/'experiments/water-residual-learning-v1/rollout.py',
                '--binary', LAB/after['binary'], '--scene', path, '--output', output,
                '--reference-control'], root, f'final-amended-fixture-{index:02d}')
            receipt = read(output/'receipt.json')
            assert receipt['status'] == 'completed' and receipt['native_process']['reaped']
            assert receipt['frames'] == scene['steps'] and not receipt['result']['hard_stop']
        output = root/'final-holdout-amended'
        command([args.torch_python, LAB/'experiments/water-multigrid-learning-v1/study.py',
            '--config', amended, '--output', output, '--workers', '8'], root, 'final-holdout-amended')
        path = args.reports/'final-amended-drift.json'
        command([args.torch_python, LAB/'experiments/water-multigrid-learning-v1/trajectory_summary.py',
            '--studies', output, '--output', path], root, 'final-amended-drift')
        weights = {'reference': selection['scores']['reference']['weights_sha256'],
            'candidate': selection['scores'][selection['challenger']]['weights_sha256']}
        final = owner.retained.population(path, [output], weights)
        ref, candidate = final['reference'], final['candidate']
        assert ref['scenes'] == candidate['scenes'] == 8
        gate = candidate['qualified_scenes'] == 8 and all(candidate[k] <= ref[k]
            for k in ('maximum_network_evaluations', 'mean_network_evaluations', 'worst_pressure_relative_l2'))
        write(args.reports/'decision.json', {'recorded_at_utc': utc(), 'candidate_promoted': False,
            'retained_weights_sha256': weights['reference'], 'challenger': selection['challenger'],
            'development': selection['populations'], 'final': final,
            'amended_final_numeric_gates_passed': gate, 'selection_sha256': digest(selection_path),
            'amendment_sha256': digest(args.amendment), 'original_final_protocol_satisfied': False,
            'scope': 'amended post-fit validation; frozen candidate and eight original seeds, only two invalid swirl values clipped; original pre-fit final protocol not satisfied; no promotion or deployment change'})
        status['state'] = 'completed'
    except BaseException as error:
        status.update(state='failed', error_type=type(error).__name__)
        raise
    finally:
        status.update(finished_at_utc=utc(), elapsed_seconds=time.monotonic()-started)
        write(status_path, status)
        observer.publish('Amended final '+status['state'],
            'Methods, original failures and amended validation retained; no model promotion',
            state='completed' if status['state']=='completed' else 'failed')


if __name__ == '__main__':
    main()
