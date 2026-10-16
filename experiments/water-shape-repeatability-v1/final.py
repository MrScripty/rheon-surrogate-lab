"""Freeze development selection, then run untouched physical comparisons once."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

from execution import LAB, command, read, utc, write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    sequence = read(args.sequence)
    root = LAB/sequence['root']
    destination = args.reports/'final-selection.json'
    if destination.exists():
        raise FileExistsError('selection already frozen; reconcile rather than reselect')
    comparison = read(args.reports/'comparison.json')
    rows = comparison['models']
    def rank(row):
        return row['adaptive_mean_evaluations'], row['fixed_eight_worst_residual']
    eligible = [row for row in rows if row['all_development_qualified']]
    choice = min(eligible, key=rank)
    challengers = [row for row in rows[1:] if row['all_development_qualified']]
    candidate = min(challengers or rows[1:], key=rank)
    model = {'name': candidate['model'],
             'project': str((root/candidate['model']/'project').relative_to(LAB)),
             'artifacts': str((root/candidate['model']/'managed-full/stage-01/artifacts').relative_to(LAB)),
             'single_evaluation': True}
    if digest(LAB/model['artifacts']/'weights.npz') != candidate['weights_sha256']:
        raise ValueError('selected checkpoint identity differs')
    configurations = []
    capture_commit = read(root/'capture.json')['source_commit']
    for index, original in enumerate(sequence['final_configs']):
        source = LAB/original
        committed = subprocess.check_output(['git','show',capture_commit+':'+original], cwd=LAB)
        if source.read_bytes() != committed:
            raise ValueError('final setup changed since pre-training capture')
        output = root/f'final-holdout-{index}'
        runtime = root/f'final-holdout-{index}.study.json'
        if output.exists() or runtime.exists():
            raise FileExistsError('final attempt already exists; no replay or reselection')
        study = read(source)
        study['models'] = [sequence['reference'], model]
        write(runtime, study)
        configurations.append({'original': original, 'original_sha256': digest(source),
                               'runtime': str(runtime.relative_to(LAB)),
                               'runtime_sha256': digest(runtime)})
    selection = {'schema': 'rheon-shape-repeatability-selection-v1', 'frozen_at_utc': utc(),
        'development_choice': choice['model'], 'candidate': model,
        'candidate_development_qualified': candidate['all_development_qualified'],
        'candidate_weights_sha256': candidate['weights_sha256'],
        'reference': sequence['reference'], 'configs': configurations,
        'comparison_sha256': digest(args.reports/'comparison.json'),
        'selection_policy': sequence['selection'],
        'scope': 'best new eligible challenger, or unqualified diagnostic if none; retain reference when it wins development; no final-informed refit/reselection'}
    # The scientific selection is immutable even if a later operational step fails.
    with destination.open('x') as stream:
        import json
        json.dump(selection, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for index, config in enumerate(configurations):
        command([sys.executable, LAB/'experiments/water-multigrid-learning-v1/study.py',
                 '--config', LAB/config['runtime'], '--output', root/f'final-holdout-{index}',
                 '--workers', '8'], root, f'final-holdout-{index}')
    print('Final comparisons complete for frozen challenger', model['name'], flush=True)


if __name__ == '__main__':
    main()
