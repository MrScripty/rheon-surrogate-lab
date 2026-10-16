"""Bind every consumed reference before creating a fresh frozen-policy study."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB/'experiments/water-inference-budget-v1'))
from scaling import summarize


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--configs', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    assert protocol['schema'] == 'rheon-pressure-bound-regression-protocol-v1'
    lineage = read(LAB/protocol['lineage_sequence'])
    policy = read(LAB/protocol['policy_study'])
    models = {m['name']: m for m in policy['models']}
    reference, candidate = models['reference-current'], models['late-pressure']
    expected = digest(LAB/reference['artifacts']/'weights.npz')
    norm = digest(LAB/reference['artifacts']/'normalization.json')
    binary = digest(LAB/policy['binary'])
    stopping = read(LAB/policy['profile'])
    populations = [
        {'name': name, 'config': lineage[key], 'reference_study': lineage['reference_development_studies'][name],
         'layout': 'study', 'group': 'original58'}
        for name, key in (('fresh', 'fresh-development'), ('prior', 'prior-grids'), ('long', 'long-study'))]
    populations += [{**p, 'layout': 'study', 'group': 'original58'} for p in lineage['regressions'] if not p.get('exploratory')]
    populations += [{**p, 'group': p['name']} for p in protocol['extra_populations']]
    cases, references, scenes = [], [], []
    for population in populations:
        descriptor = read(LAB/population['config'])
        source = LAB/population['reference_study']
        report = read(source/'report.json')
        assert report['binary_sha256'] == binary, population['name']
        matching = [m for m in report['models'] if m['weights_sha256'] == expected]
        assert len(matching) == 1
        model = matching[0]
        assert model['normalization_sha256'] == norm and model['single_evaluation']
        for index, scene in enumerate(descriptor['scenes']):
            assert report['configuration']['scenes'][index] == scene
            rows = [r for r in report['records'] if r['scene'] == index and r['model'] == model['name']]
            assert len(rows) == 1
            row = rows[0]
            directory = (LAB/row['output']/'native' if population['layout'] == 'episode' else
                         source/f'scene-{index:02d}'/f'{model["name"]}-{row["repetition"]}')
            receipt = read(directory/'receipt.json')
            assert digest(directory/'receipt.json') == row['receipt_sha256']
            assert receipt['scene'] == scene and receipt['binary_sha256'] == binary
            assert receipt['status'] == 'completed' and receipt['native_process']['reaped']
            assert not receipt['result']['hard_stop'] and receipt['result']['ai_steps'] == scene['steps']-scene['warmup_steps']
            assert not receipt['cuda_initialized'] and not receipt['native_pressure_refinement']
            assert receipt['adaptive_stopping'] == stopping and receipt['proposal_network_evaluations'] == 1
            assert digest(Path(receipt['cpu_restore'])/'weights.npz') == expected
            assert digest(Path(receipt['cpu_restore'])/'normalization.json') == norm
            callbacks = [json.loads(s) for s in (directory/'callbacks.jsonl').read_text().splitlines()]
            frames = [json.loads(s) for s in (directory/'frames.jsonl').read_text().splitlines()]
            assert len(callbacks) == receipt['result']['ai_steps'] and len(frames) == scene['steps']
            assert all(c['trace'][-1]['equation_passed'] for c in callbacks)
            metrics = summarize(directory)
            assert metrics['all_ai_steps_qualified'] and metrics['all_reference_steps_qualified']
            number = len(scenes)
            cases.append({'scene': number, 'population': population['name'], 'group': population['group'],
                'source_scene': index, 'source_report': str((source/'report.json').relative_to(LAB)),
                'source_report_sha256': digest(source/'report.json'),
                'source_receipt': str((directory/'receipt.json').relative_to(LAB)),
                'source_receipt_sha256': digest(directory/'receipt.json'),
                'source_frames_sha256': digest(directory/'frames.jsonl'),
                'source_callbacks_sha256': digest(directory/'callbacks.jsonl')})
            scenes.append(scene)
            references.append((directory, {'scene': number, 'model': 'reference-current', 'repetition': 0,
                'qualified': True, 'all_declared_stopping_checks_passed': True,
                'output': protocol['root']+f'/reference-links/scene-{number:02d}', **metrics}))
    assert len(scenes) == 74 and sum(c['group'] == 'original58' for c in cases) == 58
    assert len({json.dumps(s, sort_keys=True) for s in scenes}) == len(scenes)
    root = LAB/protocol['root']
    root.mkdir(parents=True, exist_ok=False)
    for directory, row in references:
        output = LAB/row['output']
        output.mkdir(parents=True)
        (output/'native').symlink_to(directory.absolute(), target_is_directory=True)
        write(output/'diagnostics.json', {'samples': [], 'scope': 'archived baseline has no sparse same-RHS measurements; unavailable, not zero error'})
    config = {**policy, 'root': protocol['root'], 'models': [candidate], 'scenes': scenes,
        'fixture_report': protocol['root']+'/qualification/fixture/report.json',
        'source_populations': [p['config'] for p in populations]}
    write(args.configs/'regression.json', config)
    write(args.configs/'fixture.json', {**config, 'root': protocol['root']+'/qualification',
                                     'models': [reference, candidate]})
    for index in protocol['timing_scene_indices']:
        write(args.configs/f'timing-{index:02d}.json', {**config,
            'root': protocol['root']+f'/timing-{index:02d}', 'models': [reference, candidate],
            'serial_timing_scene': index})
    fresh = []
    for index, counts in enumerate(protocol['fresh_counts']):
        seed = protocol['fresh_seed_start']+index
        rng = random.Random(seed)
        fresh.append({'seed': seed, 'counts': counts, 'cells_per_metre': 8 if index % 2 == 0 else 16,
            'steps': protocol['fresh_steps'], 'warmup_steps': 15,
            'center': [round(rng.uniform(.25, .75), 6) for _ in range(3)],
            'acceleration': round(rng.uniform(.1, .4), 6), 'radius': round(rng.uniform(.18, .35), 6),
            'swirl': round(rng.uniform(-.95, .95), 6), 'fluctuation': round(rng.uniform(.1, .7), 6),
            'frequency': round(rng.uniform(.5, 2.7), 6), 'sgs': False})
    write(args.configs/'fresh-scenes.json', {'scenes': fresh, 'scope': 'reserved independent validation; conditional on all consumed gates'})
    bindings = {m['name']: {'weights_sha256': digest(LAB/m['artifacts']/'weights.npz'),
        'normalization_sha256': digest(LAB/m['artifacts']/'normalization.json'),
        'source_witness_sha256': digest(LAB/m['project']/'source-witness.json'),
        'stopping_profile_sha256': digest(LAB/m.get('stopping_profile', policy['profile']))} for m in (reference, candidate)}
    write(args.reports/'population-bindings.json', {'cases': cases, 'reference_model': reference,
        'reference_rows': [r for _, r in references], 'models': bindings, 'binary_sha256': binary,
        'protocol_sha256': digest(args.protocol), 'regression_config_sha256': digest(args.configs/'regression.json'),
        'source_sha256': digest(__file__), 'cases_count': len(cases),
        'scope': 'immutable numerical reference reuse; archived timings are not current benchmark evidence'})
    print(json.dumps({'verified_references': len(cases), 'fresh_reserved_scenes': len(fresh)}))


if __name__ == '__main__':
    main()
