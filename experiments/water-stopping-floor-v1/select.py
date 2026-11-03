"""Freeze a consumed-data policy choice before any fresh learned trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

LAB = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-config', 'pressure-config', 'floor-summary', 'pressure-summary',
                 'fresh-scenes', 'fresh-fixtures', 'output-config'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    floor = read(args.floor_summary)
    pressure = read(args.pressure_summary)
    assert floor['consumed_acceptance_passed']
    assert floor['phase'] == pressure['phase'] == 'development'
    baseline = floor['models']['late-minimum8']
    bounded = pressure['models']['late-pressure']
    keys = ('worst_paired_pressure_relative_l2', 'mean_network_evaluations', 'maximum_network_evaluations')
    dominates = pressure['consumed_acceptance_passed'] and all(bounded[k] <= baseline[k] for k in keys)
    choice = 'late-pressure' if dominates else 'late-minimum8'
    config = read(args.base_config)
    pressure_config = read(args.pressure_config)
    fresh = read(args.fresh_scenes)
    fixture = read(args.fresh_fixtures)
    assert fixture['all_qualified'] and len(fixture['records']) == len(fresh['scenes']) == 8
    for actual, expected in zip(fixture['configuration']['scenes'], fresh['scenes']):
        assert actual == {**expected, 'steps': expected['warmup_steps']+1}
    models = {m['name']: m for m in config['models']+pressure_config['models']}
    config.update(scenes=fresh['scenes'], models=[models['reference-current'], models[choice]],
                  fresh_native_fixtures_report=str(args.fresh_fixtures.relative_to(LAB) if args.fresh_fixtures.is_absolute() else args.fresh_fixtures))
    root = LAB/config['root']
    assert not (root/'validation').exists() and not (root/'selection.json').exists()
    with args.output_config.open('x') as stream:
        stream.write(json.dumps(config, indent=2)+'\n')
    bindings = {}
    for model in config['models']:
        profile = model.get('stopping_profile', config['profile'])
        bindings[model['name']] = {'weights_sha256': digest(LAB/model['artifacts']/'weights.npz'),
            'normalization_sha256': digest(LAB/model['artifacts']/'normalization.json'),
            'stopping_profile_sha256': digest(LAB/profile),
            'minimum_network_evaluations': model['minimum_network_evaluations']}
    record = {'schema': 'rheon-stopping-floor-frozen-selection-v1',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'challenger': choice,
        'consumed_acceptance_passed': True, 'pressure_bound_dominates_floor': dominates,
        'selection_rule': fresh['selection_rule'], 'validation_config_sha256': digest(args.output_config),
        'fresh_scenes_sha256': digest(args.fresh_scenes),
        'fresh_fixture_report_sha256': digest(args.fresh_fixtures),
        'consumed_floor_summary_sha256': digest(args.floor_summary),
        'consumed_pressure_summary_sha256': digest(args.pressure_summary),
        'binary_sha256': digest(LAB/config['binary']), 'models': bindings,
        'source_sha256': digest(__file__), 'optimizer_updates': 0, 'candidate_promoted': False,
        'scope': 'frozen consumed-data choice; no final-informed reselection, fit or deployment change'}
    with (root/'selection.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
