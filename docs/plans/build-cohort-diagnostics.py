"""Add comparable chart diagnostics for trained models without legacy trial files.

CPU inference on the existing 128 validation snapshots and the exact retained
mixed-mode probe inputs. No training, simulator steps, or acceptance decisions.
Run with ../../.venv/bin/python build-cohort-diagnostics.py MODEL_NAME.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--study', default='water-early-pressure-v1')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--reuse-probes', action='store_true', help='Retain the matching spatial probe while remeasuring validation.')
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--only-validation', action='store_true', help='Measure the serial timing population only.')
    phase.add_argument('--only-probes', action='store_true', help='Reuse existing validation evidence; run spatial probes only.')
    args = parser.parse_args()
    name, study = args.name, args.study
    admissions = [f'{study}/reports/training-and-admission.json',
                  f'{study}/reports/training/admission.json',
                  f'chart-diagnostics/{study}/{name}/admission.json']
    admission_source = next(source for source in admissions if (ROOT / source).exists())
    admission = json.loads((ROOT / admission_source).read_text())
    entry = next(m for m in admission['models'] if m['name'] == name)
    project, artifacts = LAB / entry['project'], LAB / entry['artifacts']
    assert digest(artifacts / 'weights.npz') == entry['weights_sha256']
    source_lab = LAB / entry.get('source_lab', '.')
    sys.path.insert(0, str(source_lab / 'experiments/water-residual-learning-v1'))
    import audit
    cpu = args.cpu if args.cpu is not None else sorted(os.sched_getaffinity(0))[-1]
    audit.initialize(project.resolve(), artifacts.resolve(), [cpu])
    torch = audit.TORCH
    target = ROOT / 'chart-diagnostics' / study / name
    if args.only_probes:
        saved = json.loads((target / 'validation.json').read_text())
        assert saved['weights_sha256'] == entry['weights_sha256']
        assert saved['admission_sha256'] == digest(ROOT / admission_source)
    else:
        inputs = LAB / entry['validation_inputs'] if entry.get('validation_inputs') else project / 'inputs'
        manifest = json.loads((inputs / 'validation_splits.json').read_text())
        # Tiny-smoother Projects expose a train-only conditioned reader. The
        # comparison must decode the untouched historical validation population.
        reader = audit.DATA.historical_data if entry.get('proposal_network_evaluations') == 1 and not entry.get('captured_pressure_probe') else audit.DATA
        _, parents = reader.load(((inputs / 'validation_dataset.npz').read_bytes(),
                                     (inputs / 'validation_dataset_extra.npz').read_bytes()), manifest)
        records, identities, times = [], [], []
        for index, parent in enumerate(parents):
            for slot, frame in enumerate(parent['frames']):
                sample = audit.DATA.sample(parent, slot)
                with torch.no_grad():
                    start = time.perf_counter()
                    pressure = audit.MODEL.predict_pressure(audit.NETWORK, sample, audit.NORM)
                    elapsed = time.perf_counter() - start
                    residual = audit.ratio(pressure, sample)
                identity = dict(trajectory_id=parent['entry']['trajectory_id'], native_frame=frame['index'],
                                frame=slot, parent=index, split='validation', counts_xyz=parent['counts_xyz'],
                                cell_size_m=parent['cell_size_m'])
                identities.append(identity)
                times.append(elapsed)
                records.append(dict(**identity, raw_learned={'quality': {'relative_residual': residual}},
                                    inference_s=elapsed))
        assert len(records) == 128 and all(r['raw_learned']['quality']['relative_residual'] >= 0 for r in records)
        target = ROOT / 'chart-diagnostics' / study / name
        target.mkdir(parents=True, exist_ok=True)
        cases_path = target / 'cases.jsonl'
        cases_path.write_text(''.join(json.dumps(r, separators=(',', ':'), allow_nan=False) + '\n' for r in records))
        errors = [r['raw_learned']['quality']['relative_residual'] for r in records]
        family = entry.get('family') or study.removeprefix('water-').removesuffix('-v1').replace('-', ' ').capitalize()
        validation = dict(schema='rheon-chart-validation-v1', name=name, family=family,
            weights_sha256=entry['weights_sha256'], cases=128,
            relative_residual=dict(minimum=min(errors), median=statistics.median(errors), maximum=max(errors)),
            per_case_inference_s=times, timing_cases=identities,
            run_root='../../' + str((LAB / entry['project']).parent.relative_to(LAB)),
            case_source=str(cases_path.relative_to(ROOT)), cases_sha256=digest(cases_path),
            admission_source=admission_source, admission_sha256=digest(ROOT / admission_source),
            project_source='../../' + entry['project'] + '/source-witness.json', project_sha256=digest(project / 'source-witness.json'),
            inputs={str(p.relative_to(inputs)): digest(p) for p in [inputs / 'validation_splits.json', inputs / 'validation_dataset.npz', inputs / 'validation_dataset_extra.npz']},
            builder_sha256=digest(Path(__file__)), optimizer_updates=0, simulation_steps=0,
            replay_project=os.path.relpath(project, ROOT), replay_artifacts=os.path.relpath(artifacts, ROOT),
            replay_source_lab=os.path.relpath(source_lab, ROOT),
            runtime=dict(threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_initialized=torch.cuda.is_initialized()),
            scope='New chart diagnostic on the same consumed 128 snapshots as the original-proposal comparison. '
                  'One timed predict_pressure call per case, excludes setup/sample assembly/error verification. '
                  'Equation error only; native physical qualification not measured. No independent CFD validation or promotion decision.')
        (target / 'validation.json').write_text(json.dumps(validation, separators=(',', ':'), allow_nan=False) + '\n')
        print(name, '128-state diagnostic', 'worst', max(errors), 'median ms', statistics.median(times)*1000, flush=True)

    if args.only_validation:
        assert not torch.cuda.is_initialized()
        return

    if args.reuse_probes:
        previous = json.loads((target / 'mode-probe.json').read_text())
        assert previous['weights_sha256'] == entry['weights_sha256'] and len(previous['cases']) == 120
        return

    # Reuse the exact 120 input specifications, preserving the benchmark identity.
    page = (ROOT / 'cohort-progress.html').read_text()
    data = json.loads(re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', page, re.S)[1])
    reference = data['frequency_probes'][0]
    sys.path.insert(0, str(source_lab / 'experiments/water-multigrid-learning-v1'))
    if entry.get('captured_pressure_probe'):
        sys.path.insert(0, str(source_lab / 'experiments/water-stability-learning-v1'))
        import captured_pressure_probe as mode_probe
    elif entry.get('proposal_network_evaluations') == 1:
        sys.path.insert(0, str(source_lab / 'experiments/water-stability-learning-v1'))
        import local_smoother_probe as mode_probe
    else:
        import mode_probe
    tasks = [{k: r[k] for k in ['counts_xyz', 'cell_size_m', 'population', 'components_xyz', 'amplitude_pa']} for r in reference['cases']]
    results = [mode_probe.probe(task) for task in tasks]
    aggregate = []
    for counts, population in dict.fromkeys((tuple(t['counts_xyz']), t['population']) for t in tasks):
        group = [r for r in results if tuple(r['counts_xyz']) == counts and r['population'] == population]
        aggregate.append(dict(counts_xyz=list(counts), population=population, cases=len(group),
            equation_passed=sum(r['equation_passed'] for r in group),
            mean_network_evaluations=statistics.mean(r['network_evaluations'] for r in group),
            worst_first_relative_residual=max(r['first_relative_residual'] for r in group),
            worst_final_relative_residual=max(r['final_relative_residual'] for r in group)))
    probe = dict(schema='rheon-multigrid-mixed-mode-report-v1', seed=reference['seed'], mixtures=60,
                 signed_cases=120, cases=results, aggregate=aggregate, weights_sha256=entry['weights_sha256'],
                 source_sha256=digest(Path(mode_probe.__file__)), optimizer_updates=0, simulations_advanced=0,
                 scope=('Same 120 signed inputs; at most 32 complete captured pressure proposals. Each proposal includes all declared fine CNN and learned global stages; not the older 32 single-CNN-evaluation protocol.' if entry.get('captured_pressure_probe') else 'Chart replay of the exact existing mixed-mode benchmark specifications using its retained probe implementation.'))
    (target / 'mode-probe.json').write_text(json.dumps(probe, separators=(',', ':'), allow_nan=False) + '\n')
    assert not torch.cuda.is_initialized()
    print(name, '120 mixed-mode cases completed', flush=True)


if __name__ == '__main__':
    main()
