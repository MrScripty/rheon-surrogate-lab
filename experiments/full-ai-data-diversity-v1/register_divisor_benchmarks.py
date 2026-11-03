"""Register serial prototype/control measurements after actual worker custody."""
import argparse
import json
from pathlib import Path
import hashlib
import os
import platform
import time

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT/'docs/plans/full-ai-data-diversity-v1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Fresh benchmark protocol directory required')
    plan_path = DOCS/'divisor-prototype-benchmark-plan-06.json'
    plan = read(plan_path)
    # Custody is recorded by the parent only after receiving actual terminal
    # tool results. A stale status file alone cannot populate this assertion.
    custody = read(args.custody)
    expected = set(plan['reference_tool_sessions_to_reap_before_registration'])
    observed = {row['session_id'] for row in custody['sessions']
                if row['actual_tool_observation'] == 'terminal' and row['exit_code'] == 0}
    if not expected <= observed:
        raise ValueError('Every required tool session must be observed terminal')
    for path, allowed in plan['required_terminal_reference_controllers'].items():
        controller = read(ROOT/path)
        if controller['status'] not in allowed or controller.get('active') is not None:
            raise ValueError('Live reference work blocks CPU registration: '+path)
    for field in ('harness',):
        if digest(ROOT/plan[field]) != plan[field+'_sha256']:
            raise ValueError('Frozen prototype measurement harness changed')
    native = plan['matched_native_control']
    if digest(ROOT/native['harness']) != native['harness_sha256']:
        raise ValueError('Qualified native measurement source changed')
    if digest(ROOT/native['qualification']) != native['qualification_sha256']:
        raise ValueError('Native control qualification evidence changed')
    independent = ROOT/'docs/plans/full-ai-data-diversity-v1/learned-divisor-independent-evaluation-protocol-02.json'
    base = read(independent)
    sources = dict(base['source_sha256'])
    fixed = [plan_path, args.custody.resolve(), Path(__file__).resolve(), independent,
             ROOT/native['harness'], ROOT/native['qualification'],
             ROOT/'experiments/full-ai-data-diversity-v1/verify_divisor_measured_accuracy.py',
             ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json',
             ROOT/'runs/full-ai-data-diversity-v1/learned-divisor-independent-evaluation-02/results.json']
    fixed += [ROOT/path for path in plan['required_terminal_reference_controllers']]
    for path in fixed:
        sources[str(path.relative_to(ROOT))] = digest(path)
    for path, expected_hash in sources.items():
        if digest(ROOT/path) != expected_hash:
            raise ValueError('Bound model/data/measurement source changed: '+path)
    for cores in plan['proposed_distinct_physical_cpu_sets']:
        topology = []
        for core in cores:
            directory = Path('/sys/devices/system/cpu')/f'cpu{core}'/'topology'
            topology.append(((directory/'physical_package_id').read_text().strip(),
                             (directory/'core_id').read_text().strip()))
        if len(set(topology)) != len(cores):
            raise ValueError('Distinct physical cores required')
    args.output.mkdir(parents=True)
    cpu_models = sorted({line.split(':', 1)[1].strip() for line in
                         Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')})
    hardware = args.output/'hardware.json'
    hardware.write_text(json.dumps(dict(cpu_models=cpu_models, machine=platform.machine(),
        kernel=platform.release(), orchestrator_allowed_logical_cpus=sorted(os.sched_getaffinity(0)),
        registered_distinct_physical_cpu_sets=plan['proposed_distinct_physical_cpu_sets'],
        numeric_runtime_manifest='runs/dependency-migration-v1/runtime/current-manifest-v2.json'), indent=2)+'\n')
    sources[str(hardware.relative_to(ROOT))] = digest(hardware)
    entries = []

    def register(label, harness, changes, expected_rows):
        reg = {key: base[key] for key in
               ('pair', 'library_path', 'library_sha256', 'normalization', 'maximum_error')}
        reg.update(changes)
        reg.update(status='registered_after_all_reference_owners_terminal_before_CPU_measurement',
            created_unix=time.time(), source_sha256=sources,
            reference_sessions_revalidated_terminal_before_registration=True,
            required_terminal_reference_controllers=plan['required_terminal_reference_controllers'],
            maximum_ms_per_step=plan['maximum_ms_per_generated_step'], fresh_holdout=False,
            independent_confirmation=False, latency_claim=False, full_generator_target_pass=False,
            expected_measurement_rows=expected_rows,
            independent_accuracy_evidence='runs/full-ai-data-diversity-v1/learned-divisor-independent-evaluation-02/results.json',
            measurement_does_not_replace_full_trajectory_accuracy_verification=True,
            role='Complete prototype characterization; preserve every result even when target fails')
        if label.startswith('native-control'):
            reg.update(role='Matched native Rheon PCG complete-state reference control',
                       candidate_inference=False, pressure_control='native-pcg')
        path = args.output/(label+'.json')
        path.write_text(json.dumps(reg, indent=2)+'\n')
        entries.append(dict(label=label, protocol=str(path.relative_to(ROOT)),
            protocol_sha256=digest(path), harness=harness, expected_rows=expected_rows,
            output='runs/full-ai-data-diversity-v1/divisor-'+label+'-01'))

    scenes = len(read(ROOT/plan['bank'])['trajectories'])
    cases = scenes*len(plan['starts'])*len(plan['horizons'])
    for mode in ('profile', 'benchmark'):
        repeats = plan['profile_repeats'] if mode == 'profile' else plan['benchmark_repeats']
        for cores in plan['proposed_distinct_physical_cpu_sets']:
            register('prototype-'+mode+'-p'+str(len(cores)), plan['harness'],
                dict(mode=mode, cores=cores, bank=plan['bank'], starts=plan['starts'],
                     horizons=plan['horizons'], repeats=repeats), cases*repeats)
    register('native-control-benchmark-p1', native['harness'],
        dict(mode='benchmark', cores=native['cores'], bank=native['bank'], starts=native['starts'],
             horizons=native['horizons'], repeats=native['repeats']), cases*(native['repeats']+1))
    (args.output/'index.json').write_text(json.dumps(dict(status='registered_serial_measurement_batch',
        entries=entries, reference_custody=str(args.custody.relative_to(ROOT)),
        reference_custody_sha256=digest(args.custody),
        all_measurements_must_execute_serially=True, model_targets_verified=False), indent=2)+'\n')
    print('Registered serial prototype1/2/4core profiles and benchmarks plus matched native control', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--custody', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.custody = args.custody.resolve()
    args.output = args.output.resolve()
    run(args)
