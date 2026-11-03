"""Register and own balanced timings of all 127 available named pressure models.

This is pressure-request latency, not simulation-step generation. Original
material, source, weights, native inputs and independently scored output hashes
remain fixed. No scientific arrays, model weights or project fixtures are copied.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pressure_gpu_timing import ROOT, read, save, sha, resources

BASE = ROOT / 'runs/model-portfolio-migration-v1'
MATERIAL = 'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json'
SCORE = 'docs/plans/model-portfolio-migration-v1/reports/pressure-common-score-01.json'
UI = 'runs/model-portfolio-migration-v1/native-thirtyeight-review-browser-01/report.json'


def verify_material(material):
    bindings = {}
    for entry in material['models']:
        for name, expected in {**entry['identities'], **entry['source_bindings']}.items():
            if name in bindings and bindings[name] != expected:
                raise ValueError('Two original material bindings disagree')
            if sha(ROOT / name) != expected:
                raise ValueError('Original material changed: ' + name)
            bindings[name] = expected
    return len(bindings)


def prepare(path, output):
    if path.exists() or output.exists() or not path.is_relative_to(BASE) or not output.is_relative_to(BASE):
        raise ValueError('Fresh task-owned protocol and campaign required')
    score = read(ROOT / SCORE)
    models = sorted(score['models'], key=lambda name: int(name[1:]))
    if models != [f'C{i:02}' for i in range(3, 131) if i != 68]:
        raise ValueError('The exact 127 available named pressure checkpoints are required')
    material = read(ROOT / MATERIAL)
    if {entry['id'] for entry in material['models']} != set(models):
        raise ValueError('Scientific comparison and captured material membership differ')
    bindings = verify_material(material)
    original = read(BASE / 'pressure-GPU-timing-pilot-01/protocol.json')
    sources = dict(original['source_sha256'])
    for name, expected in sources.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Preserved pilot dependency changed: ' + name)
    worker = 'experiments/model-portfolio-migration-v1/pressure_portfolio_gpu_timing.py'
    for name in (MATERIAL, SCORE, worker, 'experiments/model-portfolio-migration-v1/run_pressure_portfolio_gpu_timing.py',
                 'experiments/model-portfolio-migration-v1/legacy_pressure_cuda.py'):
        sources[name] = sha(ROOT / name)
    accuracy = {name: score['models'][name]['result'] for name in models}
    for selected in accuracy.values():
        if sha(ROOT / selected['path']) != selected['sha256']:
            raise ValueError('Original independently scored accuracy capture changed')
    save(path, dict(schema='rheon.complete-pressure-portfolio-GPU-timing-protocol.v1',
        created_unix=time.time(), models=models, material=MATERIAL, score=SCORE,
        source_sha256=sources, material_bindings=bindings, accuracy=accuracy,
        block_order=[dict(identifier=name, phase=phase) for phase, names in
                     ((0, models), (1, list(reversed(models)))) for name in names],
        worker=worker, runtime_manifest=original['runtime_manifest'], output_root=str(output.relative_to(ROOT)),
        required_terminal_browser=UI, host_cpu=4, warmups_per_case=5, samples_per_case=10,
        cases_per_model=128, pooled_samples_per_case=20, expected_timed_calls=325120,
        worker_deadline_seconds=600, aggregate_deadline_seconds=14400,
        frozen_cases=original['frozen_cases'], timing_scope=original['timing_scope'],
        excluded_identifiers=['C01', 'C02', 'C68'],
        exclusion='C01/C02 have no available checkpoint; C68 has no available named checkpoint identity. These are dispositions, not passing results.',
        numerical_policy='Captured deployment policy unchanged. C03/C04 retain their qualified CPU features/reconstruction with source-AST-matched CUDA CNN; all other models use captured CUDA eager execution. Every neural forward is CUDA.',
        comparison='Two separate processes/model, ascending then descending model order. Pool20warm samples/exact native request. No simulation-step target, K division or mixed-generator Pareto claim.',
        maximum_Torch_GPU_allocated_bytes=4*1024**3, maximum_worker_RSS_bytes=8*1024**3,
        CPU_model_forwards=0, new_fits=0, new_reference_generation=False,
        all_model_portfolio_complete=False, independent_reduction_required=True))
    print('Registered127checkpoints/254process blocks/325120timed pressure requests', flush=True)


def run(path):
    protocol = read(path)
    output = ROOT / protocol['output_root']
    if output.exists() or not output.is_relative_to(BASE):
        raise ValueError('Follow the original campaign, never replay it')
    ui = read(ROOT / protocol['required_terminal_browser'])
    if ui['status'] != 'passed_native_project_UI_actions' or ui['errors'] or not ui['browser_stopped']:
        raise ValueError('The original post-rollout UI campaign must finish first')
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Registered source changed: ' + name)
    if resources()['other_compute_pids']:
        raise ValueError('Other GPU compute is active')
    material = read(ROOT / protocol['material'])
    if verify_material(material) != protocol['material_bindings']:
        raise ValueError('Captured material membership changed')
    runtime = read(ROOT / protocol['runtime_manifest'])
    sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output.mkdir()
    state = dict(status='starting', pid=os.getpid(), created_unix=time.time(), completed=[], active=None,
                 protocol_sha256=sha(path), CPU_neural_forwards=0, all_portfolio_complete=False,
                 model_or_dataset_copies=0, independent_reduction_pending=True)
    began = time.monotonic()
    try:
        for index, block in enumerate(protocol['block_order']):
            remaining = protocol['aggregate_deadline_seconds'] - (time.monotonic() - began)
            if remaining <= 0:
                raise TimeoutError('Original registered campaign deadline reached')
            name = f"{index+1:03}-{block['identifier']}-phase{block['phase']}"
            destination = output / name
            with (output / (name + '.log')).open('x') as log:
                child = subprocess.Popen([runtime['python'], '-B', str(ROOT / protocol['worker']),
                    '--protocol', str(path), '--identifier', block['identifier'], '--output', str(destination)],
                    cwd=ROOT, env=environment(runtime, 'cuda'), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                state.update(status='original_pressure_GPU_worker_in_flight',
                             active=dict(block=index+1, pid=child.pid, output=str(destination.relative_to(ROOT)), **block))
                save(output / 'status.json', state)
                try:
                    returned = child.wait(timeout=min(remaining, protocol['worker_deadline_seconds'] + 60))
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=10)
            result_path = destination / 'results.json'
            result = read(result_path) if result_path.exists() else None
            if (returned or result is None or result['status'] != 'complete_pressure_GPU_timing_pilot'
                    or result['cases'] != 128 or result['device_qualification']['forbidden_module_calls']
                    or not result['learned_parameters_unchanged'] or not result['architecture_unchanged']
                    or not result['all_output_hashes_exact']):
                raise ValueError('Original worker incomplete; retain failure and reconcile without retry')
            state['completed'].append(dict(block=index+1, **block,
                result=str(result_path.relative_to(ROOT)), sha256=sha(result_path), cases=128))
            state.update(status='pressure_GPU_block_complete', active=None)
            save(output / 'status.json', state)
            print(name, 'complete128requests', flush=True)
        verify_material(material)
        for name, expected in protocol['source_sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Source changed during original campaign')
        state.update(status='all127_named_pressure_models_GPU_timing_complete', active=None, finished_unix=time.time())
    except BaseException as error:
        state.update(status='original_pressure_GPU_campaign_requires_reconciliation',
                     error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(output / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.output is None:
            parser.error('prepare requires --output')
        prepare(args.protocol.resolve(), args.output.resolve())
    else:
        run(args.protocol.resolve())
