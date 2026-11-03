"""Run one preregistered pressure accuracy batch serially on the real GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def run(protocol):
    plan = json.loads(protocol.read_text())
    for path, digest in plan['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Preregistered pressure source/input changed: ' + path)
    runtime = json.loads((ROOT / 'runs/dependency-migration-v1/runtime/current-manifest-v2.json').read_text())
    sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output = ROOT / plan['output_root']
    if output.exists():
        raise ValueError('Fresh batch output required; do not repeat an uncertain execution')
    output.mkdir(parents=True)
    value = dict(schema='rheon.pressure-material-GPU-accuracy-batch.v1', status='running',
                 protocol=dict(path=str(protocol.relative_to(ROOT)), sha256=sha(protocol)),
                 source_sha256=sha(Path(__file__)), models=plan['models'],
                 completed=[], active=None, pending=list(plan['models']),
                 inference_device='cuda', CPU_neural_forward=False,
                 latency_claim=False, independent_confirmation=False, new_parameter_fits=0,
                 complete_portfolio_benchmark=False)
    started = time.monotonic()
    try:
        for identifier in plan['models']:
            remaining = plan['aggregate_deadline_seconds'] - (time.monotonic() - started)
            if remaining < 1:
                raise TimeoutError('Aggregate pressure accuracy deadline reached')
            value['active'] = identifier
            write(output / 'batch.json', value)
            destination = output / identifier
            command = [runtime['python'], '-B',
                       'experiments/model-portfolio-migration-v1/pressure_gpu_accuracy_v2.py',
                       '--identifier', identifier, '--output', str(destination.relative_to(ROOT))]
            with (output / (identifier + '.log')).open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env=environment(runtime, 'cuda'), start_new_session=True)
                try:
                    returncode = process.wait(timeout=min(remaining, plan['worker_deadline_seconds']))
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
            path = destination / 'results.json'
            result = json.loads(path.read_text())
            if (result['status'] != 'completed_captured_pressure_GPU_accuracy'
                    or result['cases'] != plan['case_count_per_model']
                    or result['neural_module_forward_calls']['cpu'] != 0
                    or not result['learned_parameters_unchanged'] or not result['architecture_unchanged']):
                raise ValueError('Pressure execution qualification incomplete: ' + identifier)
            value['completed'].append(dict(identifier=identifier, result=str(path.relative_to(ROOT)),
                                            sha256=sha(path), cases=result['cases'], worst=result['worst'],
                                            neural_module_forward_calls=result['neural_module_forward_calls']))
            value['pending'].remove(identifier)
            print(identifier, result['cases'], result['worst'], flush=True)
        for path, digest in plan['source_sha256'].items():
            if sha(ROOT / path) != digest:
                raise ValueError('Pressure source/input changed during execution: ' + path)
        value.update(status='completed_original21_pressure_GPU_accuracy', active=None,
                     source_bindings_unchanged=True, diagnostic_elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        value.update(status='failed_or_incomplete_batch', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(output / 'batch.json', value)
    print(value['status'], len(value['completed']), 'models', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve())
