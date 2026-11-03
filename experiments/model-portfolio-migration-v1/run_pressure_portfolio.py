"""Qualify the remaining named pressure checkpoints without hiding adapter gaps.

Each checkpoint has one isolated GPU process and its original captured source.
No CPU neural fallback, training, architecture change or timing claim is allowed.
An integration failure is a pending adapter disposition, not an accuracy result.
"""
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


def verify_material(material):
    for entry in material['models']:
        for path, digest in {**entry['identities'], **entry['source_bindings']}.items():
            if sha(ROOT / path) != digest:
                raise ValueError('Captured portfolio material changed: ' + path)


def run(protocol):
    plan = json.loads(protocol.read_text())
    for path, digest in plan['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Preregistered source changed: ' + path)
    material = json.loads((ROOT / plan['material']).read_text())
    verify_material(material)
    runtime = json.loads((ROOT / 'runs/dependency-migration-v1/runtime/current-manifest-v2.json').read_text())
    sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output = ROOT / plan['output_root']
    if output.exists():
        raise ValueError('Fresh portfolio execution required; reconcile existing work instead of replaying it')
    output.mkdir(parents=True)
    value = dict(schema='rheon.named-pressure-GPU-qualification.v1', status='running',
                 protocol=dict(path=str(protocol.relative_to(ROOT)), sha256=sha(protocol)),
                 source_sha256=sha(Path(__file__)), pending=list(plan['models']), active=None,
                 qualified=[], adapter_gaps=[], latency_claim=False, new_parameter_fits=0,
                 independent_confirmation=False, complete_portfolio_benchmark=False)
    started = time.monotonic()
    try:
        for identifier in plan['models']:
            remaining = plan['aggregate_deadline_seconds'] - (time.monotonic() - started)
            if remaining < 1:
                raise TimeoutError('Aggregate portfolio deadline reached')
            value['active'] = identifier
            write(output / 'batch.json', value)
            destination = output / identifier
            command = [runtime['python'], '-B',
                       'experiments/model-portfolio-migration-v1/pressure_gpu_accuracy_v3.py',
                       '--material', plan['material'], '--identifier', identifier,
                       '--output', str(destination.relative_to(ROOT))]
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
                # A timeout stops the batch rather than overlapping an unknown
                # previous execution. Ordinary terminal adapter failures remain
                # visible and do not suppress unrelated checkpoints.
            path = destination / 'results.json'
            result = json.loads(path.read_text()) if path.exists() else None
            if returncode == 0 and result and result.get('status') == 'completed_captured_pressure_GPU_accuracy':
                if (result['cases'] != 128 or result['neural_module_forward_calls']['cpu']
                        or not result['learned_parameters_unchanged'] or not result['architecture_unchanged']):
                    raise ValueError('Invalid GPU qualification result: ' + identifier)
                value['qualified'].append(dict(identifier=identifier, result=str(path.relative_to(ROOT)),
                                                sha256=sha(path), cases=result['cases'], worst=result['worst']))
                print(identifier, 'qualified', result['worst'], flush=True)
            else:
                log_path = output / (identifier + '.log')
                value['adapter_gaps'].append(dict(identifier=identifier, status='adapter_qualification_gap',
                                                   process_exit=returncode, log=str(log_path.relative_to(ROOT)),
                                                   log_sha256=sha(log_path),
                                                   result=str(path.relative_to(ROOT)) if result else None,
                                                   scientific_accuracy_failure_claim=False))
                print(identifier, 'terminal adapter gap; retained log', flush=True)
            value['pending'].remove(identifier)
        verify_material(material)
        value.update(status='completed_qualification_attempts_with_explicit_gaps', active=None,
                     source_bindings_unchanged=True, diagnostic_elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        value.update(status='incomplete_batch', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(output / 'batch.json', value)
    print(value['status'], len(value['qualified']), 'qualified;', len(value['adapter_gaps']), 'adapter gaps', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve())
