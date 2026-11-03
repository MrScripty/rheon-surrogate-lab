"""Own a bounded serial CUDA correction-work matrix without automatic retries."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pressure_gpu_timing import ROOT, read, save, sha


def run(protocol_path):
    protocol = read(protocol_path)
    output = ROOT / protocol['output_root']
    if output.exists():
        raise ValueError('Observe the original matrix; never replay into an existing output')
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Preregistered source changed: ' + name)
    runtime = read(ROOT / protocol['runtime_manifest'])
    sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
    from common import environment
    output.mkdir(parents=True)
    state = dict(status='starting', pid=os.getpid(), created_unix=time.time(),
        protocol_sha256=sha(protocol_path), completed=[], active=None,
        pending=list(protocol['models']), latency_claim=False,
        inference_device='cuda', CPU_model_forward=False, full_portfolio_complete=False)
    begun = time.monotonic()
    try:
        for identifier, specification in protocol['models'].items():
            remaining = protocol['aggregate_deadline_seconds'] - (time.monotonic() - begun)
            if remaining <= 0:
                raise TimeoutError('Finite matrix deadline reached')
            destination = output / identifier
            command = [runtime['python'], '-B', str(ROOT / protocol['worker']), '--protocol', str(protocol_path),
                       '--identifier', identifier, '--output', str(destination)]
            with (output / (identifier + '.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment(runtime, 'cuda'),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='original_GPU_policy_worker_running',
                    active=dict(identifier=identifier, pid=process.pid, output=str(destination.relative_to(ROOT))))
                save(output / 'status.json', state)
                try:
                    returncode = process.wait(timeout=min(remaining, protocol['worker_deadline_seconds'] + 60))
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
            path = destination / 'results.json'
            result = read(path) if path.exists() else None
            if (returncode or result is None or result['status'] != 'complete_GPU_pressure_correction_work_matrix'
                    or result['device_calls']['forbidden'] or result['cases'] != 128
                    or result['emitted_pressure_fields'] != 128 * len(protocol['policies'])
                    or (specification['require_reference_parity'] and result['original_default_output_parity_cases'] != 128)):
                raise ValueError('Original policy worker failed; retain evidence and reconcile before another attempt')
            state['completed'].append(dict(identifier=identifier, result=str(path.relative_to(ROOT)),
                sha256=sha(path), summary=result['summary'],
                original_default_output_parity_cases=result['original_default_output_parity_cases']))
            state['pending'].remove(identifier)
            state.update(status='original_GPU_policy_worker_complete', active=None)
            save(output / 'status.json', state)
            print(identifier, result['status'], flush=True)
        for name, expected in protocol['source_sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Preregistered source changed during matrix execution: ' + name)
        state.update(status='complete_original_GPU_pressure_correction_work_matrix', active=None,
            finished_unix=time.time(), source_bindings_unchanged=True, independent_rescoring_pending=True)
    except BaseException as error:
        state.update(status='original_matrix_requires_reconciliation', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(output / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    run(parser.parse_args().protocol.resolve())
