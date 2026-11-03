"""Finite serial development-240 controller; never retry/restart a child."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def read(p): return json.loads(p.read_text())


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def main(args):
    reg = read(args.registration)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h: raise ValueError('Frozen serial evaluation dependency changed: '+p)
    journal = args.output/'controller-progress.json'
    if journal.exists(): raise ValueError('Fresh controller journal required; no automatic submission retry')
    first = read(ROOT/reg['first_result'])
    if (first['status'] != 'completed_streamed_expanded_gpu_240step_scene' or not first['all_accuracy_pass']
            or len(first['rows']) != 960 or not first['native_process']['reaped']
            or first['native_process']['returncode'] != 0 or not first['native_process']['graceful_protocol_complete']):
        raise ValueError('Actual successful firstN8native240 and drained native custody required')
    protocol = read(ROOT/reg['evaluation_protocol'])
    remaining = [c['identifier'] for c in protocol['cases'] if c['identifier'] != first['identifier']]
    if len(remaining) != 23 or remaining != reg['remaining_case_order']: raise ValueError('Frozen remaining23case population required')
    child = None
    def stop(*_): raise InterruptedError('Stop serial evaluation and reap owned group')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    progress = dict(status='running', active=None, completed=[], expected_remaining_cases=23,
                    first_result=reg['first_result'], registration_sha256=digest(args.registration),
                    automatic_retries=0, cpu_model_forward=False, optimizer_updates=0,
                    expanded_reserved52_references_opened=0, latency_claim=False)
    def save(): journal.write_text(json.dumps(progress, indent=2)+'\n')
    save()
    try:
        for identifier in remaining:
            output = args.output/identifier
            if output.exists(): raise ValueError('Refuse to overwrite/restart a prior invocation: '+identifier)
            command = [sys.executable, '-B', 'experiments/full-ai-next-priorities-v1/device_runtime.py',
                       '--device', 'cuda', '--timeout', '1900', reg['evaluation_script'],
                       '--protocol', reg['evaluation_protocol'], '--identifier', identifier, '--output', str(output)]
            progress['active'] = dict(identifier=identifier, output=str(output), launched_unix=time.time()); save()
            child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
            progress['active']['pid'] = child.pid; save()
            returncode = child.wait()
            path = output/'results.json'
            result = read(path) if path.exists() else {}
            passed = bool(returncode == 0 and result.get('status') == 'completed_streamed_expanded_gpu_240step_scene'
                          and result.get('all_accuracy_pass') and len(result.get('rows', [])) == 960)
            progress['completed'].append(dict(identifier=identifier, returncode=returncode, result=str(path),
                                              result_sha256=digest(path) if path.exists() else None,
                                              all_accuracy_pass=passed, worst_error=result.get('worst_error'),
                                              native_process=result.get('native_process')))
            progress['active'] = None; save()
            if not passed: raise RuntimeError('Preserved failed native240case; stop before any further case: '+identifier)
        progress.update(status='completed_all_remaining23_absolute_pressure_native240', elapsed_diagnostic_seconds=time.monotonic()-started)
        save()
    except BaseException as e:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        progress.update(status='stopped_with_preserved_failure', error=repr(e), elapsed_diagnostic_seconds=time.monotonic()-started)
        save(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registration', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
