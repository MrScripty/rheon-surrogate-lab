"""Attempt the frozen rejected-scene cohort with a uniform finer reference clock."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from support import ROOT, read, write, digest


def run(args):
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('Fresh reference refinement controller output required')
    reg = read(args.protocol)
    for path, expected in reg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Registered reference recovery source changed: '+path)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    child = None
    record = dict(status='running', completed=[], reference_rejections=[], active=None,
        protocol_sha256=digest(args.protocol), reference_substeps=reg['reference_substeps'],
        total_scenes_required=len(reg['rows']), optimizer_updates=0,
        model_targets_verified=False, reserved_references_consumed=False,
        controller_pid=os.getpid())

    def save():
        record['updated_unix'] = time.time()
        write(args.output/'controller.json', record)

    def stop(*_):
        raise InterruptedError('Stop reference controller and reap its active capture')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        for row in reg['rows']:
            if row['split'] != 'train':
                raise ValueError('Only frozen training-reference failures may be recovered here')
            used = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
            if used > reg['maximum_total_capture_bytes'] or shutil.disk_usage(args.output).free < reg['minimum_free_disk_bytes']:
                raise ValueError('Reference refinement disk budget reached')
            remaining = reg['maximum_controller_wall_seconds'] - (time.monotonic()-started)
            if remaining < reg['maximum_wall_seconds']+30:
                raise TimeoutError('Insufficient bounded time for the next reference scene')
            identifier = row['identifier']
            directory = args.output/identifier
            command = [sys.executable, '-B', str(Path(__file__).with_name('capture_refined_data.py')),
                '--protocol', str(args.protocol.resolve()), '--identifier', identifier,
                '--output', str(directory)]
            child = subprocess.Popen(command, cwd=ROOT)
            record['active'] = dict(identifier=identifier, pid=child.pid,
                                    output=str(directory.relative_to(ROOT)), command=command)
            save()
            try:
                code = child.wait(timeout=reg['maximum_wall_seconds']+30)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=10)
                raise
            child = None
            bank = read(directory/'bank.json')
            if code == 0 and bank['status'] == 'completed' and len(bank['trajectories']) == 1:
                record['completed'].append(bank['trajectories'][0])
            else:
                rejection = directory/identifier/'reference-rejection.json'
                custody = directory/identifier/'failed-native-custody.json'
                if not rejection.exists() or not custody.exists():
                    raise RuntimeError('Unexpected refinement failure; preserve evidence for diagnosis')
                owner = read(custody)
                if not owner['reaped'] or owner['returncode'] != 0 or not owner['graceful_protocol_complete']:
                    raise RuntimeError('Failed reference child must be reaped gracefully')
                record['reference_rejections'].append(dict(identifier=identifier,
                    evidence=str(rejection.relative_to(ROOT)), native_process=owner,
                    unresolved=True))
            record['active'] = None
            save()
        record['status'] = 'completed_refinement_attempts'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            record['capture_child_returncode'] = child.returncode
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
