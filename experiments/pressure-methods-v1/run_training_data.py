"""Own serial broader native captures; reconcile retained fields before resuming."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, write, sha, environment


def run(build, group):
    runs = ROOT/'runs/pressure-methods-v1'
    reports = ROOT/'docs/plans/pressure-methods-v1'
    config = read(runs/'config.json')
    runtime = read(ROOT/config['runtime'])
    receipt_path = reports/('native-training-'+group+'-execution.json')
    build_receipt = read(build/'build-receipt.json')
    if build_receipt['status'] != 'qualified_build':
        raise ValueError('Qualified native build required')
    reservation = read(build/'reservation.json')
    rows = [r for r in reservation['scenes'] if r['campaign_group'] == group]
    expected = dict(teacher=48, spacing=64, **{'domain-development':6,'spacing-development':12})[group]
    if len(rows) != expected:
        raise ValueError('Exact predeclared native population required')
    if receipt_path.exists():
        previous = read(receipt_path)
        if previous['status'] != 'stopped' or not all(w['reaped'] for w in previous['workers']):
            raise ValueError('Resume only a stopped and reaped group controller')
        attempt = 1
        while (reports/f'native-training-{group}-attempt-{attempt:02d}.json').exists():
            attempt += 1
        write(reports/f'native-training-{group}-attempt-{attempt:02d}.json', previous)
    state = dict(status='running', group=group, active=None, workers=[], reused=[],
        source_sha256=sha(__file__), reservation_sha256=sha(build/'reservation.json'),
        binary_sha256=sha(build/'rheon-pressure-methods-capture'))
    def stop(*_):
        raise InterruptedError('Native data controller interrupted; reaping owned child')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for row in rows:
            for fixture in ([True, False] if row['split'] == 'train' else [False]):
                mode = 'fixture' if fixture else 'capture'
                destination = runs/'broader-native-data'/group/mode/row['trajectory_id']
                if (destination/'receipt.json').exists():
                    saved = read(destination/'receipt.json')
                    if (saved['status'] != ('qualified_fixture' if fixture else 'qualified_pilot')
                            or not saved['native_process']['reaped'] or saved['row'] != row
                            or saved['reservation_sha256'] != state['reservation_sha256']
                            or saved['source_sha256'] != sha(build/'capture_development.py')):
                        raise ValueError('Retained native case differs or failed; inspect before continuing')
                    for frame in saved['frames']:
                        for field in frame['arrays'].values():
                            if sha(destination/field['path']) != field['sha256']:
                                raise ValueError('Retained raw field changed')
                    state['reused'].append(dict(path=str((destination/'receipt.json').relative_to(ROOT)), sha256=sha(destination/'receipt.json')))
                    continue
                if destination.exists():
                    raise ValueError('Incomplete native directory retained')
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:')))*1024
                if available < config['reserve_host_bytes']+config['worker_memory_bytes']:
                    raise MemoryError('Native case memory reserve unavailable')
                name = group+'-'+mode+'-'+row['trajectory_id']
                log = runs/(name+'.log')
                command = [runtime['python'], '-B', str(build/'capture_development.py'), '--build', str(build),
                    '--trajectory', row['trajectory_id'], '--output', str(destination), *(['--fixture'] if fixture else [])]
                state['active'] = name
                write(receipt_path, state)
                tick = time.perf_counter()
                with log.open('xb') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment(runtime),
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    state['active_pid'] = child.pid
                    write(receipt_path, state)
                    try:
                        if child.wait(timeout=660):
                            raise RuntimeError('Native case failed; inspect '+str(log))
                    finally:
                        if child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait(timeout=5)
                        state['workers'].append(dict(name=name, pid=child.pid, returncode=child.returncode,
                            reaped=child.poll() is not None, elapsed_s=time.perf_counter()-tick,
                            log=str(log.relative_to(ROOT)), log_sha256=sha(log)))
                        state['active_pid'] = None
                        write(receipt_path, state)
                print('Completed', name, flush=True)
        state.update(status='completed', active=None)
    except BaseException as error:
        state.update(status='stopped', error=str(error))
        raise
    finally:
        write(receipt_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--group', choices=['teacher','spacing','domain-development','spacing-development'], required=True)
    args = parser.parse_args()
    run(args.build.resolve(), args.group)
