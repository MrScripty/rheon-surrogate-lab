"""Bounded, resumable local cohort workers; retained evidence is never overwritten."""
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
from common import read, sha, write, environment

REPORTS = ROOT/'docs/plans/pressure-methods-v1'
RUNS = ROOT/'runs/pressure-methods-v1'


def run():
    config = read(HERE/'config.json')
    runtime = read(ROOT/config['runtime'])
    REPORTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    receipt_path = REPORTS/'cohort-02-execution.json'
    if receipt_path.exists():
        old = read(receipt_path)
        if old['status'] == 'completed':
            raise ValueError('Cohort screen already completed; inspect results rather than replay')
        if old['status'] != 'stopped' or not all(w['reaped'] for w in old['workers']):
            raise ValueError('Only a stopped, fully reaped controller can resume')
        index = 1
        while (REPORTS/f'cohort-02-attempt-{index:02d}.json').exists():
            index += 1
        write(REPORTS/f'cohort-02-attempt-{index:02d}.json', old)
    else:
        old = None
    config_path = RUNS/'config.json'
    if config_path.exists():
        if read(config_path) != config:
            raise ValueError('Captured configuration changed')
    else:
        write(config_path, config)
    manifest = dict(schema=config['schema'], status='active',
        objective='Complete and test all eight requested cohorts, follow up promising results, record evidence.',
        request_sha256=sha(config['request_file']), config_sha256=sha(config_path),
        cohorts=[dict(**r, status='not_started') for r in config['cohorts']])
    if not (REPORTS/'campaign.json').exists():
        write(REPORTS/'campaign.json', manifest)
    state = dict(status='running', workers=[], active=None, config_sha256=sha(config_path),
        request_sha256=sha(config['request_file']), controller_source_sha256=sha(__file__), reused=[])
    if old is not None:
        state['prior_attempt_workers'] = old['workers']
    cpu = sorted(os.sched_getaffinity(0))[-1]
    def stop(*_):
        raise InterruptedError('Cohort controller interrupted; reaping owned worker')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def worker(name, command, timeout=None):
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))*1024
        if available < config['reserve_host_bytes']+config['worker_memory_bytes']:
            raise MemoryError('Host reserve plus bounded worker memory unavailable')
        log = RUNS/(name+'.log')
        index = 1
        while log.exists():
            log = RUNS/(name+f'-attempt-{index:02d}.log')
            index += 1
        state['active'] = name
        write(receipt_path, state)
        tick = time.perf_counter()
        with log.open('xb') as stream:
            child = subprocess.Popen([str(runtime['python']), '-B', *map(str, command)], cwd=ROOT,
                env=environment(runtime), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state['active_pid'] = child.pid
            write(receipt_path, state)
            try:
                code = child.wait(timeout=timeout or config['worker_deadline_seconds'])
                if code:
                    raise RuntimeError('Worker failed; inspect '+str(log))
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
                state['workers'].append(dict(name=name, command=list(map(str, command)),
                    pid=child.pid, returncode=child.returncode, reaped=child.poll() is not None,
                    elapsed_s=time.perf_counter()-tick, log=str(log.relative_to(ROOT)), log_sha256=sha(log)))
                state['active_pid'] = None
                write(receipt_path, state)
        print('Completed', name, flush=True)
    try:
        worker('numerical-qualification', ['-m', 'unittest', 'discover', '-s', HERE, '-p', 'test_numerics.py', '-v'], 120)
        bank = RUNS/'manufactured-bank'
        if not (bank/'bank.json').exists():
            worker('freeze-bank', [HERE/'hybrid.py', 'freeze', '--config', config_path,
                                 '--output', bank, '--cpu', cpu], 120)
        else:
            value = read(bank/'bank.json')
            if value['config_sha256'] != sha(config_path) or value['data_sha256'] != sha(bank/'samples.npz'):
                raise ValueError('Retained bank identity mismatch')
            state['reused'].append(dict(path=str(bank/'bank.json'), sha256=sha(bank/'bank.json')))
        for identifier in config['checkpoints']:
            output = RUNS/'cohort-02'/identifier
            if (output/'results.json').exists():
                value = read(output/'results.json')
                if (value['status'] != 'completed' or value['config_sha256'] != sha(config_path)
                        or value['bank_sha256'] != sha(bank/'bank.json')
                        or value['source_sha256'] != sha(HERE/'hybrid.py')
                        or value['numerics_sha256'] != sha(HERE/'numerics.py')):
                    raise ValueError('Retained completed worker identity mismatch')
                state['reused'].append(dict(path=str(output/'results.json'), sha256=sha(output/'results.json')))
                continue
            if output.exists():
                raise ValueError('Incomplete output retained; inspect before making a fresh attempt')
            worker('hybrid-'+identifier, [HERE/'hybrid.py', 'evaluate', '--config', config_path,
                   '--bank', bank, '--checkpoint', identifier, '--cpu', cpu, '--output', output])
        state.update(status='completed', active=None)
    except BaseException as error:
        state.update(status='stopped', error=str(error))
        raise
    finally:
        write(receipt_path, state)


if __name__ == '__main__':
    run()
