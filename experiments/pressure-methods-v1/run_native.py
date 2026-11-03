"""Build and execute qualified native pilots, preserving every worker receipt."""
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


def run():
    reports = ROOT/'docs/plans/pressure-methods-v1'
    runs = ROOT/'runs/pressure-methods-v1'
    config = read(runs/'config.json')
    runtime = read(ROOT/config['runtime'])
    if read(reports/'cohort-02-execution.json')['status'] != 'completed':
        raise ValueError('Finish isolated CPU comparison before native work')
    path = reports/'native-pilots-execution.json'
    if path.exists():
        previous = read(path)
        if previous['status'] != 'stopped' or not all(w['reaped'] for w in previous['workers']):
            raise ValueError('Resume only a stopped, fully reaped native controller')
        index = 1
        while (reports/f'native-pilots-attempt-{index:02d}.json').exists():
            index += 1
        write(reports/f'native-pilots-attempt-{index:02d}.json', previous)
    state = dict(status='running', active=None, workers=[], source_sha256=sha(__file__),
        config_sha256=sha(runs/'config.json'), reused=[])
    def stop(*_):
        raise InterruptedError('Native controller interrupted; reaping owned worker')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def worker(name, arguments, timeout):
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))*1024
        if available < config['reserve_host_bytes']+config['worker_memory_bytes']:
            raise MemoryError('Native worker plus host memory reserve unavailable')
        log = runs/(name+'.log')
        attempt = 1
        while log.exists():
            log = runs/(name+f'-attempt-{attempt:02d}.log')
            attempt += 1
        state['active'] = name
        write(path, state)
        tick = time.perf_counter()
        with log.open('xb') as stream:
            child = subprocess.Popen([runtime['python'], '-B', *map(str, arguments)], cwd=ROOT,
                env=environment(runtime), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state['active_pid'] = child.pid
            write(path, state)
            try:
                if child.wait(timeout=timeout):
                    raise RuntimeError('Native worker failed; inspect '+str(log))
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
                state['workers'].append(dict(name=name, command=list(map(str, arguments)), pid=child.pid,
                    returncode=child.returncode, reaped=child.poll() is not None,
                    elapsed_s=time.perf_counter()-tick, log=str(log.relative_to(ROOT)), log_sha256=sha(log)))
                state['active_pid'] = None
                write(path, state)
        print('Completed', name, flush=True)
    try:
        build = runs/'native-01'
        if not (build/'build-receipt.json').exists():
            worker('native-build', [HERE/'native_prepare.py', 'build', '--output', build], 1800)
        else:
            state['reused'].append(dict(path=str(build/'build-receipt.json'), sha256=sha(build/'build-receipt.json')))
        reservation = read(build/'reservation.json')
        for row in reservation['scenes']:
            for fixture in [True, False]:
                mode = 'fixture' if fixture else 'capture'
                destination = runs/'native-pilots'/mode/row['trajectory_id']
                if (destination/'receipt.json').exists():
                    value = read(destination/'receipt.json')
                    expected = 'qualified_fixture' if fixture else 'qualified_pilot'
                    if (value['status'] != expected or not value['native_process']['reaped']
                            or value['source_sha256'] != sha(HERE/'capture_native.py')
                            or value['reservation_sha256'] != sha(build/'reservation.json')
                            or value['binary_sha256'] != sha(build/'rheon-pressure-methods-capture')):
                        raise ValueError('Retained capture incomplete or changed; inspect before retry: '+str(destination))
                    state['reused'].append(dict(path=str(destination/'receipt.json'), sha256=sha(destination/'receipt.json')))
                    continue
                if destination.exists():
                    raise ValueError('Incomplete native directory retained: '+str(destination))
                worker(mode+'-'+row['trajectory_id'], [HERE/'capture_native.py', '--build', build,
                    '--output', destination, '--trajectory', row['trajectory_id'], *(['--fixture'] if fixture else [])], 660)
        state.update(status='completed', active=None)
        campaign = read(reports/'campaign.json')
        for cohort in campaign['cohorts']:
            if cohort['id'] in [1, 3, 7]:
                cohort.update(status='native_pilots_qualified_training_pending', evidence='native-pilots-execution.json')
        write(reports/'campaign.json', campaign)
    except BaseException as error:
        state.update(status='stopped', error=str(error))
        raise
    finally:
        write(path, state)


if __name__ == '__main__':
    run()
