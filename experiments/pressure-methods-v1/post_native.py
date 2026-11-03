"""Verify/pack pilots, then run serial numerical and periodic trajectory comparisons."""
import os
from pathlib import Path
import random
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
    if read(reports/'native-pilots-execution.json')['status'] != 'completed':
        raise ValueError('Qualified complete native pilots required first')
    receipt_path = reports/'followup-execution.json'
    if receipt_path.exists():
        raise ValueError('Existing follow-up custody must be reconciled before any continuation')
    state = dict(status='running', active=None, workers=[], source_sha256=sha(__file__))
    def stop(*_):
        raise InterruptedError('Follow-up controller interrupted; reaping owned worker')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def worker(name, arguments, timeout=660):
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))*1024
        if available < config['reserve_host_bytes']+config['worker_memory_bytes']:
            raise MemoryError('Follow-up memory reserve unavailable')
        log = runs/(name+'.log')
        state['active'] = name
        write(receipt_path, state)
        tick = time.perf_counter()
        with log.open('xb') as stream:
            child = subprocess.Popen([runtime['python'], '-B', *map(str, arguments)], cwd=ROOT,
                env=environment(runtime), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state['active_pid'] = child.pid
            write(receipt_path, state)
            try:
                if child.wait(timeout=timeout):
                    raise RuntimeError('Follow-up worker failed; inspect '+str(log))
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
                write(receipt_path, state)
        print('Completed', name, flush=True)
    try:
        worker('native-pilot-report', [HERE/'pilot_report.py'], 120)
        worker('pack-domain-view', [HERE/'pack_domains.py'], 120)
        rows = []
        rng = random.Random(2026101809)
        for repeat in range(3):
            kinds = ['native-control', 'full-numerical', 'neural']
            rng.shuffle(kinds)
            rows.extend(dict(id=f'{kind}-repeat{repeat+1}', kind=kind, repeat=repeat+1, steps=512, passes=8, interval=16)
                        for kind in kinds)
        rows.extend(dict(id=f'periodic-{interval}', kind='periodic', steps=512, passes=8, interval=interval)
                    for interval in [4, 16, 64])
        rows.extend(dict(id=f'{kind}-4', kind=kind, steps=128, passes=4, interval=16) for kind in ['before', 'after'])
        plan = dict(schema='rheon-pressure-methods-native-followup-v1', rows=rows,
            seed=2026101809, population='Previously consumed development scene from the pass-sweep material; no unused scene accessed.',
            timing='Serial full native-loop timing including IPC and all native reference diagnostics; paired policies share instrumentation.',
            diagnostic_thresholds=[.001, .01],
            remaining='The existing wire exposes pressure drift; velocity/tracer drift requires a separate diagnostic extension.')
        write(reports/'followup-plan.json', plan)
        for row in rows:
            worker('trajectory-'+row['id'], [HERE/'trajectory.py', '--kind', row['kind'],
                '--steps', row['steps'], '--passes', row['passes'], '--interval', row['interval'],
                '--output', runs/'trajectories'/row['id']])
        state.update(status='completed', active=None)
        campaign = read(reports/'campaign.json')
        for cohort in campaign['cohorts']:
            if cohort['id'] == 2:
                cohort.update(status='offline_screen_and_pressure_trajectory_followups_completed',
                              followup='followup-execution.json')
            elif cohort['id'] == 8:
                cohort.update(status='pressure_trajectory_screen_completed_velocity_tracer_pending',
                              evidence='followup-execution.json')
        write(reports/'campaign.json', campaign)
    except BaseException as error:
        state.update(status='stopped', error=str(error))
        raise
    finally:
        write(receipt_path, state)


if __name__ == '__main__':
    run()
