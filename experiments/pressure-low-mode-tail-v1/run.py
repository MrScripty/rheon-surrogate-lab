"""Own two managed controllers; Tuldok controls actual CUDA/CPU admission."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def run(config_path, session_file, directory):
    config = read(config_path)
    python = ROOT/'runs/dependency-migration-v1/app-env/bin/python3.12'
    jobs = []
    started = time.perf_counter()
    status = 'interrupted'
    def stop(*_):
        raise KeyboardInterrupt('Controller owner cancelled; inspect bounded remote Runs before any resubmission')
    signal.signal(signal.SIGTERM, stop)
    def receipt():
        return dict(status=status, elapsed_s=time.perf_counter()-started, config_sha256=sha(config_path),
            controllers=[{k:v for k,v in job.items() if k not in ['process', 'log']} for job in jobs],
            all_owned_controllers_reaped=all(job['process'].poll() is not None for job in jobs),
            remote_scheduler='Existing Tuldok catalog; one CUDA training slot, independent CPU restores',
            automatic_retry=False, source_sha256=sha(__file__))
    try:
        if (directory/'controller-execution.json').exists():
            raise ValueError('Fresh controller execution required; never repeat unknown mutations')
        for arm in config['arms']:
            study = directory/(arm['id']+'-study-01')
            command = [str(python), '-B', str(HERE/'manage.py'), '--config', str(config_path),
                '--assembly', str(directory/(arm['id']+'-project-01/assembly.json')),
                '--qualification', str(directory/(arm['id']+'-qualification.json')),
                '--tuldok', str(ROOT/'runs/dependency-migration-v1/Tuldok'), '--session-file', str(session_file),
                '--output', str(study), '--endpoint', 'http://127.0.0.1:8091',
                '--catalog-id', 'a7da4a8de53541a2a0e2aad08005bb71', '--project-alias', 'current-lab']
            log = (directory/(arm['id']+'-managed.log')).open('xb')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append(dict(arm=arm['id'], model=arm['model_id'], process=process, log=log,
                             pid=process.pid, command=command, study=str(study.relative_to(ROOT)), reaped=False))
            print('Started managed controller', arm['id'], process.pid, flush=True)
        while any(job['process'].poll() is None for job in jobs):
            active = []
            for job in jobs:
                path = ROOT/job['study']/'active.json'
                if path.is_file():
                    active.append(dict(arm=job['arm'], model=job['model'], **read(path)))
            activity = ROOT/'runs/dependency-migration-v1/live-migration/activity.json'
            with (activity.parent/'.publisher.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                value = read(activity)
                value.update(sequence=value['sequence']+1, updated_at=datetime.now(timezone.utc).isoformat(),
                    state='running', phase='Matched small-grid low-mode CUDA screen',
                    message='C60 balanced control and C61 extra low-mode feedback; 1500 updates each, then fresh CPU restores. C53 remains baseline.',
                    progress=dict(managed_runs=active, completed_readbacks=sum((ROOT/job['study']/'receipt.json').exists() for job in jobs)))
                write(activity, value)
            write(directory/'controller-progress.json', receipt())
            time.sleep(10)
        for job in jobs:
            job['returncode'] = job['process'].wait()
            job['reaped'] = True
        status = 'completed' if all(job['returncode']==0 for job in jobs) else 'has_failed_arms'
    finally:
        for job in jobs:
            process = job['process']
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            job.update(returncode=process.returncode, reaped=True)
            job['log'].close()
        write(directory/'controller-execution.json', receipt())
    if status != 'completed':
        raise RuntimeError('Managed controller failure retained; inspect journal, no automatic replay')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'session-file', 'directory']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.session_file.resolve(), args.directory.resolve())
