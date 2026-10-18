"""Own one bounded managed controller and publish its visible experiment state."""
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
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read, sha, write


def run(args, *, qualification_key=None):
    if args.output.exists():
        raise ValueError('Fresh execution directory required; no mutation replay')
    args.output.mkdir()
    config = read(args.config)
    arm = config['arms'][0]
    if len(config['arms']) != 1:
        raise ValueError('Exactly one qualified arm per controller')
    qualification=read(args.qualification)
    required=qualification_key or ('single_case'if arm['id']=='single-mode-fit'else'matched_control')
    if qualification['status']!='passed' or required not in qualification:
        raise ValueError('Complete arm-specific qualification required before any admission')
    study = args.output.parent/(arm['id']+'-study-01')
    if study.exists():
        raise ValueError('Fresh managed study required; inspect existing custody before resubmission')
    command = [str(ROOT/'runs/dependency-migration-v1/app-env/bin/python3.12'),'-B',
        str(ROOT/'experiments/pressure-low-mode-tail-v1/manage.py'),
        '--config',str(args.config),'--assembly',str(args.assembly),'--qualification',str(args.qualification),
        '--tuldok',str(ROOT/'runs/dependency-migration-v1/Tuldok'),'--session-file',str(args.session_file),
        '--output',str(study),'--endpoint','http://127.0.0.1:8091',
        '--catalog-id','a7da4a8de53541a2a0e2aad08005bb71','--project-alias','current-lab']
    def stop(*_):
        raise KeyboardInterrupt('Controller cancelled; bounded remote Runs retain their own custody')
    signal.signal(signal.SIGTERM,stop)
    process = None
    started = time.perf_counter()
    status = 'interrupted'
    try:
        with (args.output/'managed.log').open('xb') as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                activity = ROOT/'runs/dependency-migration-v1/live-migration/activity.json'
                active = read(study/'active.json') if (study/'active.json').is_file() else {}
                with (activity.parent/'.publisher.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    value = read(activity)
                    value.update(sequence=value['sequence']+1,updated_at=datetime.now(timezone.utc).isoformat(),
                        state='running',phase=arm['model_id']+' '+arm['id'],
                        message='CUDA float32 training, then fresh CPU restore. C53 remains the baseline.',
                        progress=dict(managed_run=active,elapsed_s=time.perf_counter()-started))
                    write(activity,value)
                time.sleep(10)
            status = 'completed' if process.wait() == 0 else 'failed'
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        write(args.output/'execution.json',dict(status=status,elapsed_s=time.perf_counter()-started,
            command=command,pid=process.pid if process else None,returncode=process.returncode if process else None,
            owned_controller_reaped=process is not None and process.poll() is not None,
            config_sha256=sha(args.config),assembly_sha256=sha(args.assembly),qualification_sha256=sha(args.qualification),
            source_sha256=sha(__file__),automatic_retry=False,
            remote_custody='Existing Tuldok catalog owns bounded CUDA and CPU Runs; controller termination is not remote cessation.'))
    if status != 'completed':
        raise RuntimeError('Managed failure retained; inspect journal without replay')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','assembly','qualification','session-file','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    arguments=parser.parse_args()
    for name,value in vars(arguments).items():
        setattr(arguments,name,value.resolve())
    run(arguments)
