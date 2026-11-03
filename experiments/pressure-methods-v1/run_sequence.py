"""Own a finite serial list of prequalified managed fits; stop on any failure."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'experiments/pressure-four-pass-distillation-v1'),
               str(ROOT/'runs/dependency-migration-v1/Tuldok'),
               str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
from common import read, write, sha
from qualify_typed_pressure import OwnerHTTP, await_drained
import agent_api


def run(args):
    if args.output.exists() or not 1 <= len(args.arms) <= 8 or len(set(args.arms)) != len(args.arms):
        raise ValueError('Fresh finite serial command required')
    if any(arm not in ['teacher-coarse','teacher-half','teacher-quarter','spacing-data-control',
                      'numerical-coarse','sparse-anchors','learned-coarse',
                      'spacing-zero-feature','spacing-conditioned'] for arm in args.arms):
        raise ValueError('Only declared qualified arms allowed')
    base = ROOT/'runs/pressure-methods-v1'
    prepared = []
    for arm in args.arms:
        config = base/(arm+'-experiment.json')
        assembly = base/(arm+'-project-01/assembly.json')
        qualification = ROOT/'docs/plans/pressure-methods-v1'/(arm+'-qualification.json')
        result = read(qualification)
        if result['status'] != 'passed' or result['optimizer_updates'] != 0:
            raise ValueError('Actual completed zero-update qualification required')
        destination = base/(arm+'-study-01')
        if destination.exists() or destination.with_name(destination.name+'-controller-source').exists():
            raise ValueError('Study already dispatched; inspect without replay')
        prepared.append(dict(arm=arm,config=str(config),assembly=str(assembly),qualification=str(qualification),
                             output=str(destination), qualification_sha256=sha(qualification),assembly_sha256=sha(assembly)))
    args.output.mkdir()
    record = dict(status='waiting_for_prior_training',wait_for_run=args.wait_for_run,prepared=prepared,
                  completed=[],active=None,automatic_retry=False,source_sha256=sha(__file__))
    write(args.output/'execution.json',record)
    owner = OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',
                      base.parent/'tuldok-performance-v1/private/owner-session.txt')
    prior_id = args.wait_for_run
    if args.wait_for_sequence is not None:
        deadline = time.monotonic()+7200
        while True:
            sequence = read(args.wait_for_sequence/'execution.json')
            if sequence['status'] == 'completed':
                prior_id = sequence['completed'][-1]['training_run_id']
                break
            if sequence['status'] == 'stopped' or time.monotonic() >= deadline:
                raise RuntimeError('Prior serial sequence failed or exceeded wait budget; no new fits admitted')
            time.sleep(2)
    prior = await_drained(owner,prior_id,timeout=2100)
    write(args.output/'prior-training-terminal.json',prior)
    process = None
    def stop(*_):
        raise KeyboardInterrupt('Serial fit controller stopped; inspect remote Run custody')
    signal.signal(signal.SIGTERM,stop)
    try:
        for arm in prepared:
            if sha(arm['qualification']) != arm['qualification_sha256'] or sha(arm['assembly']) != arm['assembly_sha256']:
                raise ValueError('Frozen qualified input changed')
            command = [sys.executable,'-B',str(ROOT/'experiments/pressure-methods-v1/manage.py')]
            for name in ['config','assembly','qualification','output']:
                command += ['--'+name,arm[name]]
            command += ['--tuldok',str(ROOT/'runs/dependency-migration-v1/Tuldok'),
                        '--session-file',str(ROOT/'runs/tuldok-performance-v1/private/owner-session.txt'),
                        '--endpoint','http://127.0.0.1:8091','--catalog-id','a7da4a8de53541a2a0e2aad08005bb71',
                        '--project-alias','current-lab']
            record.update(status='running',active=arm['arm'])
            write(args.output/'execution.json',record)
            with (args.output/(arm['arm']+'.log')).open('xb') as log:
                process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                print('Started qualified arm',arm['arm'],flush=True)
                process.wait(timeout=4200)
            if process.returncode != 0:
                raise RuntimeError('Managed arm failed; preserved without retry: '+arm['arm'])
            receipt = read(Path(arm['output'])/'receipt.json')
            record['completed'].append(dict(arm=arm['arm'],training_run_id=receipt['training_run_id'],
                restore_run_id=receipt['restore_run_id'],weights_sha256=receipt['weights_sha256'],
                artifact_parity=receipt['artifact_parity']))
            print('Completed qualified arm',arm['arm'],flush=True)
        record.update(status='completed',active=None)
    except BaseException as error:
        record.update(status='stopped',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        record['owned_controller_reaped'] = process is None or process.poll() is not None
        write(args.output/'execution.json',record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    waiting = parser.add_mutually_exclusive_group(required=True)
    waiting.add_argument('--wait-for-run')
    waiting.add_argument('--wait-for-sequence',type=Path)
    parser.add_argument('--arms',nargs='+',required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    run(args)
