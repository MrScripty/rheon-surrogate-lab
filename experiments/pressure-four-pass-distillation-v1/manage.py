"""Run one qualified arm through the existing Tuldok catalog and CPU restore.

No private trainer fallback. Mutations use the existing durable dispatch journal;
terminal failures are retained, drained and closed rather than silently retried.
"""
import argparse
from pathlib import Path
import signal
import sys
import time

from common import ROOT, read, sha, write


class FailedRun(RuntimeError):
    def __init__(self, owner, run):
        self.owner, self.run = owner, run
        super().__init__('Managed arm ended without successful completion; no retry')


def execute(args):
    for name in ['config','assembly','qualification','tuldok','session_file','output']:
        setattr(args,name,getattr(args,name).resolve())
    config = read(args.config)
    source = ROOT/config['source_lab']/'experiments'
    sys.path[:0] = [str(source/'pressure-throughput-feedback-v1'),str(source/'dependency-migration-v1')]
    import run_study
    from qualify_typed_pressure import Journal
    assembly = read(args.assembly)
    qualification = read(args.qualification)
    project = ROOT/assembly['project']['path']
    if not (qualification['status']=='passed' and qualification['optimizer_updates']==0
            and qualification['project_sha256']==sha(project)
            and qualification['source_witness_sha256']==sha(project.with_name('source-witness.json'))
            and qualification['exact_initial_inventory'] and qualification['frozen_teacher_unchanged']
            and assembly['updates']==config['updates'] and assembly['limits']['attempt_slots']==1):
        raise ValueError('Exact objective/initialization qualification required before managed admission')
    original = run_study.await_drained
    def wait(owner, run_id, timeout=2100):
        deadline = time.monotonic()+timeout
        last = None
        while True:
            run = owner.call('runs.get',dict(run_id=run_id))
            state = run['state'],run.get('custody_phase')
            if state != last:
                write(args.output/('state-'+run_id+'.json'),run)
                print('Managed run',run_id,*state,flush=True)
                last = state
            if run['state'] in ('completed','failed','cancelled','interrupted','unknown'):
                write(args.output/('terminal-'+run_id+'.json'),run)
                if run['custody_phase']!='drained':
                    raise RuntimeError('Managed run cessation unknown; inspect retained state without resubmission')
                if run['state']!='completed':
                    raise FailedRun(owner,run)
                return run
            if time.monotonic() >= deadline:
                raise TimeoutError('Managed readback expired; inspect durable state without resubmission')
            time.sleep(2)
    run_study.await_drained = wait
    args.root=ROOT
    args.collect_existing=False
    started=time.perf_counter()
    try:
        run_study.execute(args)
    except FailedRun as failure:
        active=read(args.output/'active.json')
        state=failure.owner.call('studies.get',dict(study_id=active['study_id']))['study']
        if state['open']:
            journal=Journal(args.output/'mutations')
            paths=list((args.output/'mutations').glob('*.json'))
            if any(read(path)['status']!='acknowledged'for path in paths):
                raise RuntimeError('Unresolved mutation prevents automatic study closure')from failure
            journal.index=max(int(path.name.split('-',1)[0])for path in paths)
            key='fourpass-stop-'+run_study.sha(dict(output=str(args.output.resolve())))[:24]
            body=dict(study_id=state['id'],request_key=key,revision=state['revision'])
            journal.mutation('close-failed-study','studies.stop',body,lambda:failure.owner.call('studies.stop',body))
        write(args.output/'failure.json',dict(status='failed',run_id=failure.run['id'],state=failure.run['state'],
                  drained=True,automatic_retry=False,elapsed_s=time.perf_counter()-started))
        raise
    finally:
        run_study.await_drained=original


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','assembly','qualification','tuldok','session-file','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ['endpoint','catalog-id','project-alias']:
        parser.add_argument('--'+name,required=True)
    arguments=parser.parse_args()
    def stop(*_):
        raise KeyboardInterrupt('Managed controller interrupted; bounded catalog runs retain their own custody')
    signal.signal(signal.SIGTERM,stop)
    execute(arguments)
