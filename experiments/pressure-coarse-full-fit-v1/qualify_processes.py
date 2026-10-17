"""Reproduce leader-only cancellation, then prove owned native/pool drainage."""
import argparse
import hashlib
import json
import os
import signal
import threading
from pathlib import Path
import subprocess
import sys
import time

from process_ownership import become_subreaper, cancel_and_drain, children, drain_adopted


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def descendants(parent):
    links={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:continue
        links.setdefault(int(fields[1]),[]).append(int(p.name))
    result=set();pending=[parent]
    while pending:
        for pid in links.get(pending.pop(),[]):
            if pid not in result:result.add(pid);pending.append(pid)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','regression-template','runtime','control-project','control-artifacts','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();args.output.mkdir(parents=True)
    lab=Path(__file__).resolve().parents[2];started=time.perf_counter();become_subreaper()
    runtime=json.loads(args.runtime.read_text());config=json.loads(args.regression_template.read_text())
    config.update(scene_indices=[16,17],reserved_cpu_count=0,exclude_cpus=[])
    config_path=args.output/'two-scene-config.json';config_path.write_text(json.dumps(config)+'\n')
    benchmark=lab/'experiments/pressure-throughput-feedback-v1/benchmark.py'
    env={**os.environ,'PYTHONPATH':runtime['site_packages'],'CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    reports=[]
    def execute(label,cmd,wait_for_native,legacy=False):
        t=time.perf_counter();child=None
        with(args.output/(label+'.log')).open('x')as log:
            try:
                child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+45;native=[]
                while time.monotonic()<deadline:
                    if child.poll()is not None:raise RuntimeError('Owner ended before cancellation boundary: '+label)
                    below=descendants(child.pid)
                    native=[]
                    for pid in below:
                        try:command=Path(f'/proc/{pid}/cmdline').read_bytes()
                        except FileNotFoundError:continue
                        if Path(config['binary']).name.encode()in command:native.append(pid)
                    if(native if wait_for_native else len(below)>=2):break
                    time.sleep(.05)
                else:raise TimeoutError('Real owned native/pool workers did not start')
                before=sorted(descendants(child.pid))
                if legacy:
                    child.terminate();child.wait(timeout=5)
                    surviving=children();assert surviving,'Leader-only termination did not reproduce residual workers'
                    proof=cancel_and_drain(child,cooperative_seconds=.5)
                    reports.append(dict(case=label,direct_leader_termination_leaves_live_descendants=True,
                        surviving_adopted_pids=surviving,descendants_before=before,native_before=native,
                        rescued_custody=proof,elapsed_s=time.perf_counter()-t,command=cmd))
                else:
                    proof=cancel_and_drain(child,cooperative_seconds=1. if wait_for_native else 20.)
                    assert proof['echild']
                    reports.append(dict(case=label,descendants_before=before,native_before=native,custody=proof,
                        elapsed_s=time.perf_counter()-t,command=cmd))
            finally:
                if child is not None:
                    if child.poll()is None:cancel_and_drain(child,cooperative_seconds=1.)
                    else:drain_adopted()
    for label,legacy in (('legacy-leader-only',True),('owned-benchmark-cancellation',False)):
        cmd=['taskset','-c','5,6',sys.executable,'-B',str(benchmark),'--config',str(config_path),
             '--workers','2','--output',str(args.output/(label+'-campaign'))]
        execute(label,cmd,True,legacy)
    common=lab/'experiments/pressure-current-policy-v1/common_rhs.py'
    common_config=json.loads((common.parent/'configs/preparation.json').read_text());common_config.update(workers=2,cpus=[5,6])
    common_path=args.output/'common-config.json';common_path.write_text(json.dumps(common_config)+'\n')
    cmd=['taskset','-c','5,6',runtime['python'],'-B',str(common),'--root',str(root),'--config',str(common_path),
         '--preparation',str(root/'runs/pressure-current-policy-v1/preparation-02'),'--project',str(args.control_project),
         '--artifacts',str(args.control_artifacts),'--output',str(args.output/'cancelled-common')]
    execute('owned-common-cancellation',cmd,False)
    # Reproduce cancellation precisely while normal-path drainage owns a live child.
    pid=os.fork()
    if pid==0:
        signal.signal(signal.SIGTERM,signal.SIG_DFL)
        time.sleep(30);os._exit(0)
    timer=threading.Timer(.1,lambda:os.kill(os.getpid(),signal.SIGTERM));timer.start()
    cancelled=False;t=time.perf_counter()
    try:drain_adopted(grace_seconds=.3)
    except KeyboardInterrupt:cancelled=True
    finally:timer.join()
    assert cancelled,'Normal drainage swallowed cancellation'
    try:os.waitpid(-1,os.WNOHANG)
    except ChildProcessError:pass
    else:raise AssertionError('Cancellation propagated before kernel ECHILD')
    assert not Path(f'/proc/{pid}').exists(),'Drained child survived cancellation'
    reports.append(dict(case='normal-drain-cancellation',cancellation_deferred_until_echild=True,
        final_echild=True,elapsed_s=time.perf_counter()-t))
    final=drain_adopted();assert final['echild']
    result=dict(status='passed',elapsed_s=time.perf_counter()-started,cases=reports,final_echild=True,
        simulations_scope='Already-consumed development16/17 control trajectories intentionally cancelled at live native work; no candidate qualification or heldout scenes.',
        sources={str(p):digest(p)for p in (Path(__file__),Path(__file__).with_name('process_ownership.py'),benchmark,common,args.regression_template,args.runtime,Path(config['binary']))})
    with(args.output/'receipt.json').open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='passed',elapsed_s=result['elapsed_s'],cases=len(reports),final_echild=True)))


if __name__=='__main__':main()
