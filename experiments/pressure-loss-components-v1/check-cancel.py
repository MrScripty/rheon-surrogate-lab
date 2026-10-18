"""Exercise the real supervisor's cancellation with three owned sleeper fixtures."""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[0]/'pressure-four-pass-distillation-v1'))
from common import read,write,sha


def fixture(config,scratch):
    spec=importlib.util.spec_from_file_location('component_supervisor',HERE/'diagnose.py')
    diagnose=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnose)
    original=subprocess.Popen;jobs=[]
    def spawn(requested,**kwargs):
        actual=[sys.executable,'-c','import time; time.sleep(120)']
        child=original(actual,**kwargs);jobs.append(dict(pid=child.pid,starttime=Path('/proc',str(child.pid),'stat').read_text().split()[21],requested=requested,actual=actual))
        write(scratch/'fixture-children.json',jobs);return child
    diagnose.subprocess.Popen=spawn
    try:diagnose.run(config,scratch/'owner',scratch/'results')
    finally:diagnose.subprocess.Popen=original


def check(config,scratch,output):
    scratch.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--fixture','--config',str(config),'--scratch',str(scratch)]
    log=(scratch/'parent.log').open('xb');parent=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children=[];completed=False;started=time.monotonic()
    try:
        while time.monotonic()-started<10:
            path=scratch/'fixture-children.json'
            if path.exists():
                children=read(path)
                if len(children)==3:break
            if parent.poll()is not None:raise RuntimeError('Fixture parent exited before ready')
            time.sleep(.05)
        assert len(children)==3
        parent.send_signal(signal.SIGTERM);code=parent.wait(timeout=15)
        receipt=read(scratch/'owner/execution.json')
        assert code!=0 and receipt['status']=='failed'and len(receipt['workers'])==3
        assert all(w['reaped']and w['returncode']==-signal.SIGTERM for w in receipt['workers'])
        write(output,dict(status='passed',scope='Real supervisor and three independent child process groups; sleep replaces numerical worker code only.',
            parent_reaped=True,child_returncodes=[w['returncode']for w in receipt['workers']],all_children_reaped=True,
            elapsed_s=time.monotonic()-started,supervisor_source_sha256=sha(HERE/'diagnose.py'),source_sha256=sha(__file__)))
        completed=True
        print('Cancellation terminated and reaped all three owned fixture workers',flush=True)
    finally:
        if not completed:
            for job in children:
                try:
                    if Path('/proc',str(job['pid']),'stat').read_text().split()[21]==job['starttime']:os.killpg(job['pid'],signal.SIGKILL)
                except (FileNotFoundError,ProcessLookupError):pass
        if parent.poll()is None:
            os.killpg(parent.pid,signal.SIGKILL);parent.wait()
        log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--fixture',action='store_true');a=p.parse_args()
    if a.fixture:fixture(a.config.resolve(),a.scratch.resolve())
    else:check(a.config.resolve(),a.scratch.resolve(),a.output.resolve())
