"""Wait for exact terminal prerequisite receipts, then run a finite declared batch once."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def run(args):
 base=ROOT/'runs/pressure-methods-v2';output=base/args.output
 if output.exists():raise ValueError('Fresh dependent controller required')
 output.mkdir();(output/'controller-source.py').write_bytes(Path(__file__).read_bytes())
 intent=read(base/args.intent);record=dict(status='waiting',active=None,completed=[],source_sha256=sha(__file__),intent_sha256=sha(base/args.intent));child=None
 deadline=time.monotonic()+intent.get('wait_seconds',14400)
 def stop(*_):raise InterruptedError('Dependent controller stopped')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  while True:
   waiting=[];evidence={}
   for name in intent['prerequisites']:
    p=base/name/'execution.json'
    if not p.exists():waiting.append(name);continue
    d=read(p)
    if d['status']=='stopped':raise RuntimeError('Prerequisite stopped: '+name)
    if d['status']!='completed':waiting.append(name);continue
    if d.get('active') or not d.get('owned_child_reaped',False):raise ValueError('Completed prerequisite custody not drained: '+name)
    evidence[name]=sha(p)
   record.update(waiting=waiting,prerequisite_receipts=evidence);write(output/'execution.json',record)
   if not waiting:break
   if time.monotonic()>deadline:raise TimeoutError('Declared prerequisite wait expired')
   time.sleep(5)
  for index,row in enumerate(intent['batches']):
   command=[sys.executable,'-B',str(ROOT/'experiments/pressure-methods-v2/run_batch.py'),'--specification',str(base/row['specification']),'--output',str(base/row['output'])]
   with (output/(str(index)+'.log')).open('xb') as log:
    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(status='running',active=dict(pid=child.pid,specification=row['specification']));write(output/'execution.json',record)
    if child.wait(timeout=row.get('timeout',7200)):raise RuntimeError('Dependent batch failed: '+row['specification'])
   terminal=base/row['output']/'execution.json';d=read(terminal)
   if d['status']!='completed' or not d['owned_child_reaped']:raise ValueError('Batch terminal custody failed')
   record['completed'].append(dict(batch=row,sha256=sha(terminal),reaped=child.poll() is not None));write(output/'execution.json',record)
   print('Completed dependent batch',row['output'],flush=True)
  record.update(status='completed',active=None)
 except BaseException as error:
  record.update(status='stopped',error=repr(error));raise
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  record['owned_child_reaped']=child is None or child.poll() is not None;write(output/'execution.json',record)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--intent',required=True);p.add_argument('--output',required=True);run(p.parse_args())
