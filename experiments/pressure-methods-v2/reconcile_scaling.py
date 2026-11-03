"""Reconcile a deliberately interrupted scaling owner and two finite continuation lanes."""
import json,signal,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/'runs/pressure-methods-v2'
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def run():
 output=RUNS/'scaling-reconciliation-controller-01'
 if output.exists():raise ValueError('Fresh reconciliation required')
 output.mkdir();(output/'controller-source.py').write_bytes(Path(__file__).read_bytes())
 record=dict(status='waiting',active=None,completed=[],owned_child_reaped=True,source_sha256=sha(__file__))
 deadline=time.monotonic()+7200
 def stop(*_):raise InterruptedError('Scaling reconciliation stopped')
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 try:
  while True:
   pending=[]
   for lane in [0,1]:
    path=RUNS/f'scaling-256-lane{lane}-controller-01/execution.json';d=read(path)
    if d['status']=='stopped':raise RuntimeError('Scaling continuation stopped')
    if d['status']!='completed':pending.append(lane)
    elif not d['owned_child_reaped'] or d['active']:raise ValueError('Continuation custody not drained')
   record['pending_lanes']=pending;write(output/'execution.json',record)
   if not pending:break
   if time.monotonic()>deadline:raise TimeoutError('Finite continuation wait expired')
   time.sleep(5)
  original=read(RUNS/'scaling-256-controller-01/execution.json');drain=read(RUNS/'scaling-reschedule-drain.json')
  if original['status']!='stopped' or not original['owned_child_reaped'] or drain['live_matching_processes']:raise ValueError('Interrupted owner custody unresolved')
  manifest=read(RUNS/'scaling-reschedule-manifest.json');identifiers=set()
  for row in manifest['cases']:
   if row['id'] in identifiers:raise ValueError('Scaling request repeated')
   identifiers.add(row['id']);path=ROOT/row['result'];result=read(path)
   if result['status'] not in ['completed','pressure_rejection']:raise ValueError('Incomplete scientific measurement')
   if 'native_process' in result and (not result['native_process']['reaped'] or result['native_process']['returncode']!=0):raise ValueError('Native worker custody unresolved')
   record['completed'].append(dict(**row,result_sha256=sha(path),scientific_status=result['status']))
  expected={r['id'] for r in read(RUNS/'scaling-256-specification.json')['commands']}
  if identifiers!=expected or len(identifiers)!=14:raise ValueError('Reserved scaling matrix incomplete')
  record.update(status='completed',active=None,original_controller_stopped_deliberately=True,original_direct_child_reaped=True,exact_native_binary_absent_at_reschedule=True,
   original_controller_sha256=sha(RUNS/'scaling-256-controller-01/execution.json'),manifest_sha256=sha(RUNS/'scaling-reschedule-manifest.json'))
  print('Reconciled all 14 reserved scaling probes: four preserved, ten continued, all owners drained',flush=True)
 except BaseException as error:
  record.update(status='stopped',error=repr(error));raise
 finally:write(output/'execution.json',record)
if __name__=='__main__':run()
