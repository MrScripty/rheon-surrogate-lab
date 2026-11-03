"""Read-only reconciliation of the campaign's exact managed Run IDs and terminal custody."""
import json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/'runs/pressure-methods-v2';DOCS=ROOT/'docs/plans/pressure-methods-v2'
sys.path[:0]=[str(ROOT/'experiments/pressure-four-pass-distillation-v1'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1'),str(ROOT/'runs/dependency-migration-v1/Tuldok')]
from common import read,write,sha

def run():
 output=RUNS/'managed-run-audit-02'
 if output.exists():raise ValueError('Fresh managed Run reconciliation required')
 output.mkdir();(output/'controller-source.py').write_bytes(Path(__file__).read_bytes())
 execution=dict(status='waiting',active=None,completed=[],source_sha256=sha(__file__),read_only=True,owned_child_reaped=True)
 write(output/'execution.json',execution);deadline=time.monotonic()+7200
 try:
  while True:
   terminal=read(RUNS/'repeat-training-controller-01/execution.json')
   if terminal['status']=='stopped':raise RuntimeError('Training owner stopped; no retry')
   if terminal['status']=='completed':
    if not terminal['owned_child_reaped'] or terminal['active']:raise ValueError('Training owner custody not drained')
    break
   if time.monotonic()>deadline:raise TimeoutError('Exact training owner wait expired')
   time.sleep(5)
  import agent_api
  from qualify_typed_pressure import OwnerHTTP
  owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
  arms=['v2-small16','v2-small24','v2-mixed16','v2-retain24','v2-headbase','v2-headshape','v2-num4','v2-num6','v2-headshape-s25','v2-headshape-s26','v2-num6-s25','v2-num6-s26']
  targets=[]
  for arm in arms:
   receipt=read(RUNS/(arm+('-study-01' if '-s2' in arm else '-study-02'))/'receipt.json')
   if receipt['status']!='completed' or not all(receipt['artifact_parity'].values()):raise ValueError('Fit/restore parity incomplete')
   targets.extend((arm,kind,receipt[key],'completed') for kind,key in [('fit','training_run_id'),('cpu-restore','restore_run_id')])
  targets.append(('v2-small16-invalid-deadline','rejected-pre-optimizer-launch','b8c8c019dafe483b80615aca3ddc04f7','failed'))
  if len({x[2] for x in targets})!=25:raise ValueError('Expected exactly 25 distinct Run IDs')
  execution.update(status='running')
  rows=[]
  for arm,kind,identifier,expected in targets:
   execution['active']=dict(run_id=identifier);write(output/'execution.json',execution)
   remote=owner.call('runs.get',dict(run_id=identifier));write(output/(identifier+'.json'),remote)
   if remote['state']!=expected or remote['custody_phase']!='drained':raise ValueError('Unexpected terminal Run state/custody: '+identifier)
   rows.append(dict(arm=arm,kind=kind,run_id=identifier,state=remote['state'],custody_phase=remote['custody_phase'],evidence=str((output/(identifier+'.json')).relative_to(ROOT)),sha256=sha(output/(identifier+'.json'))))
  write(DOCS/'managed-run-audit.json',dict(status='passed',read_only=True,successful_runs=24,successful_fits=12,failed_pre_optimizer_launches=1,rows=rows,source_sha256=sha(__file__)))
  execution.update(status='completed',active=None,completed=rows);print('Verified 24 successful managed Runs and one failed pre-optimizer launch; all custody drained',flush=True)
 except BaseException as error:
  execution.update(status='stopped',error=repr(error));raise
 finally:write(output/'execution.json',execution)
if __name__=='__main__':run()
