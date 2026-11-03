"""Final read-only host process and immutable-evidence audit; no admissions or promotion."""
import hashlib,json,os
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/'runs/pressure-methods-v2';DOCS=ROOT/'docs/plans/pressure-methods-v2'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def run():
 final=RUNS/'final-report-refresh-controller-01/execution.json';controller=read(final)
 if controller['status']!='completed' or controller['active'] or not controller['owned_child_reaped']:raise ValueError('Final evidence owner not drained')
 result=read(DOCS/'results.json')
 if result['status']!='completed' or result['missing'] or result['successful_fits']!=12 or len(result['managed_run_ids'])!=24 or result['promoted'] or result['application_acceptance_claimed']:raise ValueError('Required campaign evidence incomplete')
 if len(result['application'])!=16:raise ValueError('All final application operating groups required')
 if any(r['qualified']!=16 for r in result['blind'] if r['passes']==8):raise ValueError('Blind result summary changed')
 for path,expected in result['evidence'].items():
  if sha(ROOT/path)!=expected:raise ValueError('Final evidence identity changed: '+path)
 if sha(ROOT/'experiments/pressure-methods-v2/build_report.py')!=result['source_sha256']:raise ValueError('Final report builder changed')
 figures=read(DOCS/'figure-evidence.json')
 if figures['status']!='completed' or figures['report_sha256']!=sha(DOCS/'results.json'):raise ValueError('Figures do not describe final report')
 for artifact in figures['artifacts']:
  if sha(ROOT/artifact['path'])!=artifact['sha256']:raise ValueError('Scientific export changed')
 native=[str(RUNS/p) for p in ['single-build-01/rheon-single-track','harder-drift-build-02/rheon-drift-rollout','large-native-build-01/rheon-large-single-track']]
 scripts={str(p.resolve()) for p in (ROOT/'experiments/pressure-methods-v2').glob('*.py')}
 def live_owned_workers():
  found=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit() or int(p.name)==os.getpid():continue
   try:args=(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0');args=[x.decode(errors='replace') for x in args]
   except (FileNotFoundError,PermissionError,ProcessLookupError):continue
   matched=args and args[0] in native
   if args and Path(args[0]).name.startswith('python'):
    for arg in args[1:]:
     if arg.endswith('.py') and ('\n' not in arg) and str((ROOT/arg).resolve()) in scripts:matched=True
   if matched:found.append(dict(pid=int(p.name),program=args[0]))
  return found
 deadline=time.monotonic()+10
 while True:
  workers=live_owned_workers()
  if not workers:break
  if time.monotonic()>deadline:raise ValueError('Owned process still live: '+str(workers))
  time.sleep(.5)
 audit=dict(status='passed',successful_fits=12,successful_managed_runs=24,failed_pre_optimizer_launches=1,
  original_model_cohorts=result['original_baselines'],source_evidence_files_verified=len(result['evidence']),
  no_live_owned_measurement_workers=True,host_namespace_process_audit=True,promoted=False,application_acceptance_claimed=False,
  final_controller_sha256=sha(final),report_sha256=sha(DOCS/'results.json'),html_sha256=sha(DOCS/'results.html'),figure_evidence_sha256=sha(DOCS/'figure-evidence.json'),source_sha256=sha(Path(__file__)))
 (DOCS/'final-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))
if __name__=='__main__':run()
