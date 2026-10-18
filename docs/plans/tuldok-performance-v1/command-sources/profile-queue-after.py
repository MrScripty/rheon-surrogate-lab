"""Profile the actual read owner against the live catalog without a second supervisor."""
from pathlib import Path
from types import SimpleNamespace
import cProfile,json,pstats,sys,time
ROOT=Path.cwd();source=ROOT/'runs/tuldok-performance-v1/Tuldok';sys.path.insert(0,str(source))
from metadata import Metadata
from access import Access
from projects import Projects
from pipeline import Pipelines
from runs import Runs
args=Path('/proc/855603/cmdline').read_bytes().decode().split('\0');data=args[args.index('--data')+1]
selection=json.loads((ROOT/'runs/dependency-migration-v1/ops/live-selection.private.json').read_text());db=Metadata(ROOT/data,dsn=selection['dsn'],create=False)
ds=SimpleNamespace(path=db.path,db=db,lock=db.lock,access=Access(db,db.path));ds.projects=Projects(ds,{})
try:
 ds.pipelines=Pipelines(ds);ds.runs=Runs(ds,ds.pipelines)
 assert not ds.runs.runtime_owner,'Never run a second catalog supervisor'
 token=(ROOT/'runs/tuldok-performance-v1/private/owner-session.txt').read_text().strip()
 run=json.loads((ROOT/'runs/pressure-shape-decay-v1/shape-later-decay-study-01/terminal-e17624f982b54e858698207227010410.json').read_text());artifact=next(r for r in run['artifacts']if r['name']=='training-profile.json')
 body=dict(expected_sha256=artifact['sha256'],offset=0,max_bytes=4096)
 states=db.execute("SELECT document::jsonb->>'state',count(*) FROM workspace_runs GROUP BY 1").fetchall()
 print('Saved Run state counts',[(r[0],r[1])for r in states],flush=True)
 assert not any(r[0]in ['queued','running','cancelling']and r[1]for r in states),'Only profile the historical scan while no work is active'
 profile=cProfile.Profile();wall=time.perf_counter();cpu=time.process_time()
 with ds.access.bind(token),ds.lock:
  loaded=profile.runcall(lambda:ds.runs._in_states(('queued',)))
 wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
 profile.dump_stats(str(ROOT/'runs/tuldok-performance-v1/queue-scan-after.pstats'))
 record=dict(run_count=len(loaded),queued=sum(r['state']=='queued'for r in loaded),wall_s=wall,cpu_s=cpu,
  scope='Candidate selected-state queue lookup, once; no executor or dispatch, no active Runs.')
 (ROOT/'runs/tuldok-performance-v1/queue-scan-after.json').write_text(json.dumps(record,indent=2)+'\n')
 print(record,flush=True);pstats.Stats(profile).sort_stats('cumulative').print_stats(15)

finally:ds.projects.close();db.close()
