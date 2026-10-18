"""Compare current typed and raw authenticated HTTP for the same bounded read."""
from pathlib import Path
import cProfile,json,pstats,sys,time
ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
import agent_api
from qualify_typed_pressure import OwnerHTTP
owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
run=json.loads((ROOT/'runs/pressure-shape-decay-v1/shape-later-decay-study-01/terminal-e17624f982b54e858698207227010410.json').read_text());artifact=next(r for r in run['artifacts']if r['name']=='training-profile.json')
body=dict(expected_sha256=artifact['sha256'],offset=0,max_bytes=4096);arguments=dict(run_id=run['id'],name=artifact['name'],**body);results=[]
for mode in ['raw','typed']:
 profile=cProfile.Profile();t=time.perf_counter();cpu=time.process_time()
 if mode=='raw':value=profile.runcall(owner.raw,'POST','/api/workspace/runs/'+run['id']+'/artifact-segment/'+artifact['name'],body)
 else:value=profile.runcall(owner.call,'runs.artifact_chunk',arguments)
 elapsed=time.perf_counter()-t;cpu=time.process_time()-cpu;assert value['total_bytes']==189
 profile.dump_stats(str(ROOT/('runs/tuldok-performance-v1/http-'+mode+'.pstats')));results.append(dict(mode=mode,wall_s=elapsed,client_cpu_s=cpu))
 print(mode,'wall_s',round(elapsed,4),'client_cpu_s',round(cpu,4),flush=True);pstats.Stats(profile).sort_stats('cumulative').print_stats(18)
(ROOT/'runs/tuldok-performance-v1/http-profile.json').write_text(json.dumps(results,indent=2)+'\n')
