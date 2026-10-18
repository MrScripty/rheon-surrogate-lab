"""Measure current authenticated reads on a configured immutable Run workload."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def probe(config_path,kind,output):
    if output.exists():raise ValueError('Fresh benchmark receipt required')
    config=json.loads(config_path.read_text())
    sys.path[:0]=[str(ROOT/config['client_source']),str(ROOT/config['owner_adapter_source'])]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    started=time.perf_counter();owner=OwnerHTTP(agent_api,config['endpoint'],config['catalog_id'],ROOT/config['session_file'])
    initialization=time.perf_counter()-started
    run=json.loads((ROOT/config['run_receipt']).read_text());rows=[]
    if kind=='reads':
        artifact=next(r for r in run['artifacts']if r['name']==config['artifact_name'])
        requests=[('runs.get',dict(run_id=run['id'])),('runs.artifact_chunk',dict(run_id=run['id'],name=artifact['name'],expected_sha256=artifact['sha256'],offset=0,max_bytes=4096))]*config['read_repetitions']
    else:
        graph=json.loads((ROOT/config['restore_graph_receipt']).read_text())['result']
        requests=[('graphs.preflight_external',dict(graph_id=graph['id'],revision=graph['revision']))]
    for operation,arguments in requests:
        started=time.perf_counter();result=owner.call(operation,arguments);elapsed=time.perf_counter()-started
        row=dict(operation=operation,elapsed_s=elapsed)
        if operation=='runs.get':
            if result['id']!=run['id']or result['state']!='completed'or result['custody_phase']!='drained':raise ValueError('Target Run identity/custody changed')
            row['completed_drained']=True
        elif operation=='runs.artifact_chunk':
            raw=agent_api.validate_segment(result,arguments)
            if not result['eof']or len(raw)!=artifact['bytes']or hashlib.sha256(raw).hexdigest()!=artifact['sha256']:raise ValueError('Whole tiny artifact did not verify')
            row.update(bytes=len(raw),whole_sha256_verified=True,sha256=artifact['sha256'])
        else:row.update(material_sha256=result['material_sha256'],graph_id=graph['id'],revision=graph['revision'])
        rows.append(row);print(operation,round(elapsed,4),'s',flush=True)
    summary={op:dict(samples=len(values),minimum_s=min(values),median_s=statistics.median(values),maximum_s=max(values))for op in {r['operation']for r in rows}for values in [[r['elapsed_s']for r in rows if r['operation']==op]]}
    receipt=dict(status='completed',kind=kind,owner_initialization_s=initialization,rows=rows,summary=summary,run_id=run['id'],training_started=False,
        scope='Same client, catalog and immutable workload. Read-only current-authority checks; timings are observations, not a universal product SLA.',
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--kind',choices=['reads','preflight'],default='reads');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();probe(a.config.resolve(),a.kind,a.output.resolve())
