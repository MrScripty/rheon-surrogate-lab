"""Replay actual rejection evidence, fail closed negatives and exact passed fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from process_ownership import become_subreaper,cancel_and_drain,drain_adopted
import rejection

HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('root','prefix','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True);started=time.perf_counter();become_subreaper()
    source=HERE.parent/'pressure-throughput-feedback-v1';sys.path.insert(0,str(source))
    import failure,benchmark
    case=a.prefix/'regression/case-01-scene-01'
    error=ValueError('sparse capture schedule mismatch')
    old=failure.classify(error,case);actual=rejection.classify(error,case)
    assert old['failure_class']=='infrastructure'and actual['failure_class']=='scientific_rejection'and actual['native_children_drained']
    receipts=[read(case/name)for name in('native/receipt.json','native-control/receipt.json')]
    attempt=json.loads((case/'native/rejected-attempts.jsonl').read_text())
    # The complete actual trace above proves the positive. Negative copies retain
    # only decision metadata; never duplicate raw pressure arrays for each case.
    minimal=dict(attempt);minimal['attempt']={k:v for k,v in attempt['attempt'].items()if k not in('raw_candidate_pressure','raw_reference_pressure')}
    negatives=[]
    for label in('not-reaped','nonzero-exit','control-hard-stop','no-rejection','top-level-outcome','qualified','transport','memory','timeout','native-fallback','malformed'):
        folder=a.output/label;(folder/'native').mkdir(parents=True);(folder/'native-control').mkdir()
        learned,control=json.loads(json.dumps(receipts));row=json.loads(json.dumps(minimal));exc=error
        if label=='not-reaped':learned['native_process']['reaped']=False
        if label=='nonzero-exit':learned['native_process']['returncode']=1
        if label=='control-hard-stop':control['result']['hard_stop']=True
        if label=='no-rejection':learned['rejected_attempts']=0
        if label=='top-level-outcome':row['outcome']=row['attempt'].pop('outcome')
        if label=='qualified':row['attempt']['outcome']['qualified']=True
        if label=='transport':row['attempt']['outcome']={'kind':'transport_error'}
        if label=='memory':exc=MemoryError('reserve lost')
        if label=='timeout':exc=TimeoutError('worker deadline')
        if label=='native-fallback':learned['native_fallback']=True
        if label=='malformed':row={}
        write(folder/'native/receipt.json',learned);write(folder/'native-control/receipt.json',control)
        (folder/'native/rejected-attempts.jsonl').write_text(json.dumps(row)+'\n')
        result=rejection.classify(exc,folder);assert result==dict(failure_class='infrastructure',native_children_drained=False)
        negatives.append(dict(case=label,result=result))
    runtime=read(a.runtime);config=read(a.prefix/'regression-config.json')
    env={**os.environ,'PYTHONPATH':runtime['site_packages'],'CUDA_VISIBLE_DEVICES':'','TULDOK_TORCH_DEVICE':'cpu',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','RAYON_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    command=['taskset','-c','23',runtime['python'],'-B',str(HERE/'campaign.py'),'episode','--root',str(a.root),
        '--bindings',config['bindings'],'--binary',config['binary'],'--manifest',config['manifest'],'--output',str(a.output/'first-frame'),
        '--scene','0','--fixture','--native-timeout-seconds',str(config['native_timeout_seconds'])]
    child=None;t=time.perf_counter()
    try:
        with(a.output/'first-frame.log').open('x')as log:
            child=subprocess.Popen(command,env=env,cwd=a.root,stdout=log,stderr=subprocess.STDOUT);code=child.wait()
        custody=drain_adopted();assert code==0 and custody['echild']and not custody['signals_sent']
    except BaseException:
        if child is not None:cancel_and_drain(child)
        raise
    old_fixture=a.prefix/'first-frame';new_fixture=a.output/'first-frame'
    before=read(old_fixture/'episode.json');after=read(new_fixture/'episode.json')
    assert before['status']==after['status']=='passed'and before['manifest_sha256']==after['manifest_sha256']
    assert after['native_first_frame_inputs_exact']and not after['cuda_initialized']and after['torch_threads']==1
    assert benchmark.numerical_signature(old_fixture/'native/frames.jsonl')==benchmark.numerical_signature(new_fixture/'native/frames.jsonl')
    assert {k:v for k,v in before['metrics'].items()if k!='receipt_sha256'}=={k:v for k,v in after['metrics'].items()if k!='receipt_sha256'}
    assert before['sparse']==after['sparse']
    assert digest(Path(before['captures'][0]['path']))==digest(Path(after['captures'][0]['path']))
    paths=[Path(__file__),HERE/'campaign.py',HERE/'rejection.py',HERE/'process_ownership.py',source/'benchmark.py',source/'episode.py',source/'failure.py',
        a.runtime,a.prefix/'transfer.json',a.prefix/'common-rhs/receipt.json',a.prefix/'regression-config.json',a.prefix/'timing-config.json',a.prefix/'status.json',a.prefix/'phase-clocks.json',a.prefix/'regression/receipt.json',old_fixture/'episode.json',new_fixture/'episode.json',
        case/'native/receipt.json',case/'native-control/receipt.json',case/'native/rejected-attempts.jsonl',Path(config['manifest'])]
    result=dict(status='passed',format='pressure-current-rejection-qualification-v1',elapsed_s=time.perf_counter()-started,
        legacy_trace_classification=old,current_actual_trace_classification=actual,negatives=negatives,
        first_frame_numerical_output_exact=True,first_frame_command=command,first_frame_command_s=time.perf_counter()-t,
        child_custody=custody,optimizer_updates=0,holdouts_consumed=0,
        sources={str(path):digest(path)for path in paths},scope='Current-schema exception classification only. Actual failed trace reclassified, unknown/infrastructure negatives fail closed, successful first-frame arithmetic/oracle/capture exact. No pressure threshold, model, training or accepted calculation changes.')
    write(a.output/'receipt.json',result);print(json.dumps(dict(status='passed',negatives=len(negatives),first_frame_numerical_output_exact=True,elapsed_s=result['elapsed_s'])))


if __name__=='__main__':main()
