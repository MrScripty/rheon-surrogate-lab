"""Own the serialized bounded training/development collection and record custody."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'docs/plans/full-ai-data-diversity-v1'
BASE=ROOT/'runs/full-ai-data-diversity-v1'


if __name__=='__main__':
    path=BASE/'bulk-capture-controller-01.json'
    if path.exists():raise ValueError('Fresh bulk controller receipt required')
    record=dict(status='running',pid=os.getpid(),started_unix=time.time(),
        active=None,completed=[],optimizer_updates=0,reserved_references_consumed=False,
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        for split,timeout in [('train',10920),('development',3720)]:
            protocol=DOCS/f'{split}-capture-protocol-01.json'
            command=['python3','-B',str(ROOT/'experiments/pressure-methods-v2/runtime.py'),
                '--timeout',str(timeout),str(Path(__file__).with_name('capture_diverse_data_v2.py')),
                '--protocol',str(protocol),'--output',str(BASE/f'{split}-capture-01')]
            child=subprocess.Popen(command,cwd=ROOT)
            record['active']=dict(split=split,pid=child.pid,command=command,
                protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest());save()
            result=child.wait()
            record['completed'].append(dict(split=split,pid=child.pid,returncode=result,reaped=True))
            record['active']=None;save()
            if result:raise RuntimeError('Preserved failed collection: '+split)
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:
        record['updated_unix']=time.time();save()
