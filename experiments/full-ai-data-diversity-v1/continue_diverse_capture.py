"""Collect remaining scenes while retaining each native rejection as unresolved."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DOCS=ROOT/'docs/plans/full-ai-data-diversity-v1'
BASE=ROOT/'runs/full-ai-data-diversity-v1'


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__=='__main__':
    path=BASE/'capture-continuation-02.json'
    if path.exists():raise ValueError('Fresh continuation custody record required')
    old=read(BASE/'train-capture-01/bank.json')
    if old['status']!='failed':raise ValueError('Prior collection must be authoritatively terminal')
    known=read(BASE/'native-rejection-replay-01/result.json')
    if known['status']!='reproduced_native_reference_rejection':raise ValueError('Actual native rejection evidence required')
    record=dict(status='running',completed=list(old['trajectories']),
        reference_rejections=[dict(identifier='diversity-train-g001-cpm50',split='train',
            evidence=str((BASE/'native-rejection-replay-01/result.json').relative_to(ROOT)),
            cause='Native Courant0.5010751759 exceeds unchanged0.5 gate at macro step110',
            requires_reference_temporal_refinement=True)],active=None,optimizer_updates=0,
        reserved_references_consumed=False,controller_sha256=digest(Path(__file__)))
    def save():
        record['updated_unix']=time.time();path.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        for split in ('train','development'):
            protocol=DOCS/f'{split}-capture-continuation-protocol-02.json'
            reg=read(protocol)
            skip={r['identifier']for r in record['completed']+record['reference_rejections']}
            for row in reg['rows']:
                identifier=row['identifier']
                if identifier in skip:continue
                output=BASE/f'{split}-capture-continuation-02'/identifier
                command=['python3','-B',str(ROOT/'experiments/pressure-methods-v2/runtime.py'),
                    '--timeout','3720',str(Path(__file__).with_name('capture_diverse_data_v3.py')),
                    '--protocol',str(protocol),'--identifier',identifier,'--output',str(output)]
                child=subprocess.Popen(command,cwd=ROOT)
                record['active']=dict(identifier=identifier,split=split,pid=child.pid,
                    output=str(output.relative_to(ROOT)),command=command);save()
                code=child.wait()
                bank=read(output/'bank.json')
                if code==0 and bank['status']=='completed'and len(bank['trajectories'])==1:
                    record['completed'].append(bank['trajectories'][0])
                else:
                    rejection=output/identifier/'reference-rejection.json'
                    custody=output/identifier/'failed-native-custody.json'
                    if not rejection.exists() or not custody.exists()or not read(custody)['reaped']:
                        raise RuntimeError('Unexpected collection failure; preserve evidence for diagnosis')
                    record['reference_rejections'].append(dict(identifier=identifier,split=split,
                        evidence=str(rejection.relative_to(ROOT)),requires_reference_temporal_refinement=True,
                        native_process=read(custody)))
                record['active']=None;save()
        record['status']='attempts_completed_native_reference_rejections_remain'
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:save()
