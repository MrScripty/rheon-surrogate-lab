"""Freeze development-only candidate choices before any blind access."""
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def select():
    base=ROOT/'runs/pressure-methods-v2';docs=ROOT/'docs/plans/pressure-methods-v2'
    destination=docs/'selection.json'
    if destination.exists():raise ValueError('Selection already frozen; never reselect after blind access')
    arms=['v2-small16','v2-small24','v2-mixed16','v2-retain24','v2-headbase','v2-headshape','v2-num4','v2-num6']
    rankings=[];evidence=[]
    expected={'consumed128':128,'new-rectangular':12,'new-small':4,'old-cube32':4,'old-cube64':4,'old-rectangular':4}
    for arm in arms:
        path=base/'evaluation-01'/arm/'results.json';result=read(path)
        receipt_path=base/(arm+'-study-02/receipt.json');receipt=read(receipt_path)
        if result['status']!='completed' or receipt['status']!='completed' or not all(receipt['artifact_parity'].values()):
            raise ValueError('All eight actual fits and fresh CPU restores must complete before selection')
        rows=[r for r in result['rows'] if r['policy'].get('passes')==8]
        populations={name:sum(r['population']==name for r in rows) for name in expected}
        if len(rows)!=156 or populations!=expected:raise ValueError('Full comparable development coverage required')
        row=dict(arm=arm,qualified=sum(r['qualified'] for r in rows),total=156,
            worst_pressure_relative_l2=max(r['pressure_relative_l2'] for r in rows),
            worst_relative_residual=max(r['relative_residual'] for r in rows),
            standard_qualified=sum(r['qualified'] for r in rows if r['population']=='consumed128'),
            evaluation=str(path.relative_to(ROOT)),evaluation_sha256=sha(path),
            builder_sha256=sha(base/(arm+'-builder-02.py')))
        rankings.append(row)
        evidence.append(dict(arm=arm,evaluation_sha256=sha(path),training_receipt_sha256=sha(receipt_path)))
    def score(row):return (-row['qualified'],row['worst_pressure_relative_l2'],row['arm'])
    learned=min((r for r in rankings if r['arm'] in ['v2-mixed16','v2-retain24','v2-headbase','v2-headshape']),key=score)
    numerical=min((r for r in rankings if r['arm'] in ['v2-num4','v2-num6']),key=score)
    selected=[dict(learned,family='fully-learned'),dict(numerical,family='numerical-hybrid')]
    for row in selected:row['qualification']='all-development-requests' if row['qualified']==156 else 'exploratory-tradeoff-with-failed-gates'
    write(destination,dict(status='frozen_before_blind_access',selected=selected,rankings=sorted(rankings,key=score),
        created_at=datetime.now(timezone.utc).isoformat(),evidence=evidence,
        rule='Within each declared family: eight-pass qualified development-request count first, then worst mean-free pressure relative L2. No timing or blind outcome used.',
        repeats=dict(paired_sampling_and_training_seeds=[2026101825,2026101826],same_initialization_and_data=True),
        blind_scenes_consumed=0,reservation_sha256=sha(docs/'reservation.json'),source_sha256=sha(__file__)))
    print('Frozen development choices',[(r['arm'],r['qualified']) for r in selected],flush=True)

if __name__=='__main__':select()
