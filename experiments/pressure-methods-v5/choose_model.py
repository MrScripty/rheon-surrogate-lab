"""Choose only among completed matched tiny fits using development evidence."""
from v5_support import ROOT,BASE,DOCS,read,write,sha
import statistics
def choose():
    output=DOCS/'model-selection.json'
    if output.exists():raise ValueError('Model selection already frozen')
    records=[]
    for arm in ['v5-raw','v5-multilevel','v5-spectrum']:
        path=BASE/'evaluation-final'/(arm+'-gcr.json');d=read(path);rows=d['rows']
        if d['status']!='completed'or len(rows)!=220:raise ValueError('All candidate development evidence required')
        new=[r for r in rows if r['population'].startswith('v5-')];old=[r for r in rows if not r['population'].startswith('v5-')]
        study=BASE/(arm+'-study-01');receipt=read(study/'receipt.json')
        if receipt['status']!='completed'or not all(receipt['artifact_parity'].values()):raise ValueError('Completed fit and exact CPU restore required')
        score=[sum(r['qualified']for r in new),sum(r['qualified']for r in old),-statistics.median(r['coarse_calls']for r in new),-max(r['pressure_relative_l2']for r in new)]
        records.append(dict(arm=arm,score=score,entry=d['entry'],evaluation=str(path.relative_to(ROOT)),evaluation_sha256=sha(path),study_receipt_sha256=sha(study/'receipt.json')))
    selected=max(records,key=lambda r:tuple(r['score']))
    write(output,dict(status='model_frozen_before_reuse_development_and_blind_capture',selected=selected,candidates=records,
        procedure='Maximize new native/spectral gates, then retained gates; minimize median coarse calls, then worst new pressure error. No exploratory timing used to select model weights.',
        selection_rule_sha256=sha(DOCS/'selection-rule.json'),source_sha256=sha(__file__),no_further_fits=True,promotion=False))
    commands=[]
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!='rollout-development':continue
        for capacity,strategy in [(0,'contribution'),(2,'contribution'),(4,'contribution'),(4,'tail')]:
            suffix='-tail'if strategy=='tail'else''
            name=f"{row['trajectory_id']}-tiny-recycle{capacity}"+suffix;out=BASE/'rollouts-final'/name
            args=['runs/pressure-methods-v5/native-capture-03.py','--trajectory',row['trajectory_id'],'--method','gcr','--reuse',str(capacity),'--retention',strategy,'--optimized',
                '--entry',str((BASE/'entries'/(selected['arm']+'.json')).relative_to(ROOT)),
                '--build','runs/pressure-methods-v4/streamed-build-01','--output',str(out.relative_to(ROOT)),'--cpu','19','--states']
            commands.append(dict(id=name,runtime='torch',arguments=args,timeout=900))
    write(BASE/'rollout-model-plan-02.json',dict(commands=commands))
    print('Frozen tiny model using development only:',selected['arm'],flush=True)
if __name__=='__main__':choose()

