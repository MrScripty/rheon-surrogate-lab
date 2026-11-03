"""Freeze finite blind commands from the already frozen model/policy selection."""
from v4_support import ROOT,BASE,DOCS,read,write,sha

def prepare():
    selection=read(DOCS/'selection.json')
    if selection['status']!='frozen_before_blind_capture':raise ValueError('Frozen complete development selection required')
    capture_file=BASE/'blind-capture-plan-01.json';evaluation_file=BASE/'blind-evaluation-plan-01.json'
    if capture_file.exists()or evaluation_file.exists():raise ValueError('Fresh blind command reservation required')
    captures=[]
    for row in read(DOCS/'scene-reservation.json')['rows']:
        if row['population']!='blind':continue
        identity=row['trajectory_id']
        captures.append(dict(id=identity,timeout=900,arguments=['experiments/pressure-methods-v4/streamed_capture.py','--trajectory',identity,
            '--policy','spectral','--build',str((BASE/'streamed-build-01').relative_to(ROOT)),
            '--output',str((BASE/'native-data'/(identity+'-spectral')).relative_to(ROOT))]))
    candidates={r['identifier']:r for r in selection['selected']+selection['references']}
    entries=BASE/'blind-entries';entries.mkdir()
    evaluations=[]
    for identifier,row in candidates.items():
        path=entries/(identifier+'.json');write(path,row['entry'])
        evaluations.append(dict(id=identifier,timeout=1200,arguments=['experiments/pressure-methods-v4/evaluate_candidate.py',
            '--entry',str(path.relative_to(ROOT)),'--policy',row['policy'],'--population','blind','--cpu','21',
            '--output',str((BASE/'blind-evaluation'/(identifier+'.json')).relative_to(ROOT))]))
    write(capture_file,dict(commands=captures,selection_sha256=sha(DOCS/'selection.json'),source_sha256=sha(__file__)))
    write(evaluation_file,dict(commands=evaluations,selection_sha256=sha(DOCS/'selection.json'),source_sha256=sha(__file__)))
    print('Prepared',len(captures),'blind scenes and',len(evaluations),'fixed-policy blind evaluations',flush=True)

if __name__=='__main__':prepare()
