"""Freeze development-only comparisons before generating reserved blind scenes."""
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def select():
    docs=ROOT/'docs/plans/pressure-methods-v3';base=ROOT/'runs/pressure-methods-v3';destination=docs/'selection.json'
    if destination.exists():raise ValueError('Selection already frozen')
    rule=read(docs/'selection-rule.json');selected=[];rankings={};evidence={}
    for family,identifiers in rule['families'].items():
        rows=[]
        for identifier in identifiers:
            path=base/'evaluation'/f'{identifier}.json';result=read(path)
            cases=[r for r in result['rows']if r['population']=='native-development'and r['max_passes']==16]
            if result['status']!='completed'or len(cases)!=8:raise ValueError('Complete predeclared native development coverage required')
            evidence[str(path.relative_to(ROOT))]=sha(path)
            rows.append(dict(identifier=identifier,entry=result['entry'],kind=result['kind'],
                qualified=sum(r['qualified']for r in cases),cases=8,worst_pressure_relative_l2=max(r['pressure_relative_l2']for r in cases)))
        rows.sort(key=lambda r:(-r['qualified'],r['worst_pressure_relative_l2'],r['identifier']))
        rankings[family]=rows;selected.append(dict(rows[0],family=family))
    write(destination,dict(status='frozen_before_blind_capture',created_at=datetime.now(timezone.utc).isoformat(),
        selected=selected,rankings=rankings,evidence=evidence,rule_sha256=sha(docs/'selection-rule.json'),
        source_sha256=sha(__file__),promotion=False,application_acceptance=False))
    capture=[]
    for row in read(docs/'reservation.json')['rows']:
        if row['population']!='blind':continue
        identity=row['trajectory_id'];scene=base/'scenes'/f'{identity}.json';write(scene,row['scene'])
        capture.append(dict(id=identity,arguments=['experiments/pressure-methods-v3/capture.py','--build',str(base/'profile-build-02'),
            '--scene',str(scene),'--output',str(base/'native-data'/identity),'--cpu','19','--save-fields'],timeout=1800))
    write(base/'blind-capture-spec.json',dict(commands=capture,selection_sha256=sha(destination)))
    # One bounded 16-pass evaluation; no redundant eight-pass replay on blind data.
    text=(ROOT/'experiments/pressure-methods-v3/evaluate.py').read_text().replace('else[8,16]','else[16]')
    evaluator=ROOT/'experiments/pressure-methods-v3/evaluate_blind.py'
    if evaluator.exists():raise ValueError('Fresh blind evaluator capture required')
    evaluator.write_text(text)
    choices={r['identifier']:r['kind']for r in selected};choices.update({'v2-headshape':'learned','v2-num6':'numerical'})
    commands=[]
    for identifier,kind in choices.items():
        commands.append(dict(id=identifier,arguments=[str(evaluator),'--entry',str(base/'entries'/f'{identifier}.json'),
            '--kind',kind,'--population','blind','--cpu','21','--output',str(base/'blind-evaluation'/f'{identifier}.json')],timeout=1800))
    write(base/'blind-evaluation-spec.json',dict(commands=commands,selection_sha256=sha(destination)))
    coupled=[]
    for identifier,kind in choices.items():
        if identifier=='v2-num6':continue
        for row in read(docs/'reservation.json')['rows']:
            if row['population']!='blind'or row['scene']['cells_per_metre']!=16:continue
            size=row['scene']['counts'][0];scene=base/'scenes'/f'coupled-blind-n{size}.json'
            if not scene.exists():write(scene,dict(row['scene'],steps=4))
            label=f'{identifier}-n{size}'
            coupled.append(dict(id=label,arguments=[str(base/'coupled-adapter-01.py'),'--entry',str(base/'entries'/f'{identifier}.json'),
                '--kind',kind,'--build',str(base/'profile-build-02'),'--scene',str(scene),'--output',str(base/'coupled'/label),
                '--cpu','18','--passes','16'],timeout=900))
    for size in [64,128]:
        coupled.append(dict(id=f'spectral-n{size}',arguments=['experiments/pressure-methods-v3/capture.py',
            '--build',str(base/'profile-build-02'),'--scene',str(base/'scenes'/f'coupled-blind-n{size}.json'),
            '--output',str(base/'coupled'/f'spectral-n{size}'),'--cpu','18'],timeout=900))
    write(base/'coupled-spec.json',dict(commands=coupled,selection_sha256=sha(destination)))
    print('Frozen native-development selection',[(r['identifier'],r['qualified'])for r in selected],flush=True)

if __name__=='__main__':select()
