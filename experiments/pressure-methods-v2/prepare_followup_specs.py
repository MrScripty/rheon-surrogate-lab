"""Create finite preparation/training commands for the frozen paired seed repeats."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/pressure-four-pass-distillation-v1'))
from common import read,write,sha

def prepare():
    base=ROOT/'runs/pressure-methods-v2';docs=ROOT/'docs/plans/pressure-methods-v2'
    selection_path=docs/'selection.json';selection=read(selection_path)
    if selection['status']!='frozen_before_blind_access':raise ValueError('Frozen development choices required')
    preparation=[];training=[];identifiers=[]
    for row in selection['selected']:
        arm=row['arm']
        for seed in [2026101825,2026101826]:
            identifier=arm+'-s'+str(seed)[-2:];identifiers.append(identifier)
            project=base/(identifier+'-project-01');qualification=docs/(identifier+'-qualification-01.json')
            preparation.append(dict(id='prepare-'+identifier,runtime='control',arguments=[
                str(ROOT/'experiments/pressure-methods-v2/prepare_repeat.py'),'--arm',arm,'--seed',str(seed)],timeout=300))
            preparation.append(dict(id='qualify-'+identifier,arguments=[str(ROOT/'experiments/pressure-methods-v2/qualify_shapes.py'),
                '--project',str(project),'--output',str(qualification)],timeout=600))
            arguments=[str(ROOT/'experiments/pressure-methods-v1/manage.py')]
            for key,path in [('config',base/(identifier+'-experiment.json')),('assembly',project/'assembly.json'),
                ('qualification',qualification),('output',base/(identifier+'-study-01')),
                ('tuldok',ROOT/'runs/dependency-migration-v1/Tuldok'),
                ('session-file',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')]:
                arguments+=['--'+key,str(path)]
            arguments+=['--endpoint','http://127.0.0.1:8091','--catalog-id','a7da4a8de53541a2a0e2aad08005bb71','--project-alias','current-lab']
            training.append(dict(id=identifier,runtime='control',arguments=arguments,timeout=4200))
    for name,commands in [('repeat-preparation-specification.json',preparation),('repeat-training-specification.json',training)]:
        path=base/name
        if path.exists():raise ValueError('Follow-up specification already exists')
        write(path,dict(commands=commands,selection_sha256=sha(selection_path),source_sha256=sha(__file__)))
    source=ROOT/'experiments/pressure-methods-v2/evaluate_after_training.py';text=source.read_text()
    before="    arms=['v2-small16','v2-small24','v2-mixed16','v2-retain24','v2-headbase','v2-headshape']"
    if text.count(before)!=1:raise ValueError('Evaluation population seam changed')
    text=text.replace(before,'    arms='+repr(identifiers))
    text=text.replace('ROOT=Path(__file__).resolve().parents[2]','ROOT=Path('+repr(str(ROOT))+')')
    for before,after in [('training-controller-02/execution.json','repeat-training-controller-01/execution.json'),
        ('-study-02','-study-01'),('-project-02','-project-01'),("'evaluation-01'","'replicate-evaluation-01'"),
        ('evaluation-controller-01','repeat-evaluation-controller-01'),('time.monotonic()+14400','time.monotonic()+18000')]:
        if before not in text:raise ValueError('Repeat evaluator seam changed: '+before)
        text=text.replace(before,after)
    evaluator=base/'repeat-evaluator.py'
    if evaluator.exists():raise ValueError('Fresh repeat evaluator source required')
    evaluator.write_text(text)
    write(docs/'repeat-execution-plan.json',dict(status='prepared_not_dispatched',selection_sha256=sha(selection_path),
        identifiers=identifiers,original_evaluator_sha256=sha(source),captured_evaluator_sha256=sha(evaluator),
        preparation_specification_sha256=sha(base/'repeat-preparation-specification.json'),
        training_specification_sha256=sha(base/'repeat-training-specification.json')))
    print('Prepared',len(identifiers),'paired seed-repeat commands and evaluator',flush=True)

if __name__=='__main__':prepare()
