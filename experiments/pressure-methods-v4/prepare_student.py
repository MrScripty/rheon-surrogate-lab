"""Prepare paired distillation/non-distillation Projects for each student width."""
import argparse
from pathlib import Path
import shutil
from v4_support import ROOT,HERE,BASE,read,write,sha
from package_project import package

def change(text,before,after):
    if text.count(before)!=1:raise ValueError('Captured source seam changed: '+before[:100])
    return text.replace(before,after)

def prepare(arm):
    output=BASE/(arm+'-project-01')
    if output.exists():raise ValueError('Fresh student Project required')
    prior=ROOT/'runs/pressure-methods-v3/v3-native-project-07';witness=read(prior/'source-witness.json')
    for name,expected in witness['sources'].items():
        if sha(prior/name)!=expected:raise ValueError('Original qualified source changed')
    width=75 if '75'in arm else 50
    initial=BASE/(f'student{width}-initialization-01');initialization=read(initial/'initialization.json')
    teacher=read(ROOT/'runs/pressure-methods-v3/entries/v3-native.json')
    shutil.copytree(prior,output,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(prior/'model.py',output/'teacher_model.py')
    shutil.copyfile(prior/'fine_tuning.py',output/'feedback_objective.py')
    for source,target in [('student_model.py','model.py'),('student_slice.py','student_slice.py'),('distillation.py','fine_tuning.py')]:shutil.copyfile(HERE/source,output/target)
    spec=dict(schema='rheon-pressure-student-distillation-v1',width_fraction=initialization['width_fraction'],weight=1. if arm.endswith('distill')else 0.,
        teacher_weights_sha256=teacher['weights_sha256'],teacher_inventory=initialization['teacher_inventory'],
        source_entry=teacher,teacher_frozen=True,teacher_used_at_inference=False)
    write(output/'distillation.json',spec)
    config=read(output/'config.json');config['id']=arm;config['training'].update(seed=2026101905,learning_rate=3e-4,
        schedule=[dict(update=1801,learning_rate=3e-5),dict(update=2251,learning_rate=1e-5)])
    write(output/'config.json',config)
    objective=read(output/'fine-tuning.json');objective.update(arm=arm,initial_weights_sha256=initialization['weights_sha256'],initial_inventory=initialization['inventory'])
    write(output/'fine-tuning.json',objective)
    optimizer=(output/'head_optimizer.py').read_text().replace('learning_rate*1000.','learning_rate*(10/3)').replace('lr_multiplier=1000.','lr_multiplier=(10/3)')
    (output/'head_optimizer.py').write_text(optimizer)
    trainer=(output/'train.py').read_text()
    trainer=change(trainer,"return {**DATA,'initial_weights':'initial.npz',", "return {**DATA,'teacher_weights':'teacher.npz','initial_weights':'initial.npz',")
    trainer=change(trainer,'fine_tuning.initialize(network, args.initial_weights, helpers)','fine_tuning.initialize(network, args.initial_weights, helpers, args.teacher_weights)')
    trainer=change(trainer,"    parser.add_argument('--initial-weights',type=Path,required=True)","    parser.add_argument('--initial-weights',type=Path,required=True)\n    parser.add_argument('--teacher-weights',type=Path,required=True)")
    trainer=change(trainer,"    receipt['fine_tuning_statistics'] = dict(fine_tuning.STATS)","    receipt['fine_tuning_statistics'] = dict(fine_tuning.STATS)\n    require(helpers.state_inventory(fine_tuning.FULL_TEACHER) == fine_tuning.DISTILL['teacher_inventory'], 'frozen_teacher', 'every full teacher weight remains unchanged')\n    require(all(p.grad is None and not p.requires_grad for p in fine_tuning.FULL_TEACHER.parameters()), 'frozen_teacher', 'teacher owns no gradient')\n    receipt['frozen_full_teacher_unchanged'] = True")
    (output/'train.py').write_text(trainer)
    project=read(output/'project.json')
    for row in project['data']:
        if row['id']=='initial_weights':row.update(path=str((initial/'weights.npz').relative_to(ROOT)),sha256=initialization['weights_sha256'])
    teacher_path=ROOT/teacher['restore']/'weights.npz'
    project['data'].append(dict(id='teacher_weights',label='Frozen completed v3-native teacher',path=str(teacher_path.relative_to(ROOT)),sha256=sha(teacher_path)))
    write(output/'project.json',project)
    for mode in ['train','restore']:
        recipe=read(output/(mode+'-recipe.json'));recipe['inputs']['teacher_weights']={'max_bytes':2*1024**2}
        recipe['arguments'] +=[{'literal':'--teacher-weights'},{'input':'teacher_weights'}]
        write(output/(mode+'-recipe.json'),recipe)
    package(output,arm,['teacher_model.py','feedback_objective.py','student_slice.py','distillation.json'])
    print('Prepared student',width,'distillation',spec['weight'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=['v4-student75-control','v4-student75-distill','v4-student50-control','v4-student50-distill'],required=True)
    prepare(parser.parse_args().arm)
