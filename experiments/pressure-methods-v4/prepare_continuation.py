"""Exact full-width continuation controls and a solver-aware objective comparison."""
import argparse
from pathlib import Path
import shutil
from v4_support import ROOT,HERE,BASE,read,write,sha
from package_project import package

def prepare(arm,generation=1):
    output=BASE/(arm+f'-project-{generation:02d}')
    if output.exists():raise ValueError('Fresh continuation Project required')
    prior=ROOT/'runs/pressure-methods-v3/v3-native-project-07';witness=read(prior/'source-witness.json')
    for name,expected in witness['sources'].items():
        if sha(prior/name)!=expected:raise ValueError('Original qualified source changed')
    teacher=read(ROOT/'runs/pressure-methods-v3/entries/v3-native.json');weights=ROOT/teacher['restore']/'weights.npz'
    history=read(ROOT/teacher['restore']/'history.json');inventory=history['checkpoints']['checkpoint-2400.npz']['inventory']
    if sha(weights)!=teacher['weights_sha256']:raise ValueError('Completed continuation weights changed')
    shutil.copytree(prior,output,ignore=shutil.ignore_patterns('__pycache__'))
    config=read(output/'config.json');config['id']=arm;config['training']['seed']=2026101906;write(output/'config.json',config)
    objective=read(output/'fine-tuning.json');objective.update(arm=arm,initial_weights_sha256=sha(weights),initial_inventory=inventory);write(output/'fine-tuning.json',objective)
    extra=[]
    if arm=='v4-gcr':
        shutil.copyfile(prior/'model.py',output/'teacher_model.py');shutil.copyfile(prior/'fine_tuning.py',output/'feedback_objective.py')
        shutil.copyfile(HERE/'convergence_objective.py',output/'fine_tuning.py');shutil.copyfile(HERE/'convergence_model.py',output/'model.py')
        extra=['teacher_model.py','feedback_objective.py']
    project=read(output/'project.json')
    for row in project['data']:
        if row['id']=='initial_weights':row.update(path=str(weights.relative_to(ROOT)),sha256=sha(weights))
    write(output/'project.json',project)
    package(output,arm,extra)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=['v4-replay','v4-gcr'],required=True)
    parser.add_argument('--generation',type=int,choices=[1,2],default=1)
    args=parser.parse_args();prepare(args.arm,args.generation)
