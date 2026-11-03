"""Capture the two bounded conditional full-AI architecture screens."""
import argparse
from pathlib import Path
import shutil
import subprocess
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha,runtime
from package_pressure import package
from common import environment


def run(args):
    registration=read(DOCS/'architecture-registration.json')
    identifier=args.identifier;spec=registration['arms'][identifier]
    output=BASE/(identifier+f'-project-{args.project_generation:02d}')
    if output.exists():raise ValueError('Fresh architecture Project required')
    prior=BASE/'a-combined-project-01'
    shutil.copytree(prior,output,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(output/'model.py',output/'student_original.py')
    shutil.copyfile(output/'mixed_order.py',output/'original_fit_sampling.py')
    shutil.copyfile(HERE/'architecture_model.py',output/'model.py')
    shutil.copyfile(HERE/'architecture_sampling.py',output/'mixed_order.py')
    write(output/'architecture-variant.json',dict(spec,registration_sha256=sha(DOCS/'architecture-registration.json')))
    config=read(output/'config.json');config['id']=identifier
    updates=spec['updates'];checkpoints=spec['checkpoints']
    config['training'].update(updates=updates,checkpoints=checkpoints,schedule=spec['schedule'])
    config['correction']['steps']=spec['pressure_stages']
    if spec['pressure_stages']==1:config['schema']='rheon-water-residual-experiment-v12'
    write(output/'config.json',config)
    objective=read(output/'fine-tuning.json')
    objective.update(arm=identifier,schema='rheon-fine-tuning-objective-v2',training_passes=spec['pressure_stages'])
    write(output/'fine-tuning.json',objective)
    source=(output/'feedback_objective.py').read_text()
    source=source.replace("spec['schema'] != 'rheon-fine-tuning-objective-v1'","spec['schema'] != 'rheon-fine-tuning-objective-v2'")
    source=source.replace("spec['training_passes'] not in (2, 4)","spec['training_passes'] not in (1, 2, 4)")
    (output/'feedback_objective.py').write_text(source)
    source=(output/'policy.py').read_text()
    source=source.replace("    if value.get('schema') == 'rheon-water-residual-experiment-v11':", "    if value.get('schema') == 'rheon-water-residual-experiment-v12':\n        require(value.get('correction')=={'activation':'linear','steps':1}, 'one_stage_architecture', 'registered one-stage full AI')\n        import copy\n        prior=copy.deepcopy(value);prior['schema']='rheon-water-residual-experiment-v11';prior['correction']['steps']=2\n        experiment(prior)\n        return decode(canonical(value))\n    if value.get('schema') == 'rheon-water-residual-experiment-v11':",1)
    source=source.replace("def execution(config):", "def execution(config):\n    if config['schema']=='rheon-water-residual-experiment-v12':\n        return execution({**config,'schema':'rheon-water-residual-experiment-v11'})",1)
    (output/'policy.py').write_text(source)
    if spec['variant']=='one-stage-modes8':
        source=(output/'physical_residual.py').read_text().replace('min(4,n)','min(8,n)').replace('reshape(4,4,4)','reshape(8,8,8)')
        (output/'physical_residual.py').write_text(source)
    for mode in ('train','restore'):
        recipe=read(output/(mode+'-recipe.json'))
        kept=[r for r in recipe['outputs']if not r['path'].startswith('checkpoint-')]
        template_npz=next(r for r in recipe['outputs']if r['path']=='checkpoint-1200.npz')
        template_json=next((r for r in recipe['outputs']if r['path']=='checkpoint-1200.json'),None)
        kept.extend(dict(template_npz,path=f'checkpoint-{u:03d}.npz')for u in checkpoints)
        if template_json is not None:kept.append(dict(template_json,path=f'checkpoint-{updates:03d}.json'))
        recipe['outputs']=kept
        if mode=='restore':
            old_keys=[k for k in recipe['inputs']if k.startswith('trained_checkpoint_')]
            cap=recipe['inputs'][old_keys[0]]
            for k in old_keys:recipe['inputs'].pop(k)
            recipe['arguments']=[r for r in recipe['arguments']if not ('input'in r and r['input'].startswith('trained_checkpoint_'))
                and not ('literal'in r and r['literal'].startswith('--trained-checkpoint-'))]
            for u in checkpoints:
                slot=f'trained_checkpoint_{u:03d}_npz';recipe['inputs'][slot]=cap
                recipe['arguments'].extend([dict(literal=f'--trained-checkpoint-{u:03d}-npz'),dict(input=slot)])
        write(output/(mode+'-recipe.json'),recipe)
    selected=runtime()
    subprocess.run([selected['python'],'-B',str(HERE/'initialize_architecture.py'),'--project',str(output)],
        cwd=ROOT,env=environment(selected),check=True,timeout=180)
    package(output,identifier,['fit-sampling.json','student_original.py','original_fit_sampling.py','architecture-variant.json'],
        experiment_path=BASE/(identifier+f'-experiment-{args.project_generation:02d}.json'))
    write(BASE/(identifier+'-preparation.json'),dict(status='captured',project=str(output.relative_to(ROOT)),
        registration_sha256=sha(DOCS/'architecture-registration.json'),source_sha256=sha(__file__),optimizer_updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--identifier',required=True)
    parser.add_argument('--project-generation',type=int,default=1)
    run(parser.parse_args())
