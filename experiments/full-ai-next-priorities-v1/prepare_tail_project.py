"""Capture a managed final-tail fit from a verified frozen parent and local seeded head."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    registration=read(DOCS/'tail-head-registration.json')
    if args.seed_generation not in(1,2,3):raise ValueError('Bounded registered seed generations required')
    identifier='a-tail8'+(f'-s{args.seed_generation}'if args.seed_generation>1 else'')
    output=BASE/(identifier+f'-project-{args.project_generation:02d}')
    initial_output=BASE/(identifier+f'-initial-{args.project_generation:02d}')
    if output.exists()or initial_output.exists():raise ValueError('Fresh Project and initialization required')
    bank=BASE/'tail-teacher-bank-01';qualification=read(BASE/'tail-teacher-qualification-01.json')
    if qualification['status']!='passed'or qualification['teacher_manifest_sha256']!=sha(bank/'manifest.json'):
        raise ValueError('Qualified eight-mode teacher required')
    parent_entry_path=BASE/'entries/a-head-calibration-s2-routed.json';parent_entry=read(parent_entry_path)
    for name,digest in parent_entry['identities'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Parent identity changed')
    prior=ROOT/parent_entry['project'];old=read(prior/'source-witness.json')
    output.mkdir()
    for name,digest in old['sources'].items():
        if sha(prior/name)!=digest:raise ValueError('Parent inference source changed')
        shutil.copyfile(prior/name,output/name)
    shutil.copytree(prior/'tuldok_extension_sdk',output/'tuldok_extension_sdk',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(prior/'model.py',output/'parent_model.py')
    copies={'tail_head_pressure_model.py':'model.py','tail_head_descriptor.py':'tail_head_descriptor.py',
        'tail_physical.py':'tail_physical.py','tail_head_objective.py':'head_objective.py',
        'tail_head_train.py':'head_train.py','head_utils.py':'head_utils.py'}
    for source,target in copies.items():shutil.copyfile(HERE/source,output/target)
    custody=ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01/custody-validator.py'
    shutil.copyfile(custody,output/'custody_validator.py')
    calibration={key:registration[key]for key in('schema','updates','batch_size','learning_rate','schedule','wall_seconds','seed','sampling_seed','objective')}
    calibration.update(identifier=identifier,seed=calibration['seed']+args.seed_generation-1,
        sampling_seed=calibration['sampling_seed']+args.seed_generation-1,registration_sha256=sha(DOCS/'tail-head-registration.json'))
    write(output/'calibration.json',calibration)
    config=read(output/'config.json');config['id']=identifier
    config['training'].update(updates=calibration['updates'],checkpoints=[0,calibration['updates']],seed=calibration['seed'],
        sampling_seed=calibration['sampling_seed'],learning_rate=calibration['learning_rate'],schedule=calibration['schedule'])
    write(output/'config.json',config)
    write(output/'initial-context.json',dict(parent_entry=str(parent_entry_path.relative_to(ROOT)),teacher_bank=str(bank.relative_to(ROOT))))
    subprocess.run([sys.executable,'-B',str(ROOT/'experiments/pressure-methods-v2/runtime.py'),'--timeout','180',
        str(HERE/'initialize_tail_project.py'),'--project',str(output),'--initial-output',str(initial_output)],cwd=ROOT,check=True)
    initial=read(initial_output/'initial-inventory.json');artifacts=ROOT/parent_entry['restore']
    static=dict(teacher=bank/'teacher.npz',teacher_manifest=bank/'manifest.json',initial_weights=initial_output/'weights.npz',
        initial_inventory=initial_output/'initial-inventory.json',normalization=artifacts/'normalization.json',
        runtime_manifest=ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    names=set(old['sources'])|set(copies.values())|{'parent_model.py','custody_validator.py','calibration.json','initial-context.json'}
    witness=dict(schema='rheon-learned-tail-source-v1',parent_witness_sha256=sha(prior/'source-witness.json'),
        parent_entry_sha256=sha(parent_entry_path),architecture=initial['architecture'],
        experiment_sha256=sha(output/'config.json'),normalization_sha256=sha(static['normalization']),
        sources={name:sha(output/name)for name in sorted(names)},
        inputs={slot:dict(bytes=path.stat().st_size,sha256=sha(path))for slot,path in static.items()},
        task='Final neural8-mode tail; all94956parentparameters frozen; no inverse at inference')
    write(output/'source-witness.json',witness)
    prefix=str(output.relative_to(ROOT))+'/'
    sources=[dict(path=prefix+name,sha256=sha(output/name))for name in sorted(names|{'source-witness.json'})]
    sources.extend(dict(path=prefix+'tuldok_extension_sdk/'+path.name,sha256=sha(path))for path in sorted((output/'tuldok_extension_sdk').glob('*.py')))
    declarations=[('weights.npz','checkpoint',2*1024**2),('initial-weights.npz','checkpoint',2*1024**2),
        ('checkpoint-000.npz','checkpoint',2*1024**2),(f'checkpoint-{calibration["updates"]:03d}.npz','checkpoint',2*1024**2),
        ('normalization.json','preprocessing',1024**2),('history.json','training_diagnostics',2*1024**2),
        ('probe.npy','predictions',1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map={'trained_weights':('weights.npz',2*1024**2),'trained_initial':('initial-weights.npz',2*1024**2),
        'trained_history':('history.json',2*1024**2),'trained_probe':('probe.npy',1024**2),'trained_receipt':('receipt.json',2*1024**2)}
    recipes=[]
    for mode in('train','restore'):
        inputs={slot:dict(max_bytes=16*1024**2)for slot in static}
        arguments=[dict(literal=mode),dict(literal='--config'),dict(source=prefix+'calibration.json'),
            dict(literal='--input-evidence'),dict(input_evidence=True),dict(literal='--output-dir'),dict(output_dir=True)]
        for slot in static:arguments.extend([dict(literal='--'+slot.replace('_','-')),dict(input=slot)])
        if mode=='restore':
            for slot,(name,cap)in restore_map.items():
                inputs[slot]=dict(max_bytes=cap);arguments.extend([dict(literal='--'+slot.replace('_','-')),dict(input=slot)])
        recipe=dict(schema_version=1,format='external_recipe_v1',id=identifier+'-'+mode,name=identifier+' '+mode,version=1,
            runtime='managed_torch_python_v1',runtime_snapshot_sha256=sha(static['runtime_manifest']),runtime_device='cuda'if mode=='train'else'cpu',
            sources=sources,entrypoint=prefix+'head_train.py',arguments=arguments,parameters={},inputs=inputs,
            outputs=[dict(path=name,role='evaluation_result'if name=='receipt.json'and mode=='restore'else role,max_bytes=cap,required=True)for name,role,cap in declarations],
            environment={},resources=dict(wall_seconds=1800 if mode=='train'else 600,max_output_bytes=32*1024**2,cpu_cores=1,memory_bytes=4*1024**3))
        write(output/(mode+'-recipe.json'),recipe);recipes.append(dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=sha(output/(mode+'-recipe.json'))))
    document=dict(schema_version=2,id=uuid.uuid4().hex,name='Full AI '+identifier,
        data=[dict(id=slot,label=slot,path=str(path.relative_to(ROOT)),sha256=sha(path))for slot,path in static.items()],
        recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(output/'project.json',document)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'));import projects,recipe_contract
    projects.manifest(document)
    for mode in('train','restore'):recipe_contract.decode_definition((output/(mode+'-recipe.json')).read_bytes())
    write(output/'assembly.json',dict(status='captured',project=prefix+'project.json',project_sha256=sha(output/'project.json'),sources=sources,
        static_inputs={slot:dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)for slot,path in static.items()},
        restore_artifact_map={key:value[0]for key,value in restore_map.items()},config=calibration,
        cuda_qualification=str((BASE/(identifier+'-cuda-qualification-01.json')).relative_to(ROOT)),
        source_sha256=sha(__file__),optimizer_updates_executed=0,parent_entry_sha256=sha(parent_entry_path),
        no_numerical_solver_at_candidate_inference=True))
    print('Captured managed learned-tail Project',identifier,'128938total/33982trainable',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-generation',type=int,default=1);parser.add_argument('--project-generation',type=int,default=1)
    run(parser.parse_args())
