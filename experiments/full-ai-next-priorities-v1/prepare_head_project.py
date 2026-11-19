"""Capture a small generic CUDA head-calibration Project and exact CPU restore."""
import argparse
from pathlib import Path
import shutil
import sys
import uuid
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha
from head_descriptor import describe


def run(args):
    registration=read(DOCS/'head-calibration-registration.json')
    identifier='a-head-calibration'+(f'-s{args.seed_generation}'if args.seed_generation>1 else'')
    output=BASE/(identifier+f'-project-{args.project_generation:02d}')
    if output.exists():raise ValueError('Fresh head Project required')
    bank=BASE/'head-teacher-bank-01';qualified=read(BASE/'head-teacher-qualification-01.json')
    if qualified['status']!='passed'or qualified['teacher_manifest_sha256']!=sha(bank/'manifest.json'):
        raise ValueError('Qualified offline teacher required')
    prior=BASE/'a-combined-project-01';old=read(prior/'source-witness.json')
    for name,digest in old['sources'].items():
        if sha(prior/name)!=digest:raise ValueError('Captured pressure source changed')
    output.mkdir(parents=True)
    for name in old['sources']:shutil.copyfile(prior/name,output/name)
    shutil.copytree(prior/'tuldok_extension_sdk',output/'tuldok_extension_sdk',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(prior/'model.py',output/'calibration_base_model.py')
    shutil.copyfile(HERE/'head_calibration_model.py',output/'model.py')
    extras=['calibration_base_model.py','head_descriptor.py','head_objective.py','head_utils.py','head_train.py','calibration.json','custody_validator.py']
    for name in ('head_descriptor.py','head_objective.py','head_utils.py','head_train.py'):shutil.copyfile(HERE/name,output/name)
    shutil.copyfile(prior/'custody-validator.py',output/'custody_validator.py')
    calibration={key:registration[key]for key in ('schema','updates','batch_size','learning_rate','schedule','wall_seconds','seed','sampling_seed','objective')}
    calibration.update(identifier=identifier,seed=calibration['seed']+args.seed_generation-1,
        sampling_seed=calibration['sampling_seed']+args.seed_generation-1,registration_sha256=sha(DOCS/'head-calibration-registration.json'))
    write(output/'calibration.json',calibration)
    config=read(output/'config.json');config['id']=identifier
    config['training'].update(updates=calibration['updates'],checkpoints=[0,calibration['updates']],seed=calibration['seed'],
        sampling_seed=calibration['sampling_seed'],learning_rate=calibration['learning_rate'],schedule=calibration['schedule'])
    write(output/'config.json',config)
    entry=read(ROOT/'runs/pressure-methods-v4/entries/v4-student50-control.json');origin=ROOT/entry['restore']
    for path,digest in entry['identities'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Original C107 checkpoint identity changed')
    initial=read(origin/'history.json')['checkpoints']['checkpoint-2400.npz']['inventory']
    write(output/'initial-inventory.json',dict(inventory=initial,weights_sha256=sha(origin/'weights.npz'),
        normalization_sha256=sha(origin/'normalization.json'),C107_entry=entry))
    static=dict(teacher=bank/'teacher.npz',teacher_manifest=bank/'manifest.json',initial_weights=origin/'weights.npz',
        initial_inventory=output/'initial-inventory.json',normalization=origin/'normalization.json',
        runtime_manifest=ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    names=set(old['sources'])|set(extras)
    witness=dict(schema='rheon-head-calibration-source-witness-v1',base_witness_sha256=sha(prior/'source-witness.json'),
        architecture=describe(old['architecture'],calibration),experiment_sha256=sha(output/'config.json'),
        normalization_sha256=sha(static['normalization']),sources={name:sha(output/name)for name in sorted(names)},
        inputs={slot:dict(bytes=path.stat().st_size,sha256=sha(path))for slot,path in static.items()},
        task='Frozen C107 fine network; head calibration from offline numerical labels; no inverse at inference')
    write(output/'source-witness.json',witness)
    prefix=str(output.relative_to(ROOT))+'/'
    sources=[dict(path=prefix+name,sha256=sha(output/name))for name in sorted(names|{'source-witness.json'})]
    sources.extend(dict(path=prefix+'tuldok_extension_sdk/'+p.name,sha256=sha(p))for p in sorted((output/'tuldok_extension_sdk').glob('*.py')))
    declarations=[('weights.npz','checkpoint',2*1024**2),('initial-weights.npz','checkpoint',2*1024**2),
        ('checkpoint-000.npz','checkpoint',2*1024**2),(f'checkpoint-{calibration["updates"]:03d}.npz','checkpoint',2*1024**2),
        ('normalization.json','preprocessing',1024**2),('history.json','training_diagnostics',2*1024**2),
        ('probe.npy','predictions',1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map={'trained_weights':('weights.npz',2*1024**2),'trained_initial':('initial-weights.npz',2*1024**2),
        'trained_history':('history.json',2*1024**2),'trained_probe':('probe.npy',1024**2),'trained_receipt':('receipt.json',2*1024**2)}
    recipes=[]
    for mode in ('train','restore'):
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
    document=dict(schema_version=2,id=uuid.uuid4().hex,name='Full AI '+identifier,data=[dict(id=slot,label=slot,path=str(path.relative_to(ROOT)),sha256=sha(path))for slot,path in static.items()],
        recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(output/'project.json',document)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'));import projects,recipe_contract
    projects.manifest(document)
    for mode in ('train','restore'):recipe_contract.decode_definition((output/(mode+'-recipe.json')).read_bytes())
    write(output/'assembly.json',dict(status='captured',project=prefix+'project.json',project_sha256=sha(output/'project.json'),sources=sources,
        static_inputs={slot:dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)for slot,path in static.items()},
        restore_artifact_map={key:value[0]for key,value in restore_map.items()},config=calibration,
        cuda_qualification=str((BASE/(identifier+'-cuda-qualification-01.json')).relative_to(ROOT)),
        model_file='model.py',objective_file='head_objective.py',trainer_file='head_train.py',
        source_sha256=sha(__file__),optimizer_updates_executed=0,no_numerical_solver_at_candidate_inference=True))
    print('Captured head-only calibration Project',identifier,'updates',calibration['updates'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed-generation',type=int,default=1)
    parser.add_argument('--project-generation',type=int,default=1);run(parser.parse_args())
