"""Capture the registered viscosity-calibration managed Project and CPU restore."""
import argparse
from pathlib import Path
import shutil
import sys
import uuid
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha


def run(args):
    output=args.output.resolve()
    if output.exists():raise ValueError('Fresh viscosity Project required')
    registration=read(DOCS/'viscosity-calibration-registration-01.json')
    for name,digest in registration['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Registered viscosity calibration source changed')
    bank=BASE/'viscosity-calibration-teacher-01';manifest=read(bank/'manifest.json')
    if manifest['status']!='native_qualified'or manifest['archive_sha256']!=sha(bank/'teacher.npz'):
        raise ValueError('Qualified viscosity teacher required')
    output.mkdir(parents=True)
    for name in ('viscosity_calibration_model.py','viscosity_calibration_train.py','managed_fit_artifacts.py'):
        shutil.copyfile(HERE/name,output/name)
    prior=BASE/'b2-provisional-aux-project-01/custody_validator.py'
    shutil.copyfile(prior,output/'custody_validator.py')
    config={k:registration[k]for k in ('schema','updates','batch_size','learning_rate','seed','sampling_seed','wall_seconds','double_tolerance','published_tolerance')}
    config['identifier']='viscosity-calibration-01';config['registration_sha256']=sha(DOCS/'viscosity-calibration-registration-01.json')
    write(output/'config.json',config)
    prefix=str(output.relative_to(ROOT))+'/'
    names=['viscosity_calibration_model.py','viscosity_calibration_train.py','managed_fit_artifacts.py','custody_validator.py','config.json']
    sources=[dict(path=prefix+name,sha256=sha(output/name))for name in names]
    static=dict(teacher=bank/'teacher.npz',teacher_manifest=bank/'manifest.json',runtime_manifest=ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    declarations=[('weights.npz','checkpoint',1024**2),('initial-weights.npz','checkpoint',1024**2),
        ('history.json','training_diagnostics',1024**2),('probe.npy','predictions',1024**2),
        ('normalization.json','preprocessing',1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map=dict(trained_weights='weights.npz',trained_initial='initial-weights.npz',trained_history='history.json',trained_probe='probe.npy',trained_receipt='receipt.json')
    recipes=[]
    for mode in ('train','restore'):
        inputs={slot:dict(max_bytes=32*1024**2 if slot=='teacher'else 16*1024**2)for slot in static}
        arguments=[dict(literal=mode),dict(literal='--config'),dict(source=prefix+'config.json'),
            dict(literal='--input-evidence'),dict(input_evidence=True),dict(literal='--output-dir'),dict(output_dir=True)]
        for slot in static:arguments.extend([dict(literal='--'+slot.replace('_','-')),dict(input=slot)])
        if mode=='restore':
            for slot in restore_map:
                inputs[slot]=dict(max_bytes=2*1024**2);arguments.extend([dict(literal='--'+slot.replace('_','-')),dict(input=slot)])
        recipe=dict(schema_version=1,format='external_recipe_v1',id='viscosity-calibration-01-'+mode,name='Viscosity calibration '+mode,version=1,
            runtime='managed_torch_python_v1',runtime_snapshot_sha256=sha(static['runtime_manifest']),runtime_device='cuda'if mode=='train'else'cpu',
            sources=sources,entrypoint=prefix+'viscosity_calibration_train.py',arguments=arguments,parameters={},inputs=inputs,
            outputs=[dict(path=name,role='evaluation_result'if name=='receipt.json'and mode=='restore'else role,max_bytes=cap,required=True)for name,role,cap in declarations],
            environment={},resources=dict(wall_seconds=1800 if mode=='train'else 600,max_output_bytes=16*1024**2,cpu_cores=1,memory_bytes=4*1024**3))
        write(output/(mode+'-recipe.json'),recipe)
        recipes.append(dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=sha(output/(mode+'-recipe.json'))))
    document=dict(schema_version=2,id=uuid.uuid4().hex,name='Learned viscosity calibration 01',
        data=[dict(id=slot,label=slot,path=str(path.relative_to(ROOT)),sha256=sha(path))for slot,path in static.items()],
        recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(output/'project.json',document)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'));import projects,recipe_contract
    projects.manifest(document)
    for mode in ('train','restore'):recipe_contract.decode_definition((output/(mode+'-recipe.json')).read_bytes())
    write(output/'assembly.json',dict(status='captured',project=prefix+'project.json',project_sha256=sha(output/'project.json'),sources=sources,
        static_inputs={slot:dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)for slot,path in static.items()},
        restore_artifact_map=restore_map,config=config,registration_sha256=sha(DOCS/'viscosity-calibration-registration-01.json'),
        cpu_qualification=str((BASE/'viscosity-calibration-cpu-qualification-01.json').relative_to(ROOT)),
        cuda_qualification=str((BASE/'viscosity-calibration-cuda-qualification-01.json').relative_to(ROOT)),source_sha256=sha(__file__),optimizer_updates=0))
    print('Captured managed viscosity calibration Project',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
