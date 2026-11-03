"""Capture the registered causal request fit and independent CPU restore Project."""
import argparse
from pathlib import Path
import shutil
import sys
import uuid
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha

FILES=('pressure_request_model.py','pressure_request_objective.py','pressure_request_data.py',
       'pressure_request_train.py','managed_fit_artifacts.py','temporal_data.py',
       'temporal_features_v2.py','temporal_codec.py')


def prepare(args):
    identifier='b1-causal-request';output=BASE/f'{identifier}-project-{args.generation:02d}'
    if output.exists():raise ValueError('Fresh causal request Project required')
    registered=read(DOCS/'causal-pressure-request-component-registration.json')
    for name,digest in registered['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Qualified component source changed: '+name)
    view=BASE/'temporal-request-training-view-01';manifest=read(view/'manifest.json')
    if sha(view/'manifest.json')!=registered['data_manifest_sha256']or manifest['development_or_reserved_used']:
        raise ValueError('Qualified training-only component data required')
    output.mkdir(parents=True)
    for name in FILES:shutil.copyfile(HERE/name,output/name)
    shutil.copyfile(ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01/custody-validator.py',output/'custody_validator.py')
    config=dict(schema='rheon-causal-request-fit-v1',identifier=identifier,horizons=1,updates=registered['updates'],
        learning_rate=registered['learning_rate'],schedule=registered['schedule'],
        seed=registered['initialization_seed'],sampling_seed=registered['sampling_seed'],
        gradient_norm_cap=registered['gradient_norm_cap'],wall_seconds=1800,
        component_only=True,future_request_input=False,current_pressure_input=False,
        physical_context=dict(dt_s=1/60,rho_kg_m3=1000.,mu_pa_s=.001,boundaries='closed regular grid'),
        registration_sha256=sha(DOCS/'causal-pressure-request-component-registration.json'),
        selection='Final update10000; checkpoints0/2000/10000retained; no development checkpoint reselection',
        analytical_inverse='Offline objective only; no inverse in model inference')
    write(output/'config.json',config)
    prefix=str(output.relative_to(ROOT))+'/'
    sources=[dict(path=prefix+name,sha256=sha(output/name))for name in (*FILES,'custody_validator.py','config.json')]
    static={'manifest':view/'manifest.json','normalization':view/'normalization.json',
        'runtime_manifest':ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json',
        **{f'part_{i:02d}':view/row['path']for i,row in enumerate(manifest['shards'])}}
    common=[('weights.npz','checkpoint',2*1024**2),('initial-weights.npz','checkpoint',2*1024**2),
        ('normalization.json','preprocessing',1024**2),('history.json','training_diagnostics',4*1024**2),
        ('probe.npy','predictions',8*1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map={'trained_weights':('weights.npz',2*1024**2),'trained_initial':('initial-weights.npz',2*1024**2),
        'trained_history':('history.json',4*1024**2),'trained_probe':('probe.npy',8*1024**2),
        'trained_receipt':('receipt.json',2*1024**2)}
    recipes=[]
    for mode in ('train','restore'):
        inputs={slot:dict(max_bytes=24*1024**2 if slot.startswith('part_')else 16*1024**2 if slot=='runtime_manifest'else 1024**2)for slot in static}
        arguments=[dict(literal=mode),dict(literal='--config'),dict(source=prefix+'config.json'),
            dict(literal='--manifest'),dict(input='manifest'),dict(literal='--normalization'),dict(input='normalization'),
            dict(literal='--runtime-manifest'),dict(input='runtime_manifest'),dict(literal='--input-evidence'),dict(input_evidence=True),
            dict(literal='--output-dir'),dict(output_dir=True),dict(literal='--shards'),
            *[dict(input=slot)for slot in static if slot.startswith('part_')]]
        if mode=='restore':
            for slot,(name,cap)in restore_map.items():
                inputs[slot]=dict(max_bytes=cap);arguments.extend([dict(literal='--'+slot.replace('_','-')),dict(input=slot)])
        declarations=common+[(f'checkpoint-{update:03d}.npz','checkpoint',2*1024**2)for update in (0,2000,10000)]if mode=='train'else common
        recipe=dict(schema_version=1,format='external_recipe_v1',id=identifier+'-'+mode,name=identifier+' '+mode,version=1,
            runtime='managed_torch_python_v1',runtime_snapshot_sha256=sha(static['runtime_manifest']),
            runtime_device='cuda'if mode=='train'else'cpu',sources=sources,entrypoint=prefix+'pressure_request_train.py',
            arguments=arguments,parameters={},inputs=inputs,
            outputs=[dict(path=name,role='evaluation_result'if name=='receipt.json'and mode=='restore'else role,max_bytes=cap,required=True)for name,role,cap in declarations],
            environment={},resources=dict(wall_seconds=1800 if mode=='train'else 600,max_output_bytes=32*1024**2,cpu_cores=1,memory_bytes=4*1024**3))
        write(output/(mode+'-recipe.json'),recipe)
        recipes.append(dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=sha(output/(mode+'-recipe.json'))))
    document=dict(schema_version=2,id=uuid.uuid4().hex,name='Full AI causal pressure request component',
        data=[dict(id=slot,label=slot,path=str(path.relative_to(ROOT)),sha256=sha(path))for slot,path in static.items()],
        recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(output/'project.json',document)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'));import projects,recipe_contract
    projects.manifest(document)
    for mode in ('train','restore'):recipe_contract.decode_definition((output/(mode+'-recipe.json')).read_bytes())
    total=sum(path.stat().st_size for path in static.values())+sum((ROOT/row['path']).stat().st_size for row in sources)+20*1024**2
    if total>256*1024**2:raise ValueError('Captured component inputs plus restore reserve exceed managed limit')
    write(output/'assembly.json',dict(status='captured',project=prefix+'project.json',project_sha256=sha(output/'project.json'),
        sources=sources,static_inputs={slot:dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)for slot,path in static.items()},
        restore_artifact_map={slot:spec[0]for slot,spec in restore_map.items()},config=config,
        input_plus_source_and_restore_reserve_bytes=total,source_sha256=sha(__file__),
        cuda_qualification=f'runs/full-ai-next-priorities-v1/causal-request-trainer-cuda-qualification-{args.generation:02d}.json',
        model_file='pressure_request_model.py',objective_file='pressure_request_objective.py',trainer_file='pressure_request_train.py',
        component_only=True,full_state_temporal_candidate=False,optimizer_updates_executed=0))
    print('Captured causal request Project',identifier,'managed input reserve',total,'bytes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',type=int,default=1)
    prepare(parser.parse_args())
