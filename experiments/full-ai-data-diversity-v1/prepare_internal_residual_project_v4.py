"""Capture compact-source full-objective CUDA fit and metadata-only CPUrestore."""
import argparse,json,shutil,sys,uuid
from pathlib import Path
from managed_input_stream_v1 import streamed_identity
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return streamed_identity(p)['sha256']
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def run(args):
    if args.output.exists():raise ValueError('Fresh compact captured Project required')
    reg=read(args.registration);config=read(ROOT/reg['config'])
    for p,h in reg['source_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Registered source changed: '+p)
    if sha(ROOT/reg['config'])!=reg['config_sha256']:raise ValueError('Registered config differs')
    args.output=args.output.resolve();args.output.mkdir(parents=True)
    names=['internal_residual_network_v1.py','internal_residual_network_v2.py','internal_residual_dataset_v1.py','internal_residual_loss_v1.py',
        'internal_residual_cached_view_v1.py','compact_internal_training_view_v2.py','managed_input_stream_v1.py','capture_memory_telemetry_v1.py','train_internal_residual_v4.py']
    for name in names:shutil.copyfile(HERE/name,args.output/name)
    shutil.copyfile(ROOT/'experiments/full-ai-next-priorities-v1/managed_fit_artifacts.py',args.output/'managed_fit_artifacts.py')
    shutil.copyfile(ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01/custody-validator.py',args.output/'custody_validator.py')
    shutil.copyfile(ROOT/reg['teacher_code'],args.output/'teacher-code.zip');shutil.copyfile(ROOT/reg['config'],args.output/'config.json')
    prefix=str(args.output.relative_to(ROOT))+'/'
    sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(args.output.iterdir())]
    static={n:ROOT/p for n,p in reg['static_inputs'].items()}
    declarations=[('weights.npz','checkpoint',128*1024),('initial-weights.npz','checkpoint',128*1024),('normalization.json','preprocessing',1024**2),
        ('optimization-normalization.json','preprocessing',1024**2),('history.json','training_diagnostics',4*1024**2),('probe.npy','predictions',8*1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map=dict(trained_weights=('weights.npz',128*1024),trained_initial=('initial-weights.npz',128*1024),trained_history=('history.json',4*1024**2),trained_probe=('probe.npy',8*1024**2),trained_receipt=('receipt.json',2*1024**2))
    recipes=[]
    for mode in ('train','restore'):
        inputs={n:dict(max_bytes=1024**3 if n=='part_00' else 16*1024**2) for n in static if mode=='train' or n!='part_00'}
        arguments=[dict(literal=mode),dict(literal='--config'),dict(source=prefix+'config.json'),dict(literal='--teacher-code'),dict(source=prefix+'teacher-code.zip')]
        for flag in ('manifest','normalization','optimization_normalization','runtime_manifest'):arguments.extend([dict(literal='--'+flag.replace('_','-')),dict(input=flag)])
        arguments.extend([dict(literal='--input-evidence'),dict(input_evidence=True),dict(literal='--output-dir'),dict(output_dir=True)])
        if mode=='train':arguments.extend([dict(literal='--shards'),dict(input='part_00')])
        else:
            for n,(_,cap) in restore_map.items():inputs[n]=dict(max_bytes=cap);arguments.extend([dict(literal='--'+n.replace('_','-')),dict(input=n)])
        outputs=[dict(path=n,role='evaluation_result' if mode=='restore' and n=='receipt.json' else role,max_bytes=cap,required=True) for n,role,cap in declarations]
        if mode=='train':outputs.append(dict(path='last-evaluated-weights.npz',role='checkpoint',max_bytes=128*1024,required=True))
        definition=dict(schema_version=1,format='external_recipe_v1',id=config['identifier']+'-'+mode,name=config['identifier']+' '+mode,version=1,
            runtime='managed_torch_python_v1',runtime_snapshot_sha256=sha(static['runtime_manifest']),runtime_device='cuda' if mode=='train' else 'cpu',sources=sources,
            entrypoint=prefix+'train_internal_residual_v4.py',arguments=arguments,parameters={},inputs=inputs,outputs=outputs,environment={},
            resources=dict(wall_seconds=config['wall_seconds'] if mode=='train' else 300,max_output_bytes=24*1024**2,cpu_cores=1,memory_bytes=4*1024**3 if mode=='train' else 1024**3))
        if mode=='train':definition['capture_policy']=reg['capture_policy']
        write(args.output/(mode+'-recipe.json'),definition);recipes.append(dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=sha(args.output/(mode+'-recipe.json'))))
    project=dict(schema_version=2,id=uuid.uuid4().hex,name='Full-AI direct residual supervision compact CUDA fit',data=[dict(id=n,label=n,path=str(p.relative_to(ROOT)),sha256=sha(p)) for n,p in static.items()],recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(args.output/'project.json',project)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok-large-input-v3'))
    import projects,recipe_contract
    projects.manifest(project)
    for mode in ('train','restore'):recipe_contract.decode_definition((args.output/(mode+'-recipe.json')).read_bytes())
    write(args.output/'assembly.json',dict(status='captured_compact_full_objective_warm_fit_and_CPUrestore',project=prefix+'project.json',project_sha256=sha(args.output/'project.json'),sources=sources,
        static_inputs={n:dict(path=str(p.relative_to(ROOT)),**streamed_identity(p)) for n,p in static.items()},restore_artifact_map={n:v[0] for n,v in restore_map.items()},config=config,
        registration=str(args.registration.resolve().relative_to(ROOT)),registration_sha256=sha(args.registration),source_sha256=sha(__file__),optimizer_updates_executed=0,cpu_model_forward_allowed=False))
    print('Captured compact CUDA warm-start fit/metadata-only CPUrestore; no optimizer execution.',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
