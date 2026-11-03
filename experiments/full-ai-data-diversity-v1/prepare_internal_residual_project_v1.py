"""Capture one all148scene CUDA fit and metadata-only exact CPU restore."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import uuid

from managed_input_stream_v1 import streamed_identity

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def sha(path):return streamed_identity(path)['sha256']
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def run(args):
    if args.output.exists():raise ValueError('Fresh captured Project required')
    reg=read(args.registration);config=read(ROOT/reg['config']);qualified=read(ROOT/reg['cuda_qualification']['path'])
    for name,expected in reg['source_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Registered source changed: '+name)
    if (sha(ROOT/reg['config'])!=reg['config_sha256'] or sha(ROOT/reg['cuda_qualification']['path'])!=reg['cuda_qualification']['sha256']
            or qualified['status']!='passed_activity_preserving_normalized_all148scene_CUDA_objective_and_CPUrestore'
            or qualified['optimizer_updates']!=0 or not qualified['all148training_scenes_in_objective']
            or config['qualified_initial_inventory']!=qualified['parameter_inventory']):
        raise ValueError('Actual qualified architecture/objective/initial state required')
    args.output=args.output.resolve();args.output.mkdir(parents=True)
    names=['internal_residual_network_v1.py','internal_residual_network_v2.py','internal_residual_dataset_v1.py',
           'internal_residual_loss_v1.py','managed_input_stream_v1.py','capture_memory_telemetry_v1.py','train_internal_residual_v1.py']
    for name in names:shutil.copyfile(HERE/name,args.output/name)
    shutil.copyfile(ROOT/'experiments/full-ai-next-priorities-v1/managed_fit_artifacts.py',args.output/'managed_fit_artifacts.py')
    shutil.copyfile(ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01/custody-validator.py',args.output/'custody_validator.py')
    shutil.copyfile(ROOT/reg['config'],args.output/'config.json')
    prefix=str(args.output.relative_to(ROOT))+'/'
    sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(args.output.iterdir())]
    view=ROOT/reg['view'];manifest=read(view/'manifest.json')
    static=dict(manifest=view/'manifest.json',normalization=view/'network-normalization.json',
        optimization_normalization=view/'optimization-normalization.json',
        runtime_manifest=ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json',
        part_00=view/manifest['archive']['path'])
    declarations=[('weights.npz','checkpoint',128*1024),('initial-weights.npz','checkpoint',128*1024),
        ('normalization.json','preprocessing',1024**2),('optimization-normalization.json','preprocessing',1024**2),
        ('history.json','training_diagnostics',4*1024**2),('probe.npy','predictions',8*1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map=dict(trained_weights=('weights.npz',128*1024),trained_initial=('initial-weights.npz',128*1024),
                     trained_history=('history.json',4*1024**2),trained_probe=('probe.npy',8*1024**2),trained_receipt=('receipt.json',2*1024**2))
    recipes=[]
    for mode in ('train','restore'):
        inputs={n:dict(max_bytes=10*1024**3 if n=='part_00' else 16*1024**2) for n in static if mode=='train' or n!='part_00'}
        arguments=[dict(literal=mode),dict(literal='--config'),dict(source=prefix+'config.json')]
        for flag in ('manifest','normalization','optimization_normalization','runtime_manifest'):
            arguments.extend([dict(literal='--'+flag.replace('_','-')),dict(input=flag)])
        arguments.extend([dict(literal='--input-evidence'),dict(input_evidence=True),dict(literal='--output-dir'),dict(output_dir=True)])
        if mode=='train':arguments.extend([dict(literal='--shards'),dict(input='part_00')])
        else:
            for n,(_artifact,cap) in restore_map.items():
                inputs[n]=dict(max_bytes=cap);arguments.extend([dict(literal='--'+n.replace('_','-')),dict(input=n)])
        definition=dict(schema_version=1,format='external_recipe_v1',id=config['identifier']+'-'+mode,
            name=config['identifier']+' '+mode,version=1,runtime='managed_torch_python_v1',
            runtime_snapshot_sha256=sha(static['runtime_manifest']),runtime_device='cuda' if mode=='train' else 'cpu',
            sources=sources,entrypoint=prefix+'train_internal_residual_v1.py',arguments=arguments,parameters={},inputs=inputs,
            outputs=[dict(path=n,role='evaluation_result' if mode=='restore' and n=='receipt.json' else role,
                          max_bytes=cap,required=True) for n,role,cap in declarations],environment={},
            resources=dict(wall_seconds=config['wall_seconds'] if mode=='train' else 300,
                           max_output_bytes=24*1024**2,cpu_cores=1,memory_bytes=4*1024**3 if mode=='train' else 1024**3))
        if mode=='train':definition['capture_policy']=reg['capture_policy']
        write(args.output/(mode+'-recipe.json'),definition)
        recipes.append(dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=sha(args.output/(mode+'-recipe.json'))))
    project=dict(schema_version=2,id=uuid.uuid4().hex,name='Full-AI internal width8 residual first fit',
        data=[dict(id=n,label=n,path=str(p.relative_to(ROOT)),sha256=sha(p)) for n,p in static.items()],
        recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(args.output/'project.json',project)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok-large-input-v2'))
    import projects,recipe_contract
    projects.manifest(project)
    for mode in ('train','restore'):recipe_contract.decode_definition((args.output/(mode+'-recipe.json')).read_bytes())
    write(args.output/'assembly.json',dict(status='captured_first_fullgrid_residual_fit_and_metadata_only_CPUrestore',
        project=prefix+'project.json',project_sha256=sha(args.output/'project.json'),sources=sources,
        static_inputs={n:dict(path=str(p.relative_to(ROOT)),**streamed_identity(p)) for n,p in static.items()},
        restore_artifact_map={n:v[0] for n,v in restore_map.items()},config=config,registration=str(args.registration.resolve().relative_to(ROOT)),
        registration_sha256=sha(args.registration),cuda_qualification=reg['cuda_qualification'],
        source_sha256=sha(__file__),optimizer_updates_executed=0,cpu_model_forward_allowed=False))
    print('Captured finite full-grid CUDAfit and exact CPUrestore; no optimizer execution.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
