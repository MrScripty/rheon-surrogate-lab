"""Capture a bounded managed CUDA shared-component fit and CPU weight restore."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def run(args):
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh captured shared-component Project required')
    reg=read(args.registration)
    for path,expected in reg['source_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Registered managed source changed: '+path)
    qualified=read(ROOT/reg['cuda_qualification']['path'])
    if sha(ROOT/reg['cuda_qualification']['path'])!=reg['cuda_qualification']['sha256'] or qualified['status']!='passed' or qualified['optimizer_updates']!=0 or not qualified['all148training_scenes_in_objective']:
        raise ValueError('Actual zero-update all148scene CUDA qualification required')
    model_path=HERE/'shared_spectral_model_v1.py'
    if qualified['source_sha256'][str(model_path.relative_to(ROOT))]!=sha(model_path):raise ValueError('Actual qualified model/objective source differs')
    view=ROOT/reg['view'];manifest=read(view/'manifest.json');config_path=ROOT/reg['config'];config=read(config_path)
    if sha(config_path)!=reg['config_sha256'] or config['optimizer_calls']!=160 or config['role']!='shape_shared_spectral_pressure_component':raise ValueError('Exact finite component config required')
    if manifest['training_scenes']!=148 or manifest['development_or_reserved_used']:raise ValueError('All148train-only view required')
    args.output.mkdir(parents=True)
    for name in ('shared_spectral_model_v1.py','train_shared_spectral_v1.py'):shutil.copyfile(HERE/name,args.output/name)
    shutil.copyfile(ROOT/'experiments/full-ai-next-priorities-v1/managed_fit_artifacts.py',args.output/'managed_fit_artifacts.py')
    shutil.copyfile(ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01/custody-validator.py',args.output/'custody_validator.py')
    shutil.copyfile(config_path,args.output/'config.json')
    prefix=str(args.output.relative_to(ROOT))+'/'
    sources=[dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for path in sorted(args.output.iterdir())]
    static=dict(manifest=view/'manifest.json',normalization=view/'normalization.json',runtime_manifest=ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json',part_00=view/manifest['parts'][0]['path'])
    declarations=[('weights.npz','checkpoint',128*1024),('initial-weights.npz','checkpoint',128*1024),('normalization.json','preprocessing',1024**2),('history.json','training_diagnostics',2*1024**2),('probe.npy','predictions',8*1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map={key:(name,cap) for key,name,cap in [('trained_weights','weights.npz',128*1024),('trained_initial','initial-weights.npz',128*1024),('trained_history','history.json',2*1024**2),('trained_probe','probe.npy',8*1024**2),('trained_receipt','receipt.json',2*1024**2)]}
    recipes=[]
    for mode in ('train','restore'):
        inputs={key:dict(max_bytes=16*1024**2 if key in ('part_00','runtime_manifest') else 4*1024**2) for key in static}
        if any(path.stat().st_size>inputs[key]['max_bytes'] for key,path in static.items()):raise ValueError('Typed input cap exceeded')
        arguments=[dict(literal=mode),dict(literal='--config'),dict(source=prefix+'config.json'),dict(literal='--manifest'),dict(input='manifest'),dict(literal='--normalization'),dict(input='normalization'),dict(literal='--runtime-manifest'),dict(input='runtime_manifest'),dict(literal='--input-evidence'),dict(input_evidence=True),dict(literal='--output-dir'),dict(output_dir=True),dict(literal='--shards'),dict(input='part_00')]
        if mode=='restore':
            for key,(name,cap) in restore_map.items():
                inputs[key]=dict(max_bytes=cap);arguments.extend([dict(literal='--'+key.replace('_','-')),dict(input=key)])
        definition=dict(schema_version=1,format='external_recipe_v1',id=config['identifier']+'-'+mode,name=config['identifier']+' '+mode,version=1,
            runtime='managed_torch_python_v1',runtime_snapshot_sha256=sha(static['runtime_manifest']),runtime_device='cuda' if mode=='train' else 'cpu',
            sources=sources,entrypoint=prefix+'train_shared_spectral_v1.py',arguments=arguments,parameters={},inputs=inputs,
            outputs=[dict(path=name,role='evaluation_result' if mode=='restore' and name=='receipt.json' else role,max_bytes=cap,required=True) for name,role,cap in declarations],
            environment={},resources=dict(wall_seconds=600 if mode=='train' else 300,max_output_bytes=24*1024**2,cpu_cores=1,memory_bytes=4*1024**3))
        write(args.output/(mode+'-recipe.json'),definition)
        recipes.append(dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=sha(args.output/(mode+'-recipe.json'))))
    project=dict(schema_version=2,id=uuid.uuid4().hex,name='Expanded shape-shared learned spectral component01',data=[dict(id=key,label=key,path=str(path.relative_to(ROOT)),sha256=sha(path)) for key,path in static.items()],recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(args.output/'project.json',project)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'))
    import projects,recipe_contract
    projects.manifest(project)
    for mode in ('train','restore'):recipe_contract.decode_definition((args.output/(mode+'-recipe.json')).read_bytes())
    write(args.output/'assembly.json',dict(status='captured',project=prefix+'project.json',project_sha256=sha(args.output/'project.json'),sources=sources,
        static_inputs={key:dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size) for key,path in static.items()},
        restore_artifact_map={key:value[0] for key,value in restore_map.items()},config=config,registration=str(args.registration.resolve().relative_to(ROOT)),registration_sha256=sha(args.registration),
        cuda_qualification=reg['cuda_qualification'],source_sha256=sha(__file__),optimizer_updates_executed=0,cpu_model_forward_allowed=False))
    print('Captured one160-call managed CUDA component fit and exact typed CPU weight restore without CPU inference.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())
