"""Capture one bounded managed CUDA sampler-head fit and typed CPU restore."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT/'runs/full-ai-tight-targets-v1'
DOCS = ROOT/'docs/plans/full-ai-tight-targets-v1'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(args):
    args.output=args.output.resolve()
    if args.output.exists(): raise ValueError('Fresh captured component Project required')
    reg_path = DOCS/'sampler-fit-registration-02.json'; reg = read(reg_path)
    for path, expected in reg['source_sha256'].items():
        if sha(ROOT/path) != expected: raise ValueError('Registered component implementation changed: '+path)
    for device in ('cpu','cuda'):
        for kind in ('constructor','objective'):
            spec = reg['qualification'][kind][device]; path = ROOT/spec['path']; qualified = read(path)
            if sha(path)!=spec['sha256'] or qualified['status']!='passed' or qualified['optimizer_updates']!=0:
                raise ValueError('Actual CPU/CUDA source-bound zero-update qualification required')
    view = BASE/'recurrent-sampler-training-view-01'; manifest = read(view/'manifest.json')
    if manifest['status']!='verified_train_only_sampler_view' or manifest['development_or_reserved_used']:
        raise ValueError('Verified original native double training view required')
    config_path = BASE/'sampler-inputs-01/config.json'; config = read(config_path)
    if sha(config_path)!=reg['config_sha256'] or config['updates']!=440 or config['role']!='recurrent_sampler_transport_fit':
        raise ValueError('Exact finite component learning contract required')
    args.output.mkdir(parents=True)
    files = ('learned_sampler_head.py','train_sampler_head.py')
    for name in files: shutil.copyfile(HERE/name,args.output/name)
    shutil.copyfile(ROOT/'experiments/full-ai-next-priorities-v1/managed_fit_artifacts.py',args.output/'managed_fit_artifacts.py')
    shutil.copyfile(ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01/custody-validator.py',args.output/'custody_validator.py')
    shutil.copyfile(config_path,args.output/'config.json')
    prefix = str(args.output.relative_to(ROOT))+'/'
    sources = [dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for path in sorted(args.output.iterdir()) if path.is_file()]
    static = {'manifest':view/'manifest.json','normalization':view/'normalization.json',
              'runtime_manifest':ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json',
              **{f'part_{i:02d}':view/row['path'] for i,row in enumerate(manifest['shards'])}}
    data = [dict(id=key,label=key,path=str(path.relative_to(ROOT)),sha256=sha(path)) for key,path in static.items()]
    declarations = [('weights.npz','checkpoint',2*1024**2),('initial-weights.npz','checkpoint',2*1024**2),
        ('normalization.json','preprocessing',1024**2),('history.json','training_diagnostics',4*1024**2),
        ('probe.npy','predictions',8*1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map = {'trained_weights':('weights.npz',2*1024**2),'trained_initial':('initial-weights.npz',2*1024**2),
        'trained_history':('history.json',4*1024**2),'trained_probe':('probe.npy',8*1024**2),'trained_receipt':('receipt.json',2*1024**2)}
    recipes=[]
    for mode in ('train','restore'):
        inputs={key:dict(max_bytes=24*1024**2 if key.startswith('part_') else 16*1024**2 if key=='runtime_manifest' else 4*1024**2) for key in static}
        if any(path.stat().st_size>inputs[key]['max_bytes'] for key,path in static.items()): raise ValueError('Typed input cap exceeded')
        arguments=[dict(literal=mode),dict(literal='--config'),dict(source=prefix+'config.json'),
            dict(literal='--manifest'),dict(input='manifest'),dict(literal='--normalization'),dict(input='normalization'),
            dict(literal='--runtime-manifest'),dict(input='runtime_manifest'),dict(literal='--input-evidence'),dict(input_evidence=True),
            dict(literal='--output-dir'),dict(output_dir=True),dict(literal='--shards'),
            *[dict(input=key) for key in static if key.startswith('part_')]]
        if mode=='restore':
            for key,(name,cap) in restore_map.items():
                inputs[key]=dict(max_bytes=cap);arguments.extend([dict(literal='--'+key.replace('_','-')),dict(input=key)])
        recipe=dict(schema_version=1,format='external_recipe_v1',id=config['identifier']+'-'+mode,name=config['identifier']+' '+mode,
            version=1,runtime='managed_torch_python_v1',runtime_snapshot_sha256=sha(static['runtime_manifest']),
            runtime_device='cuda' if mode=='train' else 'cpu',sources=sources,entrypoint=prefix+'train_sampler_head.py',
            arguments=arguments,parameters={},inputs=inputs,outputs=[dict(path=name,role='evaluation_result' if name=='receipt.json' and mode=='restore' else role,max_bytes=cap,required=True) for name,role,cap in declarations],
            environment={},resources=dict(wall_seconds=900 if mode=='train' else 300,max_output_bytes=32*1024**2,cpu_cores=1,memory_bytes=8*1024**3))
        write(args.output/(mode+'-recipe.json'),recipe)
        recipes.append(dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=sha(args.output/(mode+'-recipe.json'))))
    project=dict(schema_version=2,id=uuid.uuid4().hex,name='Full AI learned recurrent sampler transport head 01',data=data,recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(args.output/'project.json',project)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'))
    import projects,recipe_contract
    projects.manifest(project)
    for mode in ('train','restore'): recipe_contract.decode_definition((args.output/(mode+'-recipe.json')).read_bytes())
    write(args.output/'assembly.json',dict(status='captured',project=prefix+'project.json',project_sha256=sha(args.output/'project.json'),sources=sources,
        static_inputs={key:dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size) for key,path in static.items()},
        restore_artifact_map={key:value[0] for key,value in restore_map.items()},config=config,registration_sha256=sha(reg_path),
        qualification=reg['qualification'],source_sha256=sha(__file__),no_analytic_inverse_at_candidate_inference=True,
        optimizer_updates_executed=0,causal_full_state_generator=False))
    print('Captured one 440-update CUDA transport-head fit and exact double CPU restore Project',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
