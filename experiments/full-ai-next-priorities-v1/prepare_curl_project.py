"""Capture a generic managed Torch temporal fit and exact CPU restore Project."""
import argparse
from pathlib import Path
import shutil
import sys
import uuid
from study_support import ROOT,HERE,BASE,DOCS,read,write,sha

FILES=('bundle_model_curl.py','bundle_model_step_scaled.py','bundle_model_sharing.py','learned_global.py','temporal_features.py','temporal_features_v2.py',
       'temporal_codec.py','temporal_data.py','temporal_objective.py','temporal_feedback.py','temporal_train_curl.py','temporal_objective_increment.py','temporal_objective_mse.py')

def prepare(args):
    if args.architecture not in ('endpoint','reuse')or args.horizons!=2 or args.updates!=10000 or args.single_block_updates!=2000:
        raise ValueError('Only registered full-state sharing comparisons may be captured')
    if args.supervision_mode!='increments'or args.architecture!='endpoint'or args.identifier!='b2-endpoint-curl-increment':raise ValueError('Registered curl-readout control required')
    output=BASE/(args.identifier+f'-project-{args.project_generation:02d}')
    if output.exists():raise ValueError('Fresh temporal Project required')
    view=BASE/'temporal-training-view-02';manifest=read(view/'manifest.json')
    if manifest['status']!='verified'or manifest['development_or_reserved_used']:
        raise ValueError('Verified train-only temporal view required')
    output.mkdir(parents=True)
    for name in FILES:shutil.copyfile(HERE/name,output/name)
    shutil.copyfile(ROOT/'runs/pressure-methods-v4/v4-student50-control-project-01/custody-validator.py',output/'custody_validator.py')
    config=dict(schema='rheon-temporal-fit-curl-readout-v1',velocity_representation='learned_edge_potential_fixed_curl',supervision_mode=args.supervision_mode,architecture=args.architecture,feedback_protocol=2,identifier=args.identifier,horizons=args.horizons,width=12,
        updates=args.updates,single_block_updates=args.single_block_updates,learning_rate=args.learning_rate,
        schedule=[],seed=args.seed,sampling_seed=2026101922,wall_seconds=1800,
        physical_context=dict(dt_s=1/60,rho_kg_m3=1000.,mu_pa_s=.001,boundaries='closed regular grid'),
        initialization='Seeded compact spatial encoder and zero future heads',
        inference='One shared encoder; joint state outputs; learned pressure modes; no numerical pressure solve',
        sample_step_range=[0,120],loss='Equal horizon weights; pressure/velocity/tracer, equation, divergence, mass, energy and bounds',
        track_b_registration_sha256=sha(DOCS/'registration.json'))
    registered=read(DOCS/'temporal-training-registration-v2.json')
    for key in ('updates','single_block_updates','learning_rate','seed','sampling_seed','schedule'):
        config[key]=registered[key]
    config['sharing_registration_sha256']=sha(DOCS/'temporal-curl-registration-01.json')
    config['step_scale_statistics_sha256']=sha(BASE/'temporal-step-scale-statistics-01.json')
    config['original_normalization_sha256']=sha(view/'normalization.json')
    shutil.copyfile(BASE/'temporal-step-scale-statistics-01.json',output/'step-scales.json')
    config['initialization']='Seeded width12 shared global encoder; zero fresh-pressure output heads; previous pressure excluded'
    config['inference']='One shared encoder; predicted-endpoint intermediate decoder or retained-feature latent updates; no numerical solve'
    config['full_fit_registration_sha256']=sha(DOCS/'temporal-training-registration-v2.json')
    if not 0<=args.single_block_updates<args.updates:raise ValueError('Actual own-history phase required')
    config['loss']='Matched MSE pressure/equation/physical terms; '+('relative full-state velocity/tracer errors'if args.supervision_mode=='full_state'else'velocity/tracer errors in fixed aggregate native-step units')
    write(output/'config.json',config)
    prefix=str(output.relative_to(ROOT))+'/'
    sources=[dict(path=prefix+name,sha256=sha(output/name))for name in (*FILES,'custody_validator.py','config.json','step-scales.json')]
    static={'manifest':view/'manifest.json','normalization':view/'normalization.json',
            'runtime_manifest':ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json',
            **{f'part_{i:02d}':view/row['path']for i,row in enumerate(manifest['shards'])}}
    data=[dict(id=slot,label=slot,path=str(path.relative_to(ROOT)),sha256=sha(path))for slot,path in static.items()]
    recipes=[]
    declarations=[('weights.npz','checkpoint',2*1024**2),('initial-weights.npz','checkpoint',2*1024**2),
                  ('normalization.json','preprocessing',1024**2),('history.json','training_diagnostics',4*1024**2),
                  ('probe.npy','predictions',8*1024**2),('receipt.json','training_result',2*1024**2)]
    restore_map={'trained_weights':('weights.npz',2*1024**2),'trained_initial':('initial-weights.npz',2*1024**2),
                 'trained_history':('history.json',4*1024**2),'trained_probe':('probe.npy',8*1024**2),
                 'trained_receipt':('receipt.json',2*1024**2)}
    for mode in ('train','restore'):
        inputs={slot:dict(max_bytes=24*1024**2 if slot.startswith('part_')else 16*1024**2 if slot=='runtime_manifest'else 1024**2)for slot in static}
        if any(path.stat().st_size>inputs[slot]['max_bytes']for slot,path in static.items()):raise ValueError('Static input exceeds declared material cap')
        arguments=[dict(literal=mode),dict(literal='--config'),dict(source=prefix+'config.json'),
            dict(literal='--manifest'),dict(input='manifest'),dict(literal='--normalization'),dict(input='normalization'),
            dict(literal='--runtime-manifest'),dict(input='runtime_manifest'),
            dict(literal='--input-evidence'),dict(input_evidence=True),dict(literal='--output-dir'),dict(output_dir=True),
            dict(literal='--shards'),*[dict(input=slot)for slot in static if slot.startswith('part_')]]
        if mode=='restore':
            for slot,(name,cap)in restore_map.items():
                inputs[slot]=dict(max_bytes=cap);arguments.extend([dict(literal='--'+slot.replace('_','-')),dict(input=slot)])
        recipe=dict(schema_version=1,format='external_recipe_v1',id=args.identifier+'-'+mode,
            name=args.identifier+' '+mode,version=1,runtime='managed_torch_python_v1',
            runtime_snapshot_sha256=sha(static['runtime_manifest']),runtime_device='cuda'if mode=='train'else'cpu',
            sources=sources,entrypoint=prefix+'temporal_train_curl.py',arguments=arguments,parameters={},inputs=inputs,
            outputs=[dict(path=name,role='evaluation_result'if name=='receipt.json'and mode=='restore'else role,max_bytes=cap,required=True)for name,role,cap in declarations],
            environment={},resources=dict(wall_seconds=1800 if mode=='train'else 600,max_output_bytes=32*1024**2,cpu_cores=1,memory_bytes=4*1024**3))
        write(output/(mode+'-recipe.json'),recipe)
        recipes.append(dict(id=mode,label=mode,path=prefix+mode+'-recipe.json',sha256=sha(output/(mode+'-recipe.json'))))
    project=dict(schema_version=2,id=uuid.uuid4().hex,name='Full AI temporal '+args.identifier,data=data,
        recipes=recipes,evidence=[],requirements=[],extensions=[])
    write(output/'project.json',project)
    sys.path.insert(0,str(ROOT/'runs/dependency-migration-v1/Tuldok'))
    import projects,recipe_contract
    projects.manifest(project)
    for mode in ('train','restore'):recipe_contract.decode_definition((output/(mode+'-recipe.json')).read_bytes())
    write(output/'assembly.json',dict(status='captured',project=str((output/'project.json').relative_to(ROOT)),
        project_sha256=sha(output/'project.json'),sources=sources,static_inputs={k:dict(path=str(v.relative_to(ROOT)),sha256=sha(v),bytes=v.stat().st_size)for k,v in static.items()},
        restore_artifact_map={k:v[0]for k,v in restore_map.items()},config=config,
        input_bytes=sum(v.stat().st_size for v in static.values()),source_sha256=sha(__file__),
        no_numerical_solver_at_candidate_inference=True,optimizer_updates_executed=0,
        cuda_qualification='runs/full-ai-next-priorities-v1/temporal-curl-trainer-cuda-qualification-01.json',
        cpu_qualification='runs/full-ai-next-priorities-v1/temporal-curl-trainer-cpu-qualification-01.json',
        model_file='bundle_model_curl.py',trainer_file='temporal_train_curl.py'))
    print('Captured generic managed temporal Project',args.identifier,'K',args.horizons,'updates',args.updates,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision-mode',choices=('full_state','increments'),required=True)
    parser.add_argument('--architecture',choices=('endpoint','reuse'),required=True)
    parser.add_argument('--identifier',required=True);parser.add_argument('--horizons',type=int,required=True)
    parser.add_argument('--updates',type=int,required=True);parser.add_argument('--single-block-updates',type=int,required=True)
    parser.add_argument('--learning-rate',type=float,default=.001);parser.add_argument('--seed',type=int,default=2026101921)
    parser.add_argument('--project-generation',type=int,default=1)
    prepare(parser.parse_args())
