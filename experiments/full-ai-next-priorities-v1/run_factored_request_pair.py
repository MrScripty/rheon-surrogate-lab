"""Own one ordinary managed CUDA fit and CPU restore with durable mutation journals."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
from study_support import ROOT,HERE,BASE,read,write,sha

def run(args):
    args.output=args.output.resolve();project_dir=args.project.resolve()
    if args.output.exists():raise ValueError('Fresh pair directory required; never replay unknown mutations')
    assembly=read(project_dir/'assembly.json');document=read(project_dir/'project.json')
    if sha(project_dir/'project.json')!=assembly['project_sha256']:raise ValueError('Captured Project changed')
    for row in assembly['sources']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('Captured training source changed')
    qualification=read(ROOT/assembly['cuda_qualification'])
    if qualification['status']!='passed'or qualification['optimizer_updates']!=3 or qualification.get('native_optimizer_updates')!=0:
        raise ValueError('Actual temporal CUDA objective must qualify first')
    if qualification['model_sha256']!=sha(project_dir/assembly['model_file'])or qualification['objective_sha256']!=sha(project_dir/assembly['objective_file']):
        raise ValueError('Qualified temporal source differs')
    if qualification.get('trainer_sha256')!=sha(project_dir/assembly['trainer_file'])or qualification.get('project_sha256')!=assembly['project_sha256']:
        raise ValueError('Actual captured component trainer must be qualified')
    if qualification.get('artifact_helpers_sha256')!=sha(project_dir/'managed_fit_artifacts.py')or qualification.get('data_sha256')!=sha(project_dir/'pressure_request_data.py'):
        raise ValueError('Qualified typed custody or component data source differs')
    if not qualification.get('component_only')or qualification.get('future_request_input')or qualification.get('current_pressure_input'):
        raise ValueError('Qualified causal component contract differs')
    CPU=read(ROOT/assembly['cuda_qualification'].replace('-cuda-','-cpu-'))
    if CPU['status']!='passed'or CPU['project_sha256']!=assembly['project_sha256']or not CPU['CPU_probe']['bitwise_repeat']:
        raise ValueError('Captured component CPU probe must qualify before admission')
    sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),
                  str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api,project_cli
    from qualify_typed_pressure import OwnerHTTP,Journal
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    args.output.mkdir(parents=True);(args.output/'mutations').mkdir()
    journal=Journal(args.output/'mutations');started=time.perf_counter()
    write(args.output/'intent.json',dict(project=str(project_dir.relative_to(ROOT)),assembly_sha256=sha(project_dir/'assembly.json'),
        project_sha256=sha(project_dir/'project.json'),source_sha256=sha(__file__),config=assembly['config'],
        automatic_retry=False,started_unix=time.time()))
    def mutate(phase,operation,body):return journal.mutation(phase,operation,body,lambda:owner.call(operation,body))
    project=mutate('open-project','project_snapshots.open',dict(root='current-lab',manifest_path=assembly['project']))
    if project['id']!=document['id']or project['manifest_sha256']!=assembly['project_sha256']:raise ValueError('Remote Project differs')
    refs={r['id']:r for r in project['data']};checked=[]
    for slot,spec in assembly['static_inputs'].items():
        row=refs[slot];path=ROOT/spec['path']
        if row['status']!='available'or sha(path)!=spec['sha256']or path.stat().st_size!=spec['bytes']:
            raise ValueError('Full input verification failed')
        if row['integrity']=='verified':
            if row['observed_sha256']!=spec['sha256']:raise ValueError('Remote observed hash differs')
        elif row.get('reason')!='Present; content exceeds the bounded hash inspection budget.':
            raise ValueError('Unknown input verification failure')
        checked.append(dict(input=slot,bytes=spec['bytes'],sha256=spec['sha256'],local_full_hash_verified=True,
            api_preview_integrity=row['integrity']))
    write(args.output/'input-verification.json',dict(status='verified',inputs=checked,
        server_material_preflight_still_required=True,preview_statuses_preserved=True))
    key='causal-request-'+hashlib.sha256(str(args.output).encode()).hexdigest()[:24]
    active={};runs={}
    def await_run(run_id,limit):
        deadline=time.monotonic()+limit;previous=None
        while True:
            observed=owner.call('runs.get',dict(run_id=run_id))
            status=observed['state'],observed['custody_phase']
            if status!=previous:
                write(args.output/('state-'+run_id+'.json'),observed)
                print('Managed causal request Run',run_id,*status,flush=True);previous=status
            if observed['state']in ('completed','failed','cancelled','interrupted','unknown'):
                write(args.output/('terminal-'+run_id+'.json'),observed)
                if observed['state']!='completed'or observed['custody_phase']!='drained':
                    raise RuntimeError('Temporal Run did not complete and drain; no automatic retry')
                return observed
            if time.monotonic()>deadline:raise TimeoutError('Observation deadline; inspect same Run without resubmission')
            time.sleep(2)
    for phase in ('train','restore'):
        definition=read(project_dir/(phase+'-recipe.json'))
        reference=next(r for r in project['recipes']if r['id']==phase)
        inspected=owner.call('recipes.inspect',dict(project_id=project['id'],reference_id=phase,expected_sha256=reference['observed_sha256']))
        if inspected['definition']!=definition:raise ValueError('Remote temporal recipe differs')
        bindings={}
        for slot in definition['inputs']:
            if slot in assembly['restore_artifact_map']:
                name=assembly['restore_artifact_map'][slot]
                artifact=next(r for r in runs['train']['artifacts']if r['name']==name)
                bindings[slot]=dict(kind='run_artifact',run_id=runs['train']['id'],artifact_name=name,expected_sha256=artifact['sha256'])
            else:bindings[slot]=dict(reference_id=slot,expected_sha256=assembly['static_inputs'][slot]['sha256'])
        graph=mutate(phase+'-graph','graphs.create_external',dict(name=document['name']+' '+phase,project_id=project['id'],
            recipe_id=phase,expected_sha256=reference['observed_sha256'],inputs=bindings))
        preflight=owner.call('graphs.preflight_external',dict(graph_id=graph['id'],revision=graph['revision']))
        body=dict(graph_id=graph['id'],graph_revision=graph['revision'],expected_digest=preflight['material_sha256'],
            consumer_actor_id=owner.identity['actor_id'],ttl_seconds=3600)
        grant=journal.mutation(phase+'-approve','runs.approve_external',body,
            lambda:owner.raw('POST','/api/workspace/runs/approve-external',body))
        admitted=mutate(phase+'-admit','runs.submit_external',dict(request_key=key+'-'+phase,graph_id=graph['id'],
            graph_revision=graph['revision'],expected_digest=preflight['material_sha256'],approval_id=grant['id']))
        active[phase]=admitted['id'];write(args.output/'active.json',active)
        runs[phase]=await_run(admitted['id'],2100 if phase=='train'else 900)
        class Reader:
            def call(self,operation,arguments):
                if operation!='runs.artifact_chunk':raise ValueError('Artifact-only reader')
                return dict(status=200,result=owner.call(operation,arguments))
        directory=args.output/phase;directory.mkdir()
        for row in runs[phase]['artifacts']:
            project_cli.download(Reader(),runs[phase]['id'],row['name'],row['sha256'],directory/row['name'])
    trained=read(args.output/'train/receipt.json');restored=read(args.output/'restore/receipt.json')
    if trained['mode']!='train'or trained['optimizer_updates']!=assembly['config']['updates']:
        raise ValueError('Actual update count differs')
    if restored['mode']!='restore'or restored['optimizer_updates']!=0 or restored['execution']['cuda_initialized']:
        raise ValueError('Fresh zero-update CPU restore required')
    if restored['originating_training_run']['run_id']!=runs['train']['id']or restored['closure']!=trained['closure']:
        raise ValueError('CPU restore provenance differs')
    for update in (0,2000,10000):
        name=f'checkpoint-{update:03d}.npz'
        if sha(args.output/'train'/name)!=trained['outputs'][name]['sha256']:
            raise ValueError('Managed component checkpoint identity differs')
    if trained.get('main_optimizer_updates')!=10000 or trained.get('linear_optimizer_updates')!=1000:raise ValueError('Separate factored optimizer work differs')
    if trained.get('component_only')is not True or restored.get('full_state_temporal_candidate')is not False:
        raise ValueError('Component result must not claim full-state temporal prediction')
    parity={name:sha(args.output/'train'/name)==sha(args.output/'restore'/name)
            for name in ('weights.npz','initial-weights.npz','history.json','normalization.json','probe.npy')}
    if not all(parity.values())or not restored['full_cpu_probe_bitwise_verified']:
        raise ValueError('Temporal CPU restoration parity failed')
    write(args.output/'receipt.json',dict(status='completed',format='rheon-causal-request-managed-pair-v1',
        training_run_id=runs['train']['id'],restore_run_id=runs['restore']['id'],optimizer_updates=trained['optimizer_updates'],
        horizons=trained['horizons'],parameters=trained['parameters'],artifact_parity=parity,
        source_project=str(project_dir.relative_to(ROOT)),source_assembly_sha256=sha(project_dir/'assembly.json'),
        training_receipt_sha256=sha(args.output/'train/receipt.json'),restore_receipt_sha256=sha(args.output/'restore/receipt.json'),
        source_sha256=sha(__file__),elapsed_s=time.perf_counter()-started,component_only=True,full_state_temporal_candidate=False,model_promotion=False))
    print('Completed managed causal component fit and exact CPU restore',assembly['config']['identifier'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
