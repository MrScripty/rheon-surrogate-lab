"""Prove each requested track from current sources, runs, arrays and runtime."""
import argparse,ast,hashlib,os,subprocess,sys,time
from pathlib import Path
from v5_support import ROOT,HERE,BASE,DOCS,read,write,sha,environment
ARMS=['v5-raw','v5-multilevel','v5-spectrum']
def audit(output):
    if output.exists()or(DOCS/'final-audit.json').exists():raise ValueError('Fresh final audit required')
    output.mkdir();evidence={};hashes={}
    def digest(p):
        p=Path(p).resolve()
        if p not in hashes:hashes[p]=sha(p)
        return hashes[p]
    def keep(p):p=Path(p).resolve();evidence[str(p.relative_to(ROOT))]=digest(p);return read(p)
    def require(value,message):
        if not value:raise ValueError(message)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    scope=keep(DOCS/'requested-scope.json');require(hashlib.sha256(scope['objective'].encode()).hexdigest()==scope['objective_sha256'],'Requested scope changed')
    for version in [3,4]:
        pub=keep(ROOT/f'docs/plans/pressure-methods-v{version}/publication.json')
        for group in ['artifacts','sources']:
            for name,expected in pub.get(group,{}).items():require(digest(ROOT/name)==expected,'Prior published material changed: '+name)
    seed=keep(BASE/'tiny-initialization-01/initialization.json');require(seed['parameters']==648 and digest(BASE/'tiny-initialization-01/weights.npz')==seed['weights_sha256'],'Shared tiny initialization changed')
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json');fits=[];small=set();ids=[]
    sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    (output/'runs').mkdir();run_records=[]
    for arm in ARMS:
        project=BASE/(arm+'-project-02');study=BASE/(arm+'-study-01');result=keep(study/'receipt.json');train=keep(study/'train/receipt.json');restore=keep(study/'restore/receipt.json')
        history=keep(study/'train/history.json');config=keep(project/'config.json');objective=keep(project/'fine-tuning.json');document=keep(project/'project.json');witness=keep(project/'source-witness.json');assembly=keep(project/'assembly.json')
        require(result['status']=='completed'and result['optimizer_updates']==2400,'Planned fit incomplete')
        require(train['mode']=='train'and train['optimizer_updates']==2400 and train['execution']['device'].startswith('cuda'),'Actual CUDA updates required')
        require(restore['mode']=='restore'and restore['optimizer_updates']==0 and not restore['execution']['cuda_initialized'],'Fresh zero-update CPU restore required')
        require(train['prediction_inventory']==restore['prediction_inventory'],'Full published CPU prediction inventory differs')
        require(result['assembly_sha256']==digest(project/'assembly.json'),'Managed assembly changed')
        for name in ['weights.npz','normalization.json','predictions.npz','history.json','checkpoint-2400.npz']:
            require(result['artifact_parity'][name]and digest(study/'train'/name)==digest(study/'restore'/name),'Actual artifact parity failed')
        inventory=history['checkpoints']['checkpoint-2400.npz']['inventory']
        require(all(v['dtype']=='<f4'for v in inventory.values())and sum(v['bytes']//4 for v in inventory.values())==648,'Actual parameter/dtype inventory differs')
        require(history['checkpoints']['checkpoint-000.npz']['inventory']==seed['inventory'],'Matched seeded starting tensors differ')
        require(objective['initial_inventory']==seed['inventory']and objective['initial_weights_sha256']==seed['weights_sha256'],'Captured initialization differs')
        require(objective['coarse_training_cycles']==(0 if arm=='v5-raw'else 1),'Required raw/post-coarse contrast changed')
        require(config['training']['updates']==2400 and config['training']['learning_rate']==.001,'Captured finite optimizer schedule differs')
        for name,expected in witness['sources'].items():require(digest(project/name)==expected,'Captured training source changed')
        for row in document['data']:require(digest(ROOT/row['path'])==row['sha256'],'Captured fitting input changed')
        require(train['closure']['source_witness_sha256']==digest(project/'source-witness.json'),'Actual fit source closure differs')
        q=keep(BASE/(arm+'-qualification-02.json'));cuda=keep(BASE/(arm+'-cuda-qualification-02.json'))
        for qualification in [q,cuda]:
            require(qualification['status']=='passed'and qualification['optimizer_updates']==0 and qualification['parameters']==648,'Zero-update objective qualification missing')
            require(qualification['project_sha256']==digest(project/'project.json')and qualification['source_witness_sha256']==digest(project/'source-witness.json'),'Qualified Project differs')
            require(qualification['source_sha256']==digest(HERE/'qualify.py'),'Qualifier source changed')
            require(qualification['finite_largest_grid_backward']and qualification['largest_counts_xyz']==[128]*3,'Full128³ finite backward not qualified')
            require(qualification['zero_equation_backward']['network_forwards']==0 and qualification['zero_equation_backward']['loss']==0,'Zero bypass differs')
            require(qualification['small_replay']['matching_updates']==1600 and qualification['small_replay']['exact_sample_subsequence'],'Original small exposure changed')
        require(not q['cuda_initialized']and cuda['cuda_initialized'],'Actual separate CPU/CUDA qualification required')
        require(q['inference_label_independent']and not q['teacher_present'],'Inference label isolation differs')
        small.add(q['sampling']['small_subsequence_sha256'])
        work=history['optimizer_work'];require(work['network_forwards']==2400 and work['forward_cells']==q['sampling']['sampled_cells'],'Actual optimizer work differs from declared one-forward sampling')
        require(train['fine_tuning_statistics']['teacher_forwards']==0,'Unexpected teacher work')
        transfer=keep(BASE/'transfer'/(arm+'.json'));require(transfer['status']=='passed'and {r['frame']for r in transfer['rows']}=={0,7},'Full128³ cold/warm staged transfer missing')
        require(all(r['raw_input_bitwise_equal']and r['raw_forward_bitwise_equal']and r['complete_first_iteration_relative_difference']<=1e-12 for r in transfer['rows']),'Actual deployed proposal/GCR transfer differs')
        for phase,key in [('fit','training_run_id'),('restore','restore_run_id')]:
            run=owner.call('runs.get',dict(run_id=result[key]));path=output/'runs'/(run['id']+'.json');write(path,run);keep(path)
            require(run['state']=='completed'and run['custody_phase']=='drained','Fresh managed custody incomplete')
            ids.append(run['id']);run_records.append(dict(arm=arm,phase=phase,run_id=run['id'],state=run['state'],custody_phase=run['custody_phase']))
        fits.append(dict(arm=arm,optimizer_updates=2400,parameters=648,optimizer_work=work,artifact_parity=True))
    require(len(set(ids))==6 and len(small)==1,'Actual six unique managed Runs and matched small stream required')
    prior_audit=read(ROOT/'docs/plans/pressure-methods-v4/final-audit.json');require(next(iter(small))==prior_audit['common_small_subsequence_sha256'],'Original1600 small choices changed')
    keep(BASE/'v5-raw-controller-metadata-clarification.json')
    view=keep(BASE/'spectrum-training-view-01/manifest.json');original=read(ROOT/'runs/pressure-methods-v3/derived-training-view-01/manifest.json')
    require(len(view['parents'])==46 and view['retained_v3_parents']==34 and view['added_spectral_parents']==12,'Actual spatial/native fitting populations differ')
    require(view['parents'][:34]==original['parents']and all(view['arrays'][k]==v for k,v in original['arrays'].items()),'Original full field declarations changed')
    require(view['encoded_total_bytes']<=236*1024**2 and all(s['bytes']<=40*1024**2 for s in view['shards']),'Ordinary input/restore reserve exceeded')
    reservation=keep(DOCS/'scene-reservation.json');train_ids={r['trajectory_id']for r in reservation['spectrum']if r['population']=='train'}
    require({r['trajectory_id']for r in view['parents'][34:]}==train_ids and all(r['split']=='train'for r in view['parents']),'Held-out scene entered fitting')
    patterns=keep(BASE/'spectrum-training-view-01/spatial-patterns.json');require(len(patterns['rows'])==12,'Explicit spatial wavelength inventory missing')
    # Independently decode every fitting field after all timing has finished.
    bank=output/'full-bank-retention.json'
    with (output/'bank.log').open('xb')as stream:
        child=subprocess.run([runtime['python'],'-B',str(HERE/'verify_bank.py'),'--output',str(bank)],cwd=ROOT,env=environment(runtime),stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    require(child.returncode==0,'Full expanded-bank decoding failed');decoded=keep(bank)
    require(decoded['parents']==46 and decoded['retained_original_parents']==34 and decoded['full_fields']==184,'Independent bank retention incomplete')
    selection=keep(DOCS/'selection.json');keep(DOCS/'model-selection.json');keep(DOCS/'recycling-study-design.json')
    require(selection['status']=='frozen_before_blind_capture'and selection['reservation_sha256']==digest(DOCS/'scene-reservation.json'),'Blind selection changed')
    for name,expected in selection['all_development_evaluations'].items():require(digest(ROOT/name)==expected,'Selection development evidence changed')
    matrix=keep(DOCS/'development-matrix.json');expected_ids={r['identifier']for r in matrix['points']}|{'v5-initial-gcr'}
    files=list((BASE/'evaluation-final').glob('*.json'));require({p.stem for p in files}==expected_ids,'Complete common development matrix missing')
    populations=[]
    for p in files:
        d=keep(p);require(d['status']=='completed'and len(d['rows'])==220,'Common192+10+18 population differs')
        populations.append({(r['population'],r['trajectory_id'],r['frame'])for r in d['rows']})
    require(all(p==populations[0]for p in populations),'Development requests differ between methods')
    blind=list((BASE/'blind-evaluation').glob('*.json'));require(len(blind)==14,'All frozen blind policies and initial control required')
    signatures=set()
    for p in blind:
        d=keep(p);require(d['status']=='completed'and len(d['rows'])==34,'Full ten-native/24-pattern blind evaluation missing');signatures.add(d['population_sha256'])
    require(len(signatures)==1,'Blind tensor inputs differ between policies')
    for row in reservation['rows']:
        if row['population']not in ['development','blind']:continue
        directory=BASE/'native-data'/(row['trajectory_id']+'-spectral');d=keep(directory/'receipt.json')
        require(d['row']==row and d['status']=='completed'and d['native_process']['reaped'],'Exact native data capture incomplete')
        if row['population']=='blind':require((directory/'receipt.json').stat().st_mtime>(DOCS/'selection.json').stat().st_mtime,'Blind capture preceded frozen selection')
        for frame in d['snapshots']:
            for spec in frame['arrays'].values():require(digest(directory/spec['path'])==spec['sha256'],'Native pressure request bytes changed')
    build=keep(ROOT/'runs/pressure-methods-v4/streamed-build-01/build-receipt.json')
    require(build['status']=='qualified_build'and all(c['exit_code']==0 for c in build['commands']),'Qualified independent native host build missing')
    require(digest(ROOT/'runs/pressure-methods-v4/streamed-build-01/rheon-large-single-track')==build['binary_sha256'],'Native binary changed')
    for name,expected in build['sources'].items():require(digest(ROOT/name)==expected,'Qualified native source changed')
    for group,expected in [('rollouts-final',21),('rollouts-blind',18)]:
        files=list((BASE/group).glob('*/receipt.json'))
        # Development: nine numerical/reference controls, twelve model tests,
        # plus no duplicate final episodes. Composition lives separately.
        require(len(files)==expected,'Native rollout comparison matrix incomplete')
        for p in files:
            d=keep(p);require(d['native_process']['reaped']and d['scene']['steps']==32,'Bounded consecutive native episode incomplete')
            require(d['status']=='completed'and len(d['frames'])==32 and all(f['advanced']and f['quality']['qualified']for f in d['frames']),'Native32-step qualification failed')
            if d['policy']=='gcr':
                require(d['callbacks'][0]['work']['cache_in']==0,'Cold trajectory inherited a direction cache')
                require(all(c['work']['cache_in']<=d['reuse_capacity']for c in d['callbacks']),'Direction storage bound exceeded')
            for state in d['states']:require(digest(p.parent/state['path'])==state['sha256'],'Full trajectory state archive changed')
    for group,count in [('drift',12),('drift-blind',6)]:
        files=list((BASE/group).glob('*.json'));require(len(files)==count,'Full-field drift matrix incomplete')
        for p in files:
            d=keep(p);require(d['matched_accepted_steps']==32 and d['diagnostic_only']and not d['application_acceptance_claimed'],'Matched diagnostic-only drift boundary differs')
    stress=keep(BASE/'sequence-stress-03.json');require(len(stress['rows'])==96 and all(r['qualified']for r in stress['rows']),'Full-grid abrupt-spectrum/geometry sequence qualification incomplete')
    screen=keep(BASE/'execution-selected-03.json');require(len(screen['rows'])==6,'Selected complete-policy execution screen missing')
    keep(BASE/'hierarchy-profile-01.json');keep(BASE/'returned-pressure-red-01.json');keep(BASE/'returned-pressure-green-03.json');keep(DOCS/'prior-report-erratum.json')
    require(keep(BASE/'pre-returned-gate-evaluation-stop.json')['status']=='stopped_reaped_excluded','Superseded evaluation not excluded')
    for name,count in [('all-timing-batch-03',88),('prior-controls-timing-batch-04',4)]:
        d=keep(BASE/name/'execution.json');require(d['status']=='completed'and d['owned_child_reaped']and len(d['completed'])==count and all(r['returncode']==0 and r['reaped']for r in d['completed']),'Final serial timing incomplete')
    for ready_name,admit_name,plan_name in [('quiet-ready-03.json','timing-admission-03.json','all-timing-plan-03.json'),
        ('quiet-ready-supplement-04.json','timing-admission-supplement-04.json','prior-controls-timing-plan-04.json')]:
        ready=keep(BASE/ready_name);admitted=keep(BASE/admit_name);keep(BASE/plan_name)
        require(ready['status']=='ready_for_serial_timing'and ready['host_namespace_process_audit']and not ready['live_owned_workers'],'Host readiness incomplete')
        require(admitted['readiness_sha256']==digest(BASE/ready_name)and admitted['plan_sha256']==digest(BASE/plan_name),'Timing admission identities differ')
        require(0<=admitted['started_unix']-ready['checked_unix']<=120,'Timing did not follow fresh completed readiness')
    files=list((BASE/'quiet-pressure').glob('*.json'));require(len(files)==26,'Thirteen matched pressure policies on both populations required')
    for p in files:
        d=keep(p);count=128 if d['population']=='historical128'else 10
        require(d['status']=='completed'and len(d['rows'])==count and d['cpu']==23 and d['threads']==1 and d['repeats']==3,'Comparable pressure timing differs')
    files=list((BASE/'quiet-native').glob('*/receipt.json'));require(len(files)==66,'Native timing66-episode matrix incomplete')
    for p in files:
        d=keep(p);require(d['native_process']['reaped']and d['cpu']==23 and d['threads']==1 and not d['request_labels_exported']and not d['states']and not d['snapshots'],'Native timing includes exported labels/states or live work')
        require(all(not f['advanced']or f['quality']['qualified']for f in d['frames']),'A rejected native step advanced')
        if d['status']=='completed':require(len(d['frames'])==d['scene']['steps']and all(f['advanced']for f in d['frames']),'Incomplete episode mislabeled complete')
        else:require(d['status']=='pressure_rejection'and not d['frames'][-1]['advanced'],'Rejected timing attempt hidden')
    # Execute the independent mathematics, representation and capacity tests.
    tests=[]
    for name in ['test_recycling_v3.py','test_fast_multilevel.py','test_spectrum.py']:
        log=output/(name+'.log')
        with log.open('xb')as stream:child=subprocess.run([runtime['python'],'-B',str(HERE/name)],cwd=ROOT,env=environment(runtime),stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        require(child.returncode==0,'Independent test failed: '+name);tests.append(dict(source=name,source_sha256=digest(HERE/name),log_sha256=digest(log),exit_code=0))
    for module,expected in [('solver','failed_returned_pressure_gate'),('solver_v3','passed')]:
        path=output/('returned-'+module+'.json');args=[runtime['python'],'-B',str(HERE/'check_returned_pressure.py'),'--module',module,'--output',str(path)]
        if module=='solver_v3':args.append('--require')
        with (output/('returned-'+module+'.log')).open('xb')as stream:child=subprocess.run(args,cwd=ROOT,env=environment(runtime),stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        require(child.returncode==0 and keep(path)['status']==expected,'Actual returned-pressure red/green regression differs')
    figures=keep(DOCS/'figure-evidence.json');require(figures['status']=='generated_from_retained_results','Final scientific figures missing')
    require(figures['source_sha256']==digest(HERE/'figures.py'),'Scientific figure source changed')
    for group in ['inputs','artifacts']:
        for name,expected in figures[group].items():require(digest(ROOT/name)==expected,'Scientific figure evidence changed')
    require(len(figures['artifacts'])>=12,'Standalone figure formats incomplete')
    workers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit()or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0');exe=Path(args[0].decode()).name
            if(exe.startswith('python')or exe=='rheon-large-single-track')and(any(b'pressure-methods-v5'in a for a in args)or any(any(id.encode()in a for id in ids)for a in args)):
                workers.append(dict(pid=int(p.name),executable=exe))
        except(OSError,UnicodeError,IndexError):pass
    require(not workers,'Live owned workers remain')
    result=dict(status='passed',successful_fits=3,successful_managed_runs=6,optimizer_updates=7200,fits=fits,managed_runs=run_records,independent_tests=tests,
        source_sha256=sha(__file__),evidence=evidence,live_owned_workers=workers,host_namespace_process_audit=True,checked_unix=time.time(),promotion=False)
    write(DOCS/'final-audit.json',result)
    requirements=[dict(id=i,status='proven',requirement=name)for i,name in enumerate([
        'Matched648-parameter raw/post-coarse/post-coarse-plus-wavelength managed training, fresh CPU restores and full128³ staged deployment transfer',
        'Bounded host-committed inter-step GCR reuse, useful-direction retention, independent mathematics/returned-field regressions, native work and repeated timing',
        'Explicit physical wavelengths and rectangular geometry with exact original-bank retention, reserved new development/blind data and matched fitting contrasts',
        'Measured operator/inference/native-exchange costs, qualified direct stencil/layout changes and repeated same-core full-policy/native timings'],1)]
    write(DOCS/'completion-audit.json',dict(status='complete',requirements=requirements,final_audit_sha256=sha(DOCS/'final-audit.json'),source_sha256=sha(__file__),
        limitations=['No application model promotion','Diagnostic32-step rollout horizons only','128³ native timing is four steps','No production256³/512³/1024³ qualification','No claimed universal neural advantage over numerical controls']))
    print('All four requested objectives proven against current retained evidence',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'final-audit-01');audit(p.parse_args().output)
