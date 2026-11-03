"""Require current direct evidence for every requested scientific track."""
import argparse,ast,hashlib,math,os,subprocess,sys,time
from pathlib import Path
from v6_support import ROOT,HERE,BASE,DOCS,read,write,sha,environment,fit_record
def run(output):
    if output.exists()or(DOCS/'final-audit.json').exists():raise ValueError('Fresh final audit required')
    output.mkdir();evidence={};cache={}
    def digest(p):
        p=Path(p).resolve()
        if p not in cache:cache[p]=sha(p)
        return cache[p]
    def keep(p):p=Path(p).resolve();evidence[str(p.relative_to(ROOT))]=digest(p);return read(p)
    def require(value,message):
        if not value:raise ValueError(message)
    campaign=keep(DOCS/'campaign.json');requirements=keep(DOCS/'evidence-requirements.json');keep(DOCS/'fit-records.json')
    require(len(campaign['tracks'])==5 and len(requirements['tracks'])==5,'Five original tracks required')
    require(digest(DOCS/'requested-scope.txt')==campaign['requested_scope_sha256'],'Original requested scope changed')
    for version in [3,4,5]:
        p=keep(ROOT/f'docs/plans/pressure-methods-v{version}/publication.json')
        for group in ['artifacts','sources']:
            for name,expected in p.get(group,{}).items():require(digest(ROOT/name)==expected,'Earlier publication changed: '+name)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    queue=HERE/'final_timing_queue.sh'
    require(subprocess.run(['bash','-n',str(queue)],capture_output=True).returncode==0,'Final timing queue syntax invalid')
    evidence[str(queue.relative_to(ROOT))]=digest(queue)
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    (output/'managed-runs').mkdir();managed=[];fits=[];small=set();initials={};matched_sampling=[];matched_training=[];order_sources=set();normalizations=set()
    counts={'v6-replay':648,'v6-unrolled':648,'v6-stencil':27,'v6-separable':9}
    for arm in campaign['fits']:
        record=fit_record(arm);project=ROOT/record['project'];study=ROOT/record['study']
        r=keep(study/'receipt.json');train=keep(study/'train/receipt.json');restore=keep(study/'restore/receipt.json');history=keep(study/'train/history.json')
        config=keep(project/'config.json');spec=keep(project/'fine-tuning.json');witness=keep(project/'source-witness.json');document=keep(project/'project.json');keep(project/'assembly.json')
        require(r['status']=='completed'and r['optimizer_updates']==2400 and r['assembly_sha256']==digest(project/'assembly.json'),'Completed captured2400-update fit required')
        require(train['mode']=='train'and train['optimizer_updates']==2400 and train['execution']['device'].startswith('cuda'),'Actual CUDA training required')
        require(restore['mode']=='restore'and restore['optimizer_updates']==0 and not restore['execution']['cuda_initialized'],'Fresh zero-update CPU restore required')
        require(train['prediction_inventory']==restore['prediction_inventory'],'Full published prediction inventories differ')
        for name in ['weights.npz','normalization.json','predictions.npz','history.json','checkpoint-2400.npz']:
            require(r['artifact_parity'][name]and digest(study/'train'/name)==digest(study/'restore'/name),'Actual artifact parity failed')
        inventory=history['checkpoints']['checkpoint-2400.npz']['inventory']
        require(all(v['dtype']=='<f4'for v in inventory.values())and sum(v['bytes']//4 for v in inventory.values())==counts[arm],'Actual trained parameter inventory differs')
        require(history['checkpoints']['checkpoint-000.npz']['inventory']==spec['initial_inventory'],'Actual initial tensors differ')
        initials[arm]=spec['initial_weights_sha256']
        require(config['training']['updates']==2400 and config['resources']['wall_seconds']==1800,'Finite update/runtime budget changed')
        matched_training.append(config['training']);order_sources.add(digest(project/'mixed_order.py'));normalizations.add(digest(project/'fixed-normalization.json'))
        require(spec['solver_training_iterations']==(3 if arm=='v6-unrolled'else 1),'Training iteration contrast changed')
        for name,expected in witness['sources'].items():require(digest(project/name)==expected,'Captured training source changed')
        for item in document['data']:require(digest(ROOT/item['path'])==item['sha256'],'Actual fitting material changed')
        for key,device in [('qualification',False),('cuda_qualification',True)]:
            q=keep(ROOT/record[key]);require(q['status']=='passed'and q['optimizer_updates']==0 and q['cuda_initialized']==device,'Separate zero-update device qualification missing')
            require(q['project_sha256']==digest(project/'project.json')and q['source_witness_sha256']==digest(project/'source-witness.json'),'Qualified source closure differs')
            require(q['source_sha256']==digest(HERE/'qualify.py')and q['parameters']==counts[arm],'Qualification source/parameter count differs')
            require(q['finite_largest_grid_backward']and q['largest_counts_xyz']==[128]*3 and q['inference_label_independent'],'Full128³ backward/label isolation missing')
            require(q['small_replay']['matching_updates']==1600 and q['small_replay']['exact_sample_subsequence'],'Original small sample stream changed')
            small.add(q['sampling']['small_subsequence_sha256'])
        matched_sampling.append(q['sampling'])
        work=history['optimizer_work'];steps=spec['solver_training_iterations']
        require(work['network_forwards']==2400*steps and work['forward_cells']==q['sampling']['sampled_cells']*steps,'Actual optimizer forward work differs')
        require(train['fine_tuning_statistics']['teacher_forwards']==0 and train['fine_tuning_statistics']['coarse_cycles']==2400*steps,'Unexpected teacher or coarse optimizer work')
        transfer=keep(BASE/'transfer'/(arm+'.json'));require(transfer['status']=='passed'and len(transfer['rows'])==2,'Complete full128³ cold/warm transfer missing')
        require(all(x['raw_local_bitwise_equal']and x['complete_first_iteration_relative_difference']<=1e-12 for x in transfer['rows']),'Actual deployed local/GCR construction differs')
        for phase,key in [('fit','training_run_id'),('restore','restore_run_id')]:
            state=owner.call('runs.get',dict(run_id=r[key]));p=output/'managed-runs'/(state['id']+'.json');write(p,state);keep(p)
            require(state['state']=='completed'and state['custody_phase']=='drained','Fresh managed custody incomplete')
            managed.append(dict(arm=arm,phase=phase,run_id=state['id'],state=state['state'],custody_phase=state['custody_phase']))
        fits.append(dict(arm=arm,parameters=counts[arm],updates=2400,optimizer_work=work,exact_cpu_artifact_parity=True))
    require(len(small)==1 and initials['v6-replay']==initials['v6-unrolled'],'Matched initialization or original small sampling differs')
    require(all(x==matched_training[0]for x in matched_training)and all(x==matched_sampling[0]for x in matched_sampling)and len(order_sources)==1 and len(normalizations)==1,'Full sampling generator, exposure, optimizer schedule or frozen normalization differs')
    original=read(ROOT/'runs/pressure-methods-v5/v5-multilevel-qualification-02.json')
    require(next(iter(small))==original['sampling']['small_subsequence_sha256'],'Original1600 small choices changed')
    failed=keep(BASE/'failed-unrolled-attempt-01/summary.json');receipt=keep(BASE/'failed-unrolled-attempt-01/receipt.json')
    require(failed['status']=='failed_drained_preserved_excluded'and receipt['reason']=='wall_timeout'and receipt['drained']and failed['minimum_completed_updates']>=2200,'Partial failed attempt not honestly retained')
    for name,item in failed['streams'].items():require(digest(BASE/'failed-unrolled-attempt-01'/name)==item['sha256'],'Archived failed stream changed')
    state=owner.call('runs.get',dict(run_id=failed['run_id']));p=output/'managed-runs'/(state['id']+'.json');write(p,state);keep(p)
    require(state['state']=='failed'and state['custody_phase']=='drained','Discarded attempt custody incomplete')
    amendment=keep(BASE/'adjoint-execution-amendment-01.json');parity=keep(BASE/'adjoint-objective-parity-01.json');keep(BASE/'unrolled-gradient-02.json')
    require(amendment['wall_seconds']==1800 and not amendment['global_runtime_limits_changed'],'Runtime ceiling was changed')
    require(parity['status']=='passed'and any(r['counts_zyx']==[128]*3 for r in parity['rows'])and all(r['loss_relative_difference']<=2e-6 and r['all_parameter_gradient_relative_difference']<=1e-5 for r in parity['rows']),'Actual composed adjoint objective/gradient differs')
    for p in [BASE/'v6-unrolled-project-01/config.json',BASE/'v6-unrolled-project-01/fine-tuning.json']:
        require(digest(p)==digest(BASE/'v6-unrolled-project-02'/p.name),'Scientific settings changed in the execution amendment')
    bank=output/'full-fitting-bank.json'
    with(output/'bank.log').open('xb')as log:child=subprocess.run([runtime['python'],'-B',str(HERE/'verify_training_bank.py'),'--output',str(bank)],cwd=ROOT,env=environment(runtime),stdout=log,stderr=subprocess.STDOUT,timeout=300)
    require(child.returncode==0,'Independent full fitting-bank decode failed');require(keep(bank)['full_fields']==136,'All136 original fitting fields required')
    selection=keep(DOCS/'selection.json');pre=keep(DOCS/'preselection.json');cal=keep(DOCS/'reuse-calibration.json');reservation=keep(DOCS/'scene-reservation.json');points=keep(DOCS/'calibration-points-03.json')['points']
    require(selection['status']=='frozen_before_blind_capture'and selection['reservation_sha256']==digest(DOCS/'scene-reservation.json'),'Blind reservation/selection differs')
    require(selection['preselection_sha256']==digest(DOCS/'preselection.json')and selection['reuse_calibration_sha256']==digest(DOCS/'reuse-calibration.json'),'Frozen development selection differs')
    for d,group in [(selection,'frozen_sources'),(pre,'evidence'),(cal,'evidence')]:
        for name,expected in d[group].items():require(digest(ROOT/name)==expected,'Frozen evidence or predictor changed: '+name)
    signatures=set()
    for point in points:
        d=keep(BASE/'development'/(point['identifier']+'.json'));t=keep(BASE/'quiet-development'/(point['identifier']+'.json'))
        require(d['status']=='completed'and d['cases']==254 and len(d['rows'])==254,'Complete common254 development requests required')
        require(t['status']=='completed'and len(t['rows'])==20 and t['cpu']==23 and t['threads']==1 and t['repeats']==3,'Matched repeated20-request timing differs')
        require(t['points_sha256']==digest(DOCS/'calibration-points-03.json')and all(len(r['repeat_seconds'])==3 for r in t['rows']),'Actual timing repetition/configuration differs')
        signatures.add((d['population_sha256'],t['population_sha256']))
    require(len(points)==19 and len(signatures)==1,'Nineteen points on identical development/timing inputs required')
    for selected in selection['selected']:
        d=read(BASE/'quiet-development'/(selected['point']['identifier']+'.json'))
        require(sum(r['network_calls']for r in d['rows'])>0,'Selected learned policy never invoked the network')
    require(cal['status']=='frozen_from_consumed_development'and cal['minimum_cosine']==.9 and cal['overhead_margin']==1.15 and cal['point']==selection['primary'],'Frozen adaptive economics differs')
    calibrated=keep(BASE/'adaptive-calibrated-qualification-01/receipt.json')
    require(calibrated['status']=='completed'and len(calibrated['frames'])==32 and calibrated['reuse_calibration_sha256']==digest(DOCS/'reuse-calibration.json')and all(f['advanced']and f['quality']['qualified']for f in calibrated['frames']),'Real calibrated adaptive native qualification missing')
    blind=[];blind_signatures=set()
    for point in selection['blind_points']:
        d=keep(BASE/'blind-evaluation'/(point['identifier']+'.json'));require(d['status']=='completed'and d['cases']==14 and len(d['rows'])==14,'All frozen points must evaluate all14 blind requests')
        require(d['selection_sha256']==digest(DOCS/'selection.json'),'Blind point used a different selection');blind_signatures.add(d['population_sha256']);blind.append(d)
    require(len(blind_signatures)==1,'Different blind input populations')
    build=keep(ROOT/'runs/pressure-methods-v4/streamed-build-01/build-receipt.json')
    require(digest(ROOT/'runs/pressure-methods-v4/streamed-build-01/rheon-large-single-track')==build['binary_sha256'],'Native binary changed')
    for name,expected in build['sources'].items():require(digest(ROOT/name)==expected,'Native physical source changed')
    captures=[]
    for row in reservation['rows']:
        if row['population']not in ['blind','large-blind']:continue
        folder=BASE/'native-data'/(row['trajectory_id']+'-spectral');d=keep(folder/'receipt.json');captures.append(d)
        require(d['row']==row and d['status']=='completed'and d['native_process']['reaped'],'Complete reserved native capture missing')
        require(d['selection_sha256']==digest(DOCS/'selection.json')and d['started_unix']>=selection['frozen_unix'],'New native data preceded frozen selection')
        require([s['index']for s in d['snapshots']]==row['snapshots'],'Full cold/warm snapshot population differs')
        for s in d['snapshots']:
            require(set(s['arrays'])=={'base','rhs','reference','native_input_residual'},'Incomplete full-field blind equation')
            for field,item in s['arrays'].items():
                require(item['shape']==[1,1,*reversed(row['scene']['counts'])]and item['dtype']=='<f8'and item['bytes']==8*math.prod(row['scene']['counts']),'Blind field was cropped or changed grid dimensions')
                require((folder/item['path']).stat().st_size==item['bytes']and digest(folder/item['path'])==item['sha256'],'Full blind field bytes changed')
    require(len(captures)==7 and sum(d['scene']['counts']==[256]*3 for d in captures)==2,'Seven scenes including two full256³ cubes required')
    long=[]
    for row in reservation['rows']:
        if row['population']!='long-blind':continue
        for policy in ['spectral','numerical','selected-fresh','selected-fixed4','selected-adaptive4']:
            folder=BASE/'long-rollouts'/(row['trajectory_id']+'-'+policy);d=keep(folder/'receipt.json');long.append(d)
            require(d['scene']==row['scene']and d['scene']['steps']==256 and d['native_process']['reaped'],'Two real256-step requested trajectories required')
            require(d['selection_sha256']==digest(DOCS/'selection.json'),'Long trajectory used a different selection')
            require(all(not f['advanced']or f['quality']['qualified']for f in d['frames']),'A rejected physical step advanced')
            if d['status']=='completed':require(len(d['frames'])==256 and len(d['states'])==256 and all(f['advanced']for f in d['frames']),'Incomplete long trajectory mislabeled complete')
            else:require(d['status']=='pressure_rejection'and not d['frames'][-1]['advanced'],'Long failed attempt hidden')
            cells=math.prod(row['scene']['counts']);counts=row['scene']['counts']
            lengths={'pressure':cells,'tracer':cells,**{field:(counts[axis]+1)*math.prod(counts[j]for j in range(3)if j!=axis)for axis,field in enumerate(['vx','vy','vz'])}}
            require([s['index']for s in d['states']]==list(range(sum(f['advanced']for f in d['frames']))),'An accepted long state is missing or reordered')
            for item in d['states']:
                require(set(item['arrays'])==set(lengths),'Full native state inventory differs')
                for field,length in lengths.items():
                    a=item['arrays'][field];require(a['shape']==[length]and a['dtype']==('<f8'if field=='pressure'else'<f4'),'State omitted native pressure cells or staggered velocity faces')
                require(digest(folder/item['path'])==item['sha256'],'Full long-state archive changed')
            if policy!='spectral':
                drift=keep(BASE/'long-drift'/(row['trajectory_id']+'-'+policy+'.json'))
                require(drift['diagnostic_only']and not drift['application_acceptance_claimed'],'Application drift limits were invented')
                require(drift['source_sha256']==digest(HERE/'drift.py')and drift['candidate_receipt_sha256']==digest(folder/'receipt.json')and drift['matched_accepted_steps']==len(drift['rows']),'Full-field drift population differs')
                reference=BASE/'long-rollouts'/(row['trajectory_id']+'-spectral')/'receipt.json'
                require(drift['reference_receipt_sha256']==digest(reference)and drift['matched_accepted_steps']==min(len(d['states']),len(read(reference)['states'])),'Drift omitted an accepted full-field step or changed reference')
                require(all(set(s['arrays'])=={'pressure','vx','vy','vz','tracer'}for s in d['states']),'Incomplete pressure/velocity/tracer state inventory')
            for c in d['callbacks']:
                w=c['work'];require(w['cache_in']<=4 and w['cache_bytes']<=64*math.prod(row['scene']['counts']),'Stored-space bound exceeded')
                a=w.get('adaptive_reuse')
                if policy=='selected-adaptive4':
                    require(a is not None and a['calibration_sha256']==digest(DOCS/'reuse-calibration.json'),'Adaptive decision used an unqualified calibration')
                    if a['admitted']:require(a['sketch_cosine']>=.9 and a['expected_savings_seconds']>1.15*a['expected_overhead_seconds'],'Unprofitable or unrelated reuse admitted')
    require(len(long)==10,'Ten full-state long attempts including fixed-four drift required')
    final_res=keep(DOCS/'timing-reservation.json');native_times=[]
    for row in final_res['rows']:
        for point in final_res['methods']:
            for repeat in range(3):
                p=BASE/'quiet-native'/(row['trajectory_id']+'-'+point['label']+f'-r{repeat}')/'receipt.json';d=keep(p);native_times.append(d)
                require(d['row']==row and d['cpu']==23 and d['threads']==1 and d['native_process']['reaped'],'Matched native timing context differs')
                require(not d['states']and not d['snapshots']and not d['request_labels_exported'],'Diagnostic export contaminated bare native timing')
                require(d['selection_sha256']==digest(DOCS/'selection.json'),'Native timing selection changed')
                require(all(not f['advanced']or f['quality']['qualified']for f in d['frames']),'Rejected timed step advanced')
                if d['status']=='completed':require(len(d['frames'])==row['scene']['steps']and all(f['advanced']for f in d['frames']),'Incomplete timing attempt hidden')
                else:require(d['status']=='pressure_rejection'and not d['frames'][-1]['advanced'],'Failed timing attempt hidden')
    require(len(native_times)==54,'All54 repeated native episodes required')
    for ready,admission,plan in [('quiet-ready-development-03.json','timing-admission-development-03.json','quiet-development-plan-03.json'),
      ('quiet-ready-reuse-cost-01.json','timing-admission-reuse-cost-01.json','quiet-reuse-cost-plan-01.json'),('quiet-ready-final-native-01.json','timing-admission-final-native-01.json','final-native-timing-plan-01.json')]:
        r=keep(BASE/ready);a=keep(BASE/admission);keep(BASE/plan)
        require(r['status']=='ready_for_serial_timing'and r['host_namespace_process_audit']and not r['live_owned_workers'],'Actual host timing readiness missing')
        require(r['source_sha256']==digest(HERE/'quiet_ready.py')and a['source_sha256']==digest(HERE/'start_timing.py'),'Readiness/admission source changed')
        require(a['readiness_sha256']==digest(BASE/ready)and a['plan_sha256']==digest(BASE/plan)and r['plan_sha256']==digest(BASE/plan)and 0<=a['started_unix']-r['checked_unix']<=120,'Timing did not follow exact fresh admission')
    batch_plans={'quiet-development-batch-03':('quiet-development-plan-03.json',19),'quiet-reuse-cost-batch-01':('quiet-reuse-cost-plan-01.json',9),
      'blind-capture-batch-01':('blind-capture-plan-01.json',7),'long-rollout-batch-01':('long-rollout-plan-01.json',8),
      'long-drift-batch-01':('long-drift-plan-01.json',6),'long-fixed-rollout-batch-01':('long-fixed-rollout-plan-01.json',2),
      'long-fixed-drift-batch-01':('long-fixed-drift-plan-01.json',2),'final-native-timing-batch-01':('final-native-timing-plan-01.json',54)}
    for name,(plan,count)in batch_plans.items():
        d=keep(BASE/name/'execution.json');require(d['status']=='completed'and d['owned_child_reaped']and all(x['returncode']==0 and x['reaped']for x in d['completed']),'Required finite batch has not completed and drained')
        specification=keep(BASE/plan)
        require(d['specification_sha256']==digest(BASE/plan)and len(d['completed'])==len(specification['commands'])==count and [r['id']for r in d['completed']]==[r['id']for r in specification['commands']],'Actual batch does not match its complete exact declared plan')
    for i in range(2):
        d=keep(BASE/f'blind-evaluation-batch-01-{i}'/'execution.json');require(d['status']=='completed'and d['owned_child_reaped'],'Blind evaluation worker has not drained')
    tests=[]
    for name in ['test_recycling.py','test_recycling_v2.py','test_adaptive.py','test_cycle_adjoint.py']:
        p=HERE/name;log=output/(name+'.log')
        with log.open('xb')as stream:child=subprocess.run([runtime['python'],'-B',str(p)],cwd=ROOT,env=environment(runtime),stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        require(child.returncode==0,'Independent regression failed: '+name);tests.append(dict(source=name,source_sha256=digest(p),log_sha256=digest(log),exit_code=0))
    for name in ['fixed-control-qualification-01.json','selective-work-qualification-01.json','instrumented-solver-parity-02.json']:
        require(keep(BASE/name)['status']=='passed','Independent operator/dispatch evidence missing: '+name)
    require(keep(BASE/'stagnation-dispatch-stress-01.json')['status']=='passed_dispatch_and_returned_field_check','Dormant dispatch branch was not exercised')
    for name in ['native-qualification-large-03.json','native-streaming-qualification-large-03.json']:
        q=keep(BASE/name);require(q['status']=='passed'and q['large']and len(q['rows'])==10,'Full native boundary/256³ parity qualification missing')
        require(q['source_sha256']==digest(HERE/'check_native.py')and any(r['counts_zyx']==[256]*3 for r in q['rows']),'Native qualifier source or actual full256³ case differs')
        require(all(r['relative_l2_difference']<=2e-6 and r['output_ownership']for r in q['rows']),'Native boundary precision or output ownership failed')
        for path,expected in q['dependencies'].items():require(digest(ROOT/path)==expected,'Native qualified dependency changed')
    returned=output/'returned-pressure.json'
    with(output/'returned-pressure.log').open('xb')as stream:
        child=subprocess.run([runtime['python'],'-B',str(HERE/'check_returned_pressure.py'),'--module','solver_v2','--output',str(returned),'--require'],cwd=ROOT,env=environment(runtime),stdout=stream,stderr=subprocess.STDOUT,timeout=120)
    returned_check=keep(returned)
    require(child.returncode==0 and returned_check['status']=='passed'and returned_check['actual_returned_field_operator']and returned_check['policy_source_sha256']==digest(HERE/'solver_v2.py'),'Actual large-base returned-pressure regression failed')

    figures=keep(DOCS/'figure-evidence.json');require(figures['source_sha256']==digest(HERE/'figures.py')and len(figures['artifacts'])==15,'Five standalone scientific figures required')
    for group in ['inputs','artifacts']:
        for name,expected in figures[group].items():require(digest(ROOT/name)==expected,'Figure evidence changed')
    visual=keep(DOCS/'visual-inspection.json')
    require(visual['status']=='visually_inspected'and visual['figure_source_sha256']==digest(HERE/'figures.py')and visual['png_artifacts']=={p:h for p,h in figures['artifacts'].items()if p.endswith('.png')},'Final five-figure visual inspection missing or stale')
    workers=[];ids=[r['run_id'].encode()for r in managed]+[failed['run_id'].encode()]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit()or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0');exe=Path(args[0].decode()).name
            if(exe.startswith('python')and(any(b'pressure-methods-v6'in a for a in args)or any(any(i in a for i in ids)for a in args)))or exe=='rheon-large-single-track':workers.append(dict(pid=int(p.name),executable=exe))
        except(OSError,IndexError,UnicodeError):pass
    require(not workers,'Owned numerical workers remain live')
    # Each original subrequirement has an explicit inspected evidence closure.
    shared=['docs/plans/pressure-methods-v6/selection.json','docs/plans/pressure-methods-v6/figure-evidence.json']
    proof_groups={
      1:['runs/pressure-methods-v6/native-qualification-large-03.json','runs/pressure-methods-v6/native-streaming-qualification-large-03.json','runs/pressure-methods-v6/quiet-development-batch-03/execution.json','runs/pressure-methods-v6/final-native-timing-batch-01/execution.json'],
      2:['runs/pressure-methods-v6/selective-work-qualification-01.json','runs/pressure-methods-v6/stagnation-dispatch-stress-01.json','runs/pressure-methods-v6/instrumented-solver-parity-02.json','runs/pressure-methods-v6/quiet-development-batch-03/execution.json'],
      3:['docs/plans/pressure-methods-v6/fit-records.json','runs/pressure-methods-v6/adjoint-objective-parity-01.json','runs/pressure-methods-v6/unrolled-gradient-02.json'],
      4:['docs/plans/pressure-methods-v6/reuse-calibration.json','runs/pressure-methods-v6/adaptive-calibrated-qualification-01/receipt.json','runs/pressure-methods-v6/long-fixed-drift-batch-01/execution.json','runs/pressure-methods-v6/long-drift-batch-01/execution.json','runs/pressure-methods-v6/final-native-timing-batch-01/execution.json'],
      5:['docs/plans/pressure-methods-v6/scene-reservation.json','runs/pressure-methods-v6/blind-capture-batch-01/execution.json','runs/pressure-methods-v6/long-rollout-batch-01/execution.json','runs/pressure-methods-v6/long-drift-batch-01/execution.json']}
    completion=[]
    for track in campaign['tracks']:
        paths=shared+proof_groups[track['id']]
        paths+= [p for p in evidence if ('/development/'in p or '/blind-evaluation/'in p or '/long-drift/'in p or '/transfer/'in p or '/managed-runs/'in p or '/train/'in p or '/restore/'in p)]
        if track['id']==3:paths += [p for p in evidence if any(k in p for k in ['qualification','fine-tuning.json','config.json'])]
        require(all(p in evidence for p in paths),'Completion map cites uninspected evidence')
        completion.append(dict(id=track['id'],name=track['name'],status='proven',requirements=[dict(requirement=r,status='proven',evidence={p:evidence[p]for p in paths})for r in track['required']]))
    write(DOCS/'final-audit.json',dict(status='passed',successful_fits=4,successful_managed_runs=8,successful_optimizer_updates=9600,discarded_partial_attempt=failed,
      fits=fits,managed_runs=managed,independent_tests=tests,evidence=evidence,source_sha256=sha(__file__),checked_unix=time.time(),live_owned_workers=[],host_namespace_process_audit=True,promotion=False))
    write(DOCS/'completion-audit.json',dict(status='complete',requirements=completion,final_audit_sha256=sha(DOCS/'final-audit.json'),
      limitations=['Diagnostic application limits only','One seed per fitted contrast','128³ training maximum','256³ timing is a bounded4-step prefix','No512³/1024³ production qualification','No application promotion']))
    print('All five requested tracks verified against current retained evidence',flush=True)
if __name__=='__main__':
    import math
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'final-audit-01');run(p.parse_args().output)
