"""Audit the complete six-mechanism exploration against retained current evidence."""
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from v4_support import ROOT,BASE,DOCS,read,write,sha
from common import environment

ARMS=['v4-student75-distill','v4-student75-control','v4-student50-distill','v4-student50-control','v4-replay','v4-gcr','v4-diverse','v4-onpolicy']

def audit():
    if (DOCS/'final-audit.json').exists():raise ValueError('Fresh final audit required')
    evidence={};hashed={}
    def digest(path):
        path=Path(path)
        if path not in hashed:hashed[path]=sha(path)
        return hashed[path]
    def keep(path):
        path=Path(path);evidence[str(path.relative_to(ROOT))]=digest(path);return read(path)
    def require(value,message):
        if not value:raise ValueError(message)
    for p in (ROOT/'experiments/pressure-methods-v4').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    requested=keep(DOCS/'requested-scope.json');require(hashlib.sha256(requested['text'].encode()).hexdigest()==requested['sha256'],'Requested scope changed')
    prior=keep(ROOT/'docs/plans/pressure-methods-v3/publication.json')
    for name,expected in prior['artifacts'].items():require(digest(ROOT/name)==expected,'Prior publication changed: '+name)
    sys.path[:0]=[str(ROOT/'runs/dependency-migration-v1/Tuldok'),str(ROOT/'runs/dependency-migration-v1/lab/experiments/dependency-migration-v1')]
    import agent_api
    from qualify_typed_pressure import OwnerHTTP
    owner=OwnerHTTP(agent_api,'http://127.0.0.1:8091','a7da4a8de53541a2a0e2aad08005bb71',ROOT/'runs/tuldok-performance-v1/private/owner-session.txt')
    remote=BASE/'managed-final-audit-01';remote.mkdir();runs=[];fits=[];small_hashes=set()
    for arm in ARMS:
        generation=2 if arm=='v4-student75-distill'else 1
        project_generation=2 if arm=='v4-gcr'else 1
        project=BASE/(arm+f'-project-{project_generation:02d}');study=BASE/(arm+f'-study-{generation:02d}')
        result=keep(study/'receipt.json');train=keep(study/'train/receipt.json');restore=keep(study/'restore/receipt.json')
        require(result['status']=='completed'and result['optimizer_updates']==2400,'Incomplete planned fit')
        require(train['execution']['device'].startswith('cuda')and train['optimizer_updates']==2400,'Actual CUDA fit required')
        require(restore['mode']=='restore'and restore['optimizer_updates']==0 and not restore['execution']['cuda_initialized'],'Fresh CPU restore required')
        require(train['prediction_inventory']==restore['prediction_inventory'],'Full published prediction hashes differ on fresh CPU restore')
        require(result['assembly_sha256']==digest(project/'assembly.json'),'Training assembly changed')
        for name in ['weights.npz','normalization.json','predictions.npz','history.json','checkpoint-2400.npz']:
            require(result['artifact_parity'][name]and digest(study/'train'/name)==digest(study/'restore'/name),'Artifact parity differs')
        history=keep(study/'train/history.json');inventory=history['checkpoints']['checkpoint-2400.npz']['inventory']
        require(all(v['dtype']=='<f4'for v in inventory.values()),'Float32 checkpoint inventory required')
        parameters=sum(v['bytes']//4 for v in inventory.values())
        expected_parameters=209374 if 'student75'in arm else 93758 if 'student50'in arm else 371214
        require(parameters==expected_parameters,'Actual reduced/full parameter count differs')
        witness=keep(project/'source-witness.json');document=keep(project/'project.json')
        for name,expected in witness['sources'].items():require(digest(project/name)==expected,'Captured model source changed')
        for row in document['data']:require(digest(ROOT/row['path'])==row['sha256'],'Captured input changed')
        require(train['closure']['source_witness_sha256']==digest(project/'source-witness.json'),'Fit source closure differs')
        qualification=keep(BASE/(arm+f'-qualification-{project_generation:02d}.json'))
        qualifier=ROOT/'experiments/pressure-methods-v4'/('qualify_student.py'if 'student'in arm else'qualify_continuation.py')
        if digest(qualifier)!=qualification['source_sha256']:
            archived=BASE/'qualification-sources'/(qualification['source_sha256']+'.py')
            require(digest(archived)==qualification['source_sha256'],'Original qualification source missing')
            evidence[str(archived.relative_to(ROOT))]=digest(archived)
        require(qualification['status']=='passed'and qualification['optimizer_updates']==0 and qualification['finite_largest_grid_backward'],'Objective qualification incomplete')
        require(qualification['project_sha256']==digest(project/'project.json')and qualification['source_witness_sha256']==digest(project/'source-witness.json'),'Qualified Project identity changed')
        require(qualification['small_replay']['matching_updates']==1600 and qualification['small_replay']['exact_sample_subsequence'],'Small-domain exposure changed')
        small_hashes.add(qualification['sampling']['small_subsequence_sha256'])
        cuda=keep(BASE/(arm+f'-cuda-qualification-{project_generation:02d}.json'))
        require(cuda['status']=='passed'and cuda['cuda_initialized']and cuda['optimizer_updates']==0,'Actual zero-update CUDA qualification required')
        require(cuda['project_sha256']==digest(project/'project.json')and cuda['source_witness_sha256']==digest(project/'source-witness.json'),'CUDA-qualified Project identity changed')
        require(cuda['source_sha256']==digest(qualifier),'CUDA qualification source changed')
        if arm=='v4-gcr':require(cuda['zero_equation_backward']['network_forwards']==0 and not cuda['zero_equation_backward']['nonfinite_gradients'],'GCR CUDA zero bypass failed')
        transfer=keep(BASE/'transfer'/(arm+'.json'))
        require(transfer['status']=='passed'and {r['frame']for r in transfer['rows']}=={0,15},'Full128 cold/warm transfer missing')
        require(all(r['raw_forward_bitwise_equal']and r['pressure_relative_difference']<=1e-8 for r in transfer['rows']),'Full128 deployment transfer differs')
        if 'student'in arm:
            require(train['frozen_full_teacher_unchanged']and restore['frozen_full_teacher_unchanged'],'Student teacher changed')
            count=train['fine_tuning_statistics']['distillation_teacher_forwards']
            require((count>0)==arm.endswith('distill'),'Distillation/control teacher work differs from declaration')
        for phase,key in [('fit','training_run_id'),('restore','restore_run_id')]:
            current=owner.call('runs.get',dict(run_id=result[key]));write(remote/(current['id']+'.json'),current)
            require(current['state']=='completed'and current['custody_phase']=='drained','Fresh managed custody is not complete')
            runs.append(dict(arm=arm,phase=phase,run_id=current['id'],state=current['state'],custody_phase=current['custody_phase']))
            keep(remote/(current['id']+'.json'))
        fits.append(dict(arm=arm,parameters=parameters,optimizer_updates=2400,artifact_parity=True,optimizer_work=history.get('optimizer_work'),
            teacher_work=train.get('fine_tuning_statistics')))
    require(len(small_hashes)==1,'Paired original small sample subsequences differ')
    for id in ['C37','C44','C53']:
        material=read(ROOT/'runs/pressure-pass-sweep-v1/cpu-01/material.json');entry=next(r for r in material['models']if r['id']==id)
        for name,expected in entry['identities'].items():require(digest(ROOT/name)==expected,'Original named checkpoint changed')
    selection=keep(DOCS/'selection.json');rule=keep(DOCS/'selection-rule.json')
    require(selection['status']=='frozen_before_blind_capture'and selection['selection_rule_sha256']==digest(DOCS/'selection-rule.json'),'Selection changed')
    for name,expected in selection['all_evaluations'].items():require(digest(ROOT/name)==expected,'Selected development evidence changed')
    for p in sorted((BASE/'evaluation').glob('*.json')):
        d=keep(p);require(d['status']=='completed'and len(d['rows'])==174,'Development matrix incomplete')
    require(len(list((BASE/'evaluation').glob('*.json')))==18,'Expected eighteen development operating points')
    blind={r['identifier']for r in selection['selected']+selection['references']}
    require(len(list((BASE/'blind-evaluation').glob('*.json')))==len(blind),'Blind operating-point matrix incomplete')
    for identifier in blind:
        d=keep(BASE/'blind-evaluation'/(identifier+'.json'))
        require(d['status']=='completed'and len(d['rows'])==10 and all(r['population']=='v4-blind'for r in d['rows']),'Blind request population differs')
    reservation=keep(DOCS/'scene-reservation.json');blind_rows=[r for r in reservation['rows']if r['population']=='blind']
    for row in blind_rows:
        directory=BASE/'native-data'/(row['trajectory_id']+'-spectral');d=keep(directory/'receipt.json')
        require(d['status']=='completed'and d['row']==row and d['native_process']['reaped'],'Blind scene capture incomplete')
        require((directory/'receipt.json').stat().st_mtime>(DOCS/'selection.json').stat().st_mtime,'Blind capture preceded frozen selection')
    for name,count in [('diverse-training-view-01',42),('onpolicy-training-view-01',38)]:
        d=keep(BASE/name/'manifest.json')
        require(len(d['parents'])==count and d['retained_v3_parents']==34 and d['exact_full_decoded_population_verified'],'Training expansion lost prior fields')
        require(all(p['split']=='train'for p in d['parents']),'Evaluation data entered fitting')
        require(d['encoded_total_bytes']<=236*1024**2,'Expanded view exceeds bounded input reserve')
        for shard in d['shards']:require(digest(BASE/name/shard['path'])==shard['sha256'],'Expanded input shard changed')
    spectrum=keep(BASE/'training-spectrum-01.json')
    require(spectrum['status']=='completed'and sum(r['qualified_before_fit']is False for r in spectrum['rows'])>0,'Added difficult cold training cases not established')
    for size,steps in [(16,600),(32,300),(64,120),(128,32)]:
        d=keep(BASE/f'drift-development-{size}-01.json');require(d['status']=='completed'and d['matched_accepted_steps']==steps,'Longer baseline rollout evidence incomplete')
    for arm in ['v4-replay','v4-onpolicy']:
        d=keep(BASE/(f'drift-{arm}-64-01.json'));require(d['status']=='completed'and d['matched_accepted_steps']>4,'Matched postfit rollout comparison missing')
    for policy in ['learned','spectral']:
        d=keep(BASE/'native-data'/('v4-early256-development-'+policy)/'receipt.json')
        require(d['row']['scene']['counts']==[256]*3 and d['native_process']['reaped'],'Bounded early256 diagnostic missing')
    for name in ['native-batch-01','student-training-batch-01','full-training-batch-01','baseline-evaluation-batch-02','student-evaluation-batch-01','gcr-evaluation-batch-01','transfer-batch-01','blind-capture-batch-01','blind-evaluation-batch-01','all-timing-batch-02']:
        d=keep(BASE/name/'execution.json');require(d['status']=='completed'and d['owned_child_reaped'],'Owned batch is incomplete: '+name)
    timing=keep(BASE/'all-timing-batch-02/execution.json');require(len(timing['completed'])==82,'Timing command matrix incomplete')
    readiness=keep(BASE/'quiet-start-01.json');admission=keep(BASE/'timing-admission-02.json')
    require(readiness['status']=='ready_for_serial_timing'and admission['readiness_sha256']==digest(BASE/'quiet-start-01.json'),'Timing did not follow completed readiness')
    require(admission['started_unix']>=readiness['checked_unix'],'Timing started before readiness')
    require(keep(BASE/'preaudit-timing-stop-01.json')['status']=='stopped_reaped_excluded','Early timings not excluded')
    require(len(list((BASE/'quiet').glob('*.json')))==19,'Historical timing population incomplete')
    for p in sorted((BASE/'quiet').glob('*.json')):
        d=keep(p);require(d['status']=='completed'and len(d['rows'])==128 and d['cpu']==23 and d['threads']==1 and d['repeats']==3,'Comparable historical timing differs')
    episodes=list((BASE/'coupled-quiet').glob('*/receipt.json'));require(len(episodes)==63,'Native repeated timing matrix incomplete')
    for p in episodes:
        d=keep(p);require(d['native_process']['reaped']and d['scene']['steps']==4 and not d['request_labels_exported']and not d['states'],'Native timing contains label/export work or live child')
        require(all(not f['advanced']or f['quality']['qualified']for f in d['frames']),'Rejected pressure advanced a native trajectory')
        if d['status']!='completed':require(not d['frames'][-1]['advanced'],'Rejected native episode was hidden')
    # Run and retain the short independent algebra/solver tests, after timing.
    runtime=read(ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json');tests=[]
    for name in ['test_multilevel.py','test_preconditioned.py']:
        path=ROOT/'experiments/pressure-methods-v4'/name;log=BASE/(name+'.final.log')
        with log.open('xb')as stream:
            child=subprocess.run([runtime['python'],'-B',str(path)],env=environment(runtime),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        require(child.returncode==0,'Independent algebra/solver test failed');tests.append(dict(source=str(path.relative_to(ROOT)),source_sha256=digest(path),log_sha256=digest(log),exit_code=0))
    green=keep(BASE/'gcr-training-parity-03.json');red=keep(BASE/'gcr-training-red-03.json')
    require(green['status']=='passed'and green['finite_zero_backward']and red['status']=='failed_zero_backward','Real zero-gradient regression evidence missing')
    build=keep(BASE/'streamed-build-01/build-receipt.json')
    require(build['status']=='qualified_build'and all(c['exit_code']==0 for c in build['commands']),'Native source/wire tests or clippy failed')
    for name,expected in build['sources'].items():require(digest(ROOT/name)==expected,'Qualified native source changed')
    require(digest(BASE/'streamed-build-01/rheon-large-single-track')==build['binary_sha256'],'Qualified binary changed')
    for p in (BASE/'native-data').glob('*/receipt.json'):
        d=keep(p);require(d['native_process']['reaped'],'Native capture left a live child')
        for frame in d['snapshots']:
            for spec in frame['arrays'].values():require(digest(p.parent/spec['path'])==spec['sha256'],'Captured native request bytes changed')
        for state in d.get('states',[]):
            require(digest(p.parent/state['path'])==state['sha256'],'Full native trajectory state archive changed')
    figures=keep(DOCS/'figure-evidence.json')
    require(figures['source_sha256']==digest(ROOT/'experiments/pressure-methods-v4/figures.py'),'Figure source changed')
    for name,expected in figures['inputs'].items():require(digest(ROOT/name)==expected,'Figure input changed')
    for name,expected in figures['artifacts'].items():require(digest(ROOT/name)==expected,'Scientific figure changed')
    require(len(figures['artifacts'])>=9,'Standalone scientific figure formats incomplete')
    workers=[];run_ids={r['run_id']for r in runs}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit()or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0');exe=Path(args[0].decode()).name
            if (exe.startswith('python')or exe=='rheon-large-single-track')and(any(b'pressure-methods-v4'in a for a in args)or any(any(r.encode()in a for r in run_ids)for a in args)):
                workers.append(dict(pid=int(p.name),executable=exe))
        except(OSError,UnicodeError,IndexError):pass
    require(not workers,'Live campaign workers remain')
    result=dict(status='passed',successful_fits=8,successful_managed_runs=16,managed_runs=runs,fits=fits,optimizer_updates=19200,
        common_small_subsequence_sha256=next(iter(small_hashes)),independent_tests=tests,evidence=evidence,
        live_owned_workers=workers,host_namespace_process_audit=True,source_sha256=sha(__file__),checked_unix=time.time(),promotion=False,application_drift_acceptance=False)
    write(DOCS/'final-audit.json',result)
    requirements=[dict(item=i,status='proven',name=name)for i,name in enumerate([
        'Size-adaptive multilevel correction with frozen/numerical controls and coupled checks',
        '75% and 50% channel students with frozen-teacher distillation and matched controls',
        'Full-field baseline drift, model-generated state training, and matched postfit rollouts',
        'Flexible GCR controls, independent algebra/breakdown checks, and solver-aware training',
        'Independent scene diversity, retained original population, spatial-spectrum/hard-cold diagnosis, and early256 probe',
        'Execution alternatives, complete policy coverage, repeated matched CPU and native-fluid timing'],1)]
    write(DOCS/'completion-audit.json',dict(status='complete',requirements=requirements,final_audit_sha256=sha(DOCS/'final-audit.json'),source_sha256=sha(__file__),
        outside_this_qualification=['Production promotion','512³/1024³ domains','Application-specific drift acceptance','End-to-end differentiable fluid-rollout training','Independent spatial-wavenumber curriculum'],
        interpretation='Completed controlled exploration includes negative results; model qualification and application acceptance are not inferred from a completed fit.'))
    print('All six exploration requirements audited against current retained evidence',flush=True)

if __name__=='__main__':audit()
