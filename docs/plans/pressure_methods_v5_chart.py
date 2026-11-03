"""Import the sealed lightweight-correction and timestep-reuse campaign."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

STUDY = 'pressure-methods-v5'
ARMS = ['v5-raw', 'v5-multilevel', 'v5-spectrum']
IDENTIFIERS = dict(zip(ARMS, ['C112', 'C113', 'C114']))


def completed_methods_v5(base, lab, source_lab):
    base, lab = Path(base), Path(lab)
    path = base / STUDY / 'results.json'
    if not path.exists():
        return [], None
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = read(path)
    if report['status'] != 'complete':
        return [], None
    publication = read(base/STUDY/'publication.json')
    final, completion = read(base/STUDY/'final-audit.json'), read(base/STUDY/'completion-audit.json')
    if (publication['status'] != 'complete' or publication['promotion'] or report['promotion']
            or report['successful_fits'] != 3 or final['status'] != 'passed' or completion['status'] != 'complete'
            or final['successful_fits'] != 3 or final['successful_managed_runs'] != 6
            or final['optimizer_updates'] != 7200 or final['live_owned_workers'] or final['promotion']
            or completion['final_audit_sha256'] != sha(base/STUDY/'final-audit.json')
            or any(r['state'] != 'completed' or r['custody_phase'] != 'drained' for r in final['managed_runs'])
            or sha(lab/'experiments'/STUDY/'report.py') != report['source_sha256']):
        raise ValueError('Completed fifth-campaign fits, restore and custody required')
    for source, expected in {**publication['artifacts'], **publication['sources'], **report['evidence'], **final['evidence']}.items():
        if sha(lab/source) != expected:
            raise ValueError(f'Fifth-campaign evidence changed: {source}')
    fits = {f['arm']:f for f in final['fits']}
    if set(fits) != set(ARMS):
        raise ValueError('All three fifth-campaign fits required')
    models = []
    for arm in ARMS:
        entry_path = lab/'runs'/STUDY/'entries'/(arm+'.json')
        entry = read(entry_path)
        project, artifacts = lab/entry['project'], lab/entry['restore']
        receipt_path = artifacts.parent/'receipt.json'
        receipt, restore = read(receipt_path), read(artifacts/'receipt.json')
        history, objective = read(artifacts/'history.json'), read(project/'fine-tuning.json')
        fit = fits[arm]
        if (receipt['status'] != 'completed' or receipt['optimizer_updates'] != 2400
                or not all(receipt['artifact_parity'].values()) or len(receipt['artifact_parity']) != 5
                or receipt['restore_receipt'] != restore or restore['mode'] != 'restore'
                or restore['optimizer_updates'] or restore['execution']['cuda_initialized']
                or sha(artifacts/'weights.npz') != entry['weights_sha256']
                or history['optimizer_work'] != fit['optimizer_work']
                or history['optimizer_work']['network_forwards'] != 2400 or fit['parameters'] != 648
                or sha(artifacts/'checkpoint-000.npz') != objective['initial_weights_sha256']):
            raise ValueError(f'Incomplete fifth-campaign checkpoint: {arm}')
        for source, expected in entry['identities'].items():
            if sha(lab/source) != expected:
                raise ValueError(f'Fifth-campaign checkpoint identity changed: {source}')
        for source, expected in read(project/'source-witness.json')['sources'].items():
            if sha(project/source) != expected:
                raise ValueError(f'Captured fifth-campaign source changed: {arm}/{source}')
        scope = ('Fresh 648-parameter seeded initialization, shared across the three fits; 2,400 optimizer updates '
                 '(1,600 original small-grid draws and 800 larger-tier draws). Available native snapshots and '
                 'independent synthetic spatial patterns are counted separately, not as optimizer frequencies. '
                 'No C53 weights or distillation. Coarse training preprocessing is frozen CPU work.')
        models.append(dict(name=arm, cohort_id=IDENTIFIERS[arm], study_id=STUDY,
            family='Tiny local linear correction', project=os.path.relpath(project,lab),
            artifacts=os.path.relpath(artifacts,lab),weights_sha256=entry['weights_sha256'],
            source_lab=os.path.relpath(source_lab,lab),
            started_at=datetime.fromtimestamp(history['training_started_unix'],timezone.utc).isoformat(),
            validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
            training_source=os.path.relpath(receipt_path,base),training_sha256=sha(receipt_path),
            study_source=os.path.relpath(path,base),study_sha256=sha(path),
            config_source=os.path.relpath(project/'config.json',base),
            initialization_sha256=objective['initial_weights_sha256'],
            initialization_description='Shared seeded tiny initialization; no inherited trained weights',
            network_parameters=648,optimizer_updates=2400,optimizer_examples=2400,
            optimizer_work=history['optimizer_work'],physical_training_stage=True,
            proposal_network_evaluations=1,training_exposure_scope=scope,
            evaluation_source=os.path.relpath(entry_path,base),evaluation_sha256=sha(entry_path)))
    return sorted(models,key=lambda m:m['started_at']), report


def enrich_methods_v5(data, base, lab, load):
    from current_cohort_records import SOURCE_LAB
    models, report = completed_methods_v5(base, lab, SOURCE_LAB)
    if report is None:
        return
    base, lab = Path(base), Path(lab)
    native_root = lab/'runs'/STUDY
    relative = lambda p: os.path.relpath(p,base)
    final = load(base/STUDY/'final-audit.json')
    for name in ['results','publication','completion-audit','selection','model-selection','scene-reservation','timing-reservation']:
        load(base/STUDY/(name+'.json'))
    for source, expected in {**final['evidence'], **report['evidence']}.items():
        if Path(source).suffix == '.json':
            load(lab/source,expected)
        else:
            data['history_sources'][relative(lab/source)] = expected
    known = {p['weights_sha256']:p for p in data['cohort_points']}
    for m in models:
        known[m['weights_sha256']].update(training_stage='Fresh seeded tiny network; native pressure requests'+
            (' and independent spatial-band patterns' if m['name']=='v5-spectrum' else ''),
            policy='One captured local-network proposal; numerical hierarchy and adaptive GCR are excluded from this diagnostic',
            proposal_network_evaluations=1,optimizer_updates=2400)
    def binding(result):
        initial = result['entry'].get('initialization_only',False) or result.get('initialization_control',False)
        numerical = result.get('parameters') == 0 or result.get('neural') is False or result.get('method') == 'spectral'
        point = known.get(result['entry']['weights_sha256'])
        if not initial and not numerical and point is None:
            raise ValueError('Unrecognized trained fifth-campaign model')
        return dict(model=point['id'] if point and not initial and not numerical else None,
                    control_kind='untrained initialization' if initial else 'numerical' if numerical else None)
    def policy(result):
        return (result['method']+' · '+('optimized' if result.get('optimized') else 'original')+
                (' · channels-last' if result.get('channels_last') else '')+
                ' · reuse '+str(result.get('reuse_capacity',0)))
    marker=STUDY+'-chart'
    data['history']=[r for r in data['history'] if r.get('chart_importer')!=marker]
    rows,summaries,quiet=[],[],[]
    for folder,stage in [('evaluation-final','development'),('blind-evaluation','blind')]:
        for path in sorted((native_root/folder).glob('*.json')):
            result=load(path,report['evidence'][os.path.relpath(path,lab)])
            common=dict(**binding(result),identifier=result['identifier'],stage=stage,policy=policy(result),source=relative(path))
            rows.extend(dict(row,**common) for row in result['rows'])
            summaries.extend(dict(row,**common) for row in result['summary'])
            for summary in result['summary']:
                group=[r for r in result['rows'] if r['population']==summary['population']]
                data['history'].append(dict(chart_importer=marker,group='20 · Pressure methods V · solver policies',
                    name=result['identifier'],weights_sha256=None if common['model'] is None else result['entry']['weights_sha256'],
                    source=relative(path),median=statistics.median(r['relative_residual'] for r in group),
                    worst=summary['worst_relative_residual'],qualified=summary['qualified'],cases=summary['cases'],
                    population=stage+' · '+summary['population'],policy=common['policy']+' · adaptive ceiling 8',updates=0))
    for path in sorted((native_root/'quiet-pressure').glob('*.json')):
        result=load(path,report['evidence'][os.path.relpath(path,lab)])
        if result['cpu']!=23 or result['threads']!=1 or result['repeats']!=3 or result['optimizer_updates']:
            raise ValueError('Fifth-campaign serial timing protocol differs')
        residual=max(r['relative_residual'] for r in result['rows'])
        quiet.append(dict(result['summary'],**binding(result),identifier=result['identifier'],policy=policy(result),
            population=result['population'],worst_relative_residual=residual,source=relative(path)))
    reservation=load(base/STUDY/'timing-reservation.json')
    timing=[]
    for row in report['native_timing']:
        scene=next(r for r in reservation['rows'] if r['scene']['counts']==row['counts'])
        receipts=[]
        for rep in range(3):
            path=native_root/'quiet-native'/(scene['trajectory_id']+'-'+row['identifier']+f'-r{rep}')/'receipt.json'
            receipt=load(path,report['evidence'][os.path.relpath(path,lab)])
            if not receipt['native_process']['reaped'] or receipt['native_process']['returncode']!=0:
                raise ValueError('Fifth-campaign timing child not drained')
            receipts.append((path,receipt))
        result=receipts[0][1]
        timing.append(dict(row,**binding(result),policy=policy(dict(result,method=result['policy'])),
            sources=[relative(p) for p,_ in receipts],source=relative(receipts[0][0]),
            complete=row['repeats']==3 and row['accepted_repeats']==3))
    trace_ids={t['id'] for t in data['drift_traces']}
    drift=[]
    build_path=lab/'runs/pressure-methods-v4/streamed-build-01/build-receipt.json'
    for path in sorted((native_root/'drift-blind').glob('*.json')):
        result=load(path,report['evidence'][os.path.relpath(path,lab)])
        ref_path=native_root/'rollouts-blind'/(result['trajectory']+'-spectral')/'receipt.json'
        ai_path=native_root/'rollouts-blind'/(result['trajectory']+'-'+result['policy'])/'receipt.json'
        reference=load(ref_path,result['reference_receipt_sha256'])
        candidate=load(ai_path,result['candidate_receipt_sha256'])
        build=load(build_path,candidate['build_receipt_sha256'])
        if (reference['scene']!=candidate['scene'] or result['status']!='completed'
                or result['application_acceptance_claimed'] or not result['no_fallback']
                or not all(r['native_process']['reaped'] and r['native_process']['returncode']==0 for r in [reference,candidate])
                or len(result['rows'])!=result['matched_accepted_steps']):
            raise ValueError('Fifth-campaign independent drift custody differs')
        point=known[candidate['entry']['weights_sha256']]
        points=[]
        for i,row in enumerate(result['rows']):
            cq,rq=candidate['frames'][i],reference['frames'][i]
            if (row['index']!=i or not cq['advanced'] or not rq['advanced']
                    or row['candidate_quality']!=cq['quality'] or row['reference_quality']!=rq['quality']
                    or not cq['quality']['qualified'] or not rq['quality']['qualified'] or row['time_s']!=(i+1)/60):
                raise ValueError('Drift rows include an unmatched or rejected timestep')
            for field in ['pressure','velocity','tracer']:
                metrics=row[field]
                if metrics['relative_l2'] is None or not math.isfinite(metrics['relative_l2']):
                    raise ValueError('Undefined drift norm must not become zero')
                if metrics['reference_l2'] and not math.isclose(metrics['relative_l2'],metrics['absolute_l2']/metrics['reference_l2'],rel_tol=1e-12):
                    raise ValueError('Independent drift norm ratio differs')
            points.append([i+1,row['time_s'],row['pressure']['relative_l2'],True,cq['quality']['relative_residual']])
        scene=candidate['scene']
        scene_key=hashlib.sha256(json.dumps(dict(scene=scene,binary=build['binary_sha256'],reference='spectral',gauge='mean-free'),sort_keys=True).encode()).hexdigest()[:20]
        trace_id=hashlib.sha256((STUDY+':'+relative(path)).encode()).hexdigest()[:20]
        trace=dict(id=trace_id,scene_key=scene_key,scene=scene,model=point['name'],weights_sha256=point['weights_sha256'],
            policy_label=result['policy']+' · mean-free pressure vs spectral reference',
            policy=dict(learned_proposals_per_pressure_solve=candidate['max_passes'],proposal_network_evaluations=1,
                minimum_network_evaluations=0,adaptive_stopping={'equation_residual':1e-6,'predicted_divergence':1e-7},
                method='GCR + numerical hierarchy + tiny local correction',reuse_capacity=candidate['reuse_capacity'],
                retention=candidate['retention'],optimized=candidate['optimized'],execution=candidate['execution'],
                pressure_gauge='mean-free',reference_solver='spectral',timing_scope='Independent trajectory diagnostic; not single-track timing'),
            points=points,completed=len(points)==result['requested_steps'],hard_stop=any(f['quality']['hard_stop'] for f in candidate['frames']),
            qualified_steps=len(points),source=relative(path),frames_source=relative(path),receipt_source=relative(ai_path),
            reference_receipt_source=relative(ref_path),build_receipt_source=relative(build_path),chart_study=STUDY,
            native_binary_sha256=build['binary_sha256'],recording_format='independent-full-field-v5',pressure_gauge='mean-free')
        if trace_id not in trace_ids:
            data['drift_traces'].append(trace);trace_ids.add(trace_id)
        drift.append(dict(model=point['id'],policy=result['policy'],source=relative(path),scene_key=scene_key,steps=len(points),time_s=points[-1][1]))
    data['methods_v5']=dict(id=STUDY,source=STUDY+'/results.json',models=[m['cohort_id'] for m in models],
        controls=sorted({r['model'] for r in quiet if r['model'] and r['model'] not in IDENTIFIERS.values()}),
        rows=rows,summaries=summaries,quiet=quiet,timing=timing,drift=drift,
        training=[dict(model=m['cohort_id'],name=m['name'],parameters=648,optimizer_updates=2400,
            optimizer_work=m['optimizer_work'],teacher_work={},initialization_description=m['initialization_description']) for m in models])
