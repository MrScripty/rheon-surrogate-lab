"""Read the sealed distillation, solver-aware and data-expansion campaign."""
from datetime import datetime, timezone
import hashlib
import json
import os
import math
from pathlib import Path
import statistics

STUDY = 'pressure-methods-v4'
ARMS = ['v4-student75-distill','v4-student75-control','v4-student50-distill','v4-student50-control',
        'v4-replay','v4-gcr','v4-diverse','v4-onpolicy']
IDENTIFIERS = dict(zip(ARMS, (f'C{i}' for i in range(104,112))))


def completed_methods_v4(base, lab, source_lab):
    base, lab = Path(base), Path(lab)
    path = base / STUDY / 'results.json'
    if not path.exists():
        return [], None
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = read(path)
    if report['status'] != 'complete':
        return [], None
    publication = read(base / STUDY / 'publication.json')
    final, completion = read(base / STUDY / 'final-audit.json'), read(base / STUDY / 'completion-audit.json')
    if (publication['status'] != 'complete' or publication['promotion'] or report['promotion']
            or report['successful_fits'] != 8 or final['status'] != 'passed' or completion['status'] != 'complete'
            or final['successful_fits'] != 8 or final['successful_managed_runs'] != 16
            or final['optimizer_updates'] != 19200 or final['live_owned_workers']
            or final['promotion'] or final['application_drift_acceptance']
            or completion['final_audit_sha256'] != sha(base / STUDY / 'final-audit.json')
            or any(r['state'] != 'completed' or r['custody_phase'] != 'drained' for r in final['managed_runs'])
            or sha(lab / 'experiments' / STUDY / 'report.py') != report['source_sha256']):
        raise ValueError('Completed fourth-campaign fits, restore and custody required')
    for source, expected in {**publication['artifacts'], **publication['sources'], **report['evidence'], **final['evidence']}.items():
        if sha(lab / source) != expected:
            raise ValueError(f'Fourth-campaign evidence changed: {source}')
    fits = {f['arm']:f for f in final['fits']}
    if set(fits) != set(ARMS):
        raise ValueError('All eight fourth-campaign fits required, including controls')
    models = []
    for arm in ARMS:
        entry = read(lab / 'runs' / STUDY / 'entries' / (arm+'.json'))
        project, artifacts = lab / entry['project'], lab / entry['restore']
        receipt_path = artifacts.parent / 'receipt.json'
        receipt, restore = read(receipt_path), read(artifacts / 'receipt.json')
        history, objective = read(artifacts / 'history.json'), read(project / 'fine-tuning.json')
        fit = fits[arm]
        if (receipt['status'] != 'completed' or receipt['optimizer_updates'] != 2400
                or not all(receipt['artifact_parity'].values()) or len(receipt['artifact_parity']) != 5
                or receipt['restore_receipt'] != restore or restore['mode'] != 'restore'
                or restore['optimizer_updates'] or restore['execution']['cuda_initialized']
                or sha(artifacts / 'weights.npz') != entry['weights_sha256']
                or history['optimizer_work'] != fit['optimizer_work']
                or history['optimizer_work']['network_forwards'] != 4800):
            raise ValueError(f'Incomplete fourth-campaign checkpoint: {arm}')
        for source, expected in entry['identities'].items():
            if sha(lab / source) != expected:
                raise ValueError(f'Fourth-campaign checkpoint identity changed: {source}')
        for source, expected in read(project / 'source-witness.json')['sources'].items():
            if sha(project / source) != expected:
                raise ValueError(f'Captured fourth-campaign source changed: {arm}/{source}')
        student = 'student' in arm
        fraction = '75%' if '75' in arm else '50%'
        initialization = fraction+'-width channel-sliced C101 initialization' if student else 'C101 initialization'
        family = (fraction+'-width learned coarse correction' if student else
                  'GCR-aware neural correction' if arm=='v4-gcr' else 'Shape-informed coarse correction')
        scope = (f'Additional 2,400-update physical-request fitting stage only; {initialization}. '
                 'Original 1,600-small-sample stream retained; available snapshots and optimizer draw frequencies are distinct. '
                 'Inherited training is excluded. Distillation teacher work is reported separately from student optimizer work.')
        models.append(dict(name=arm, cohort_id=IDENTIFIERS[arm], study_id=STUDY, family=family,
            project=os.path.relpath(project,lab), artifacts=os.path.relpath(artifacts,lab),
            weights_sha256=entry['weights_sha256'], source_lab=os.path.relpath(source_lab,lab),
            started_at=datetime.fromtimestamp(history['training_started_unix'],timezone.utc).isoformat(),
            validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
            training_source=os.path.relpath(receipt_path,base), training_sha256=sha(receipt_path),
            study_source=os.path.relpath(path,base), study_sha256=sha(path),
            config_source=os.path.relpath(project/'config.json',base), initialized_from='C101',
            initialization_sha256=objective['initial_weights_sha256'], initialization_description=initialization,
            network_parameters=fit['parameters'], optimizer_updates=2400, optimizer_examples=2400,
            optimizer_work=history['optimizer_work'], teacher_work=fit['teacher_work'], physical_training_stage=True,
            training_exposure_scope=scope, evaluation_source=os.path.relpath(lab/'runs'/STUDY/'entries'/(arm+'.json'),base),
            evaluation_sha256=sha(lab/'runs'/STUDY/'entries'/(arm+'.json'))))
    return sorted(models,key=lambda m:m['started_at']), report


def enrich_methods_v4(data, base, lab, load):
    from current_cohort_records import SOURCE_LAB
    models, report = completed_methods_v4(base, lab, SOURCE_LAB)
    if report is None:
        return
    base, lab = Path(base), Path(lab)
    relative = lambda p: os.path.relpath(p,base)
    final = load(base/STUDY/'final-audit.json')
    load(base/STUDY/'results.json')
    load(base/STUDY/'publication.json')
    load(base/STUDY/'completion-audit.json')
    for source, expected in {**final['evidence'], **report['evidence']}.items():
        if Path(source).suffix == '.json':
            load(lab/source,expected)
        else:
            data['history_sources'][relative(lab/source)] = expected
    known = {p['weights_sha256']:p for p in data['cohort_points']}
    for m in models:
        known[m['weights_sha256']]['training_stage'] = 'Recorded physical pressure requests; '+m['initialization_description']+'; inherited training excluded'
    marker = STUDY+'-chart'
    data['history'] = [r for r in data['history'] if r.get('chart_importer') != marker]
    measurements, summaries, quiet = [], [], []
    for folder, stage in [('evaluation','development'),('blind-evaluation','blind')]:
        for path in sorted((lab/'runs'/STUDY/folder).glob('*.json')):
            source_lab = os.path.relpath(path,lab)
            if source_lab not in report['evidence']:
                continue
            result = load(path,report['evidence'][source_lab])
            point = known[result['entry']['weights_sha256']]
            numeric = result['parameters']==0
            model = None if numeric else point['id']
            identifier = path.stem
            for row in result['rows']:
                measurements.append(dict(row,model=model,identifier=identifier,stage=stage,policy=result['policy'],source=relative(path)))
            for row in result['summary']:
                summaries.append(dict(row,model=model,identifier=identifier,stage=stage,policy=result['policy'],source=relative(path)))
                group=[r for r in result['rows'] if r['population']==row['population']]
                data['history'].append(dict(chart_importer=marker,group='19 · Pressure methods IV · solver policies',
                    name=identifier,weights_sha256=None if numeric else point['weights_sha256'],source=relative(path),
                    median=statistics.median(r['relative_residual'] for r in group),worst=row['worst_relative_residual'],
                    qualified=row['qualified'],cases=row['cases'],population=stage+' · '+row['population'],
                    policy=result['policy']+' · adaptive ceiling 8',updates=0))
    for path in sorted((lab/'runs'/STUDY/'quiet').glob('*.json')):
        result=load(path,report['evidence'][os.path.relpath(path,lab)])
        point=known[result['entry']['weights_sha256']]
        quiet.append(dict(result['summary'],identifier=result['identifier'],policy=result['policy'],
                          model=None if result['parameters']==0 else point['id'],source=relative(path)))
    selection=load(base/STUDY/'selection.json')
    by_identifier={r['identifier']:r for r in quiet}
    timing=[]
    reservation=load(base/STUDY/'timing-reservation.json')
    for row in report['native_timing']:
        records=[]
        scene=next(r for r in reservation['rows'] if r['scene']['counts']==row['counts'])
        for rep in range(3):
            path=lab/'runs'/STUDY/'coupled-quiet'/(scene['trajectory_id']+'-'+row['identifier']+f'-r{rep}')/'receipt.json'
            receipt=load(path,report['evidence'][os.path.relpath(path,lab)])
            if not receipt['native_process']['reaped'] or receipt['native_process']['returncode']!=0:
                raise ValueError('Fourth-campaign timing child not drained')
            records.append(relative(path))
        policy=by_identifier.get(row['identifier'])
        model=policy['model'] if policy else 'C101' if row['identifier'].startswith('v3-native-') else None
        method = (policy['policy'] if policy else 'gcr-multilevel' if row['identifier']=='numerical-gcr-multilevel'
                  else 'spectral' if row['identifier']=='spectral' else 'large-channels-last' if row['identifier'].endswith('-layout') else 'learned')
        timing.append(dict(row,model=model,policy=method,sources=records,source=records[0],complete=row['accepted_repeats']==3))
    native_root=lab/'runs'/STUDY
    trace_ids={t['id'] for t in data['drift_traces']}
    drift=[]
    build_path=native_root/'streamed-build-01/build-receipt.json'
    for path in sorted(native_root.glob('drift-*-01.json')):
        source_lab=os.path.relpath(path,lab)
        if source_lab not in report['evidence']:
            continue
        result=load(path,report['evidence'][source_lab])
        ref_path=native_root/'native-data'/(result['trajectory']+'-spectral')/'receipt.json'
        ai_path=native_root/'native-data'/(result['trajectory']+'-'+result['policy'])/'receipt.json'
        reference=load(ref_path,result['reference_receipt_sha256'])
        candidate=load(ai_path,result['candidate_receipt_sha256'])
        build=load(build_path,candidate['build_receipt_sha256'])
        if (reference['scene']!=candidate['scene'] or result['status']!='completed'
                or result['application_acceptance_claimed'] or not result['no_fallback']
                or not all(r['native_process']['reaped'] and r['native_process']['returncode']==0 for r in [reference,candidate])
                or len(result['rows'])!=result['matched_accepted_steps']):
            raise ValueError('Fourth-campaign independent drift custody differs')
        point=known[candidate['entry']['weights_sha256']]
        points=[]
        for i,row in enumerate(result['rows']):
            cq,rq=candidate['frames'][i],reference['frames'][i]
            if (row['index']!=i or not cq['advanced'] or not rq['advanced']
                    or row['candidate_quality']!=cq['quality'] or row['reference_quality']!=rq['quality']
                    or not cq['quality']['qualified'] or not rq['quality']['qualified']
                    or row['time_s']!=(i+1)/60):
                raise ValueError('Drift rows include an unmatched or rejected timestep')
            for field in ['pressure','velocity','tracer']:
                metrics=row[field]
                if metrics['relative_l2'] is None or not math.isfinite(metrics['relative_l2']):
                    raise ValueError('Undefined drift norm must not become a plotted zero')
                if metrics['reference_l2'] and not math.isclose(metrics['relative_l2'],metrics['absolute_l2']/metrics['reference_l2'],rel_tol=1e-12):
                    raise ValueError('Independent drift norm ratio differs')
            points.append([i+1,row['time_s'],row['pressure']['relative_l2'],True,cq['quality']['relative_residual']])
        scene=candidate['scene']
        scene_key=hashlib.sha256(json.dumps(dict(scene=scene,binary=build['binary_sha256'],reference='spectral',gauge='mean-free'),sort_keys=True).encode()).hexdigest()[:20]
        trace_id=hashlib.sha256((STUDY+':'+relative(path)).encode()).hexdigest()[:20]
        trace=dict(id=trace_id,scene_key=scene_key,scene=scene,model=point['name'],weights_sha256=point['weights_sha256'],
            policy_label='Learned · up to 8 CNN calls · mean-free pressure vs spectral reference',
            policy=dict(learned_proposals_per_pressure_solve=8,proposal_network_evaluations=1,minimum_network_evaluations=0,
                adaptive_stopping={'equation_residual':1e-6,'predicted_divergence':1e-7},method=candidate['policy'],
                pressure_gauge='mean-free',reference_solver='spectral',timing_scope='Independent trajectory diagnostic; not single-track timing'),
            points=points,completed=len(points)==result['requested_steps'],hard_stop=any(f['quality']['hard_stop'] for f in candidate['frames']),
            qualified_steps=len(points),source=relative(path),frames_source=relative(path),receipt_source=relative(ai_path),
            reference_receipt_source=relative(ref_path),build_receipt_source=relative(build_path),chart_study=STUDY,
            native_binary_sha256=build['binary_sha256'],recording_format='independent-full-field-v4',pressure_gauge='mean-free')
        if trace_id not in trace_ids:
            data['drift_traces'].append(trace);trace_ids.add(trace_id)
        drift.append(dict(model=point['id'],source=relative(path),scene_key=scene_key,steps=len(points),time_s=points[-1][1]))
    data['methods_v4']=dict(id=STUDY,source=STUDY+'/results.json',models=[m['cohort_id'] for m in models],
        controls=sorted({r['model'] for r in quiet if r['model'] and r['model'] not in IDENTIFIERS.values()}),
        rows=measurements,summaries=summaries,quiet=quiet,timing=timing,drift=drift,
        training=[dict(model=m['cohort_id'],name=m['name'],parameters=m['network_parameters'],initialization_description=m['initialization_description'],
            optimizer_updates=2400,optimizer_work=m['optimizer_work'],teacher_work=m['teacher_work']) for m in models],
        selection=[dict(identifier=r['identifier'],family=r['family']) for r in selection['selected']])
