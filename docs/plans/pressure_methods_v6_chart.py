"""Import the sealed cost-aware correction campaign."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

STUDY = 'pressure-methods-v6'
ARMS = ['v6-replay', 'v6-stencil', 'v6-separable', 'v6-unrolled']
IDENTIFIERS = dict(zip(ARMS, ['C115', 'C116', 'C117', 'C118']))


def completed_methods_v6(base, lab, source_lab):
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
            or report['successful_fits'] != 4 or final['status'] != 'passed' or completion['status'] != 'complete'
            or final['successful_fits'] != 4 or final['successful_managed_runs'] != 8
            or final['successful_optimizer_updates'] != 9600 or final['live_owned_workers'] or final['promotion']
            or completion['final_audit_sha256'] != sha(base/STUDY/'final-audit.json')
            or any(r['state'] != 'completed' or r['custody_phase'] != 'drained' for r in final['managed_runs'])
            or sha(lab/'experiments'/STUDY/'final_report.py') != report['source_sha256']):
        raise ValueError('Completed sixth-campaign fits, restore and custody required')
    for source, expected in {**publication['artifacts'], **publication['sources'], **report['evidence'], **final['evidence']}.items():
        if sha(lab/source) != expected:
            raise ValueError(f'Sixth-campaign evidence changed: {source}')
    fits = {f['arm']:f for f in final['fits']}
    if set(fits) != set(ARMS):
        raise ValueError('All four sixth-campaign fits required')
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
                or history['optimizer_work']['network_forwards'] != (7200 if arm=='v6-unrolled' else 2400)
                or fit['parameters'] != objective['parameters']
                or sha(artifacts/'checkpoint-000.npz') != objective['initial_weights_sha256']):
            raise ValueError(f'Incomplete sixth-campaign checkpoint: {arm}')
        for source, expected in entry['identities'].items():
            if sha(lab/source) != expected:
                raise ValueError(f'Sixth-campaign checkpoint identity changed: {source}')
        for source, expected in read(project/'source-witness.json')['sources'].items():
            if sha(project/source) != expected:
                raise ValueError(f'Captured sixth-campaign source changed: {arm}/{source}')
        inherited = arm in ('v6-replay','v6-unrolled')
        initialization = 'C113 post-coarse checkpoint' if inherited else 'Seeded '+str(fit['parameters'])+'-parameter '+objective['architecture']+' initialization'
        scope = ('Additional 2,400-update solver-aware stage from '+initialization+'. '
                 'The same original 34 native parents (68 snapshots) supply 1,600 small-grid and 800 larger-tier draws; '
                 'no reserved V6 scene enters fitting. Available snapshots are distinct from optimizer draw frequencies. '
                 'Inherited training is excluded; actual network forwards and cells include the declared solver-training iterations.')
        models.append(dict(name=arm, cohort_id=IDENTIFIERS[arm], study_id=STUDY,
            family=('Local stencil correction' if arm=='v6-stencil' else 'Separable local correction' if arm=='v6-separable' else 'Tiny local linear correction'), project=os.path.relpath(project,lab),
            artifacts=os.path.relpath(artifacts,lab),weights_sha256=entry['weights_sha256'],
            source_lab=os.path.relpath(source_lab,lab),
            started_at=datetime.fromtimestamp(history['training_started_unix'],timezone.utc).isoformat(),
            validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
            training_source=os.path.relpath(receipt_path,base),training_sha256=sha(receipt_path),
            study_source=os.path.relpath(path,base),study_sha256=sha(path),
            config_source=os.path.relpath(project/'config.json',base),
            initialization_sha256=objective['initial_weights_sha256'],
            initialization_description=initialization,
            **(dict(initialized_from='C113') if inherited else {}),
            solver_training_iterations=objective['solver_training_iterations'],
            network_parameters=fit['parameters'],optimizer_updates=2400,optimizer_examples=2400,
            optimizer_work=history['optimizer_work'],physical_training_stage=True,
            proposal_network_evaluations=1,training_exposure_scope=scope,
            evaluation_source=os.path.relpath(entry_path,base),evaluation_sha256=sha(entry_path)))
    return sorted(models,key=lambda m:m['started_at']), report



def enrich_methods_v6(data, base, lab, load):
    from current_cohort_records import SOURCE_LAB
    models, report = completed_methods_v6(base, lab, SOURCE_LAB)
    if report is None:
        return
    base, lab = Path(base), Path(lab)
    native_root = lab/'runs'/STUDY
    relative = lambda p: os.path.relpath(p,base)
    final = load(base/STUDY/'final-audit.json')
    for name in ['results','publication','completion-audit','selection','scene-reservation','timing-reservation']:
        load(base/STUDY/(name+'.json'))
    for source, expected in {**final['evidence'], **report['evidence']}.items():
        if Path(source).suffix == '.json':
            load(lab/source,expected)
        else:
            data['history_sources'][relative(lab/source)] = expected
    known = {p['weights_sha256']:p for p in data['cohort_points']}
    for m in models:
        known[m['weights_sha256']].update(training_stage='Solver-aware native-request training; '+m['initialization_description']+'; inherited training excluded',
            policy='One captured local-network proposal; numerical hierarchy and adaptive GCR are excluded from this diagnostic',
            proposal_network_evaluations=1,optimizer_updates=2400,solver_training_iterations=m['solver_training_iterations'])
    def binding(result):
        point = known.get(result['entry']['weights_sha256'])
        control = result.get('parameters') == 0 or result.get('neural') is False or result.get('point',{}).get('numerical',False)
        inactive = bool(result.get('rows')) and not any(r['network_calls'] for r in result['rows']) and not control
        if not control and not inactive and point is None:
            raise ValueError('Unrecognized trained sixth-campaign model')
        return dict(model=point['id'] if point and not control and not inactive else None,
                    control_kind='inactive neural policy' if inactive else 'numerical' if control else None)
    def policy(result):
        config = result.get('point',result)
        return (config.get('method','gcr')+' · '+config.get('variant','all')+' · '+config.get('local_execution','torch')+
                (' · channels-last' if config.get('channels_last') else ''))
    marker=STUDY+'-chart'
    data['history']=[r for r in data['history'] if r.get('chart_importer')!=marker]
    rows,summaries,quiet=[],[],[]
    for folder,stage in [('development','development'),('blind-evaluation','blind')]:
        for path in sorted((native_root/folder).glob('*.json')):
            result=load(path,report['evidence'][os.path.relpath(path,lab)])
            common=dict(**binding(result),identifier=result['identifier'],stage=stage,policy=policy(result),source=relative(path))
            rows.extend(dict(row,**common) for row in result['rows'])
            for population in sorted({r['population'] for r in result['rows']}):
                group=[r for r in result['rows'] if r['population']==population]
                summary=dict(population=population,cases=len(group),qualified=sum(r['qualified'] for r in group),
                    worst_relative_residual=max(r['relative_residual'] for r in group),worst_pressure_relative_l2=max(r['pressure_relative_l2'] for r in group))
                summaries.append(dict(summary,**common))
                data['history'].append(dict(chart_importer=marker,group='21 · Pressure methods VI · solver policies',
                    name=result['identifier'],weights_sha256=None if common['model'] is None else result['entry']['weights_sha256'],
                    source=relative(path),median=statistics.median(r['relative_residual'] for r in group),
                    worst=summary['worst_relative_residual'],qualified=summary['qualified'],cases=summary['cases'],
                    population=stage+' · '+population,policy=common['policy']+' · adaptive ceiling 8',updates=0))
    for path in sorted((native_root/'quiet-development').glob('*.json')):
        result=load(path,report['evidence'][os.path.relpath(path,lab)])
        if result['cpu']!=23 or result['threads']!=1 or result['repeats']!=3 or result['optimizer_updates']:
            raise ValueError('Sixth-campaign serial timing protocol differs')
        quiet.append(dict(result['summary'],**binding(result),identifier=result['identifier'],policy=policy(result),
            population='v6-native20',worst_relative_residual=max(r['relative_residual'] for r in result['rows']),source=relative(path),
            population_sha256=result['population_sha256']))
    reservation=load(base/STUDY/'timing-reservation.json')
    timing=[]
    for row in report['native_timing']:
        scene=next(r for r in reservation['rows'] if r['scene']['counts']==row['counts'])
        receipts=[]
        for rep in range(3):
            path=native_root/'quiet-native'/(scene['trajectory_id']+'-'+row['method']+f'-r{rep}')/'receipt.json'
            receipt=load(path,report['evidence'][os.path.relpath(path,lab)])
            if not receipt['native_process']['reaped'] or receipt['native_process']['returncode']!=0:
                raise ValueError('Sixth-campaign timing child not drained')
            receipts.append((path,receipt))
        result=receipts[0][1]
        timing.append(dict(row,**binding(result),identifier=row['method'],policy=policy(dict(result,point=row['point'])),
            sources=[relative(p) for p,_ in receipts],source=relative(receipts[0][0]),accepted_repeats=row['completed_repeats'],
            complete=row['repeats']==3 and row['completed_repeats']==3))
    trace_ids={t['id'] for t in data['drift_traces']}
    drift,drift_controls=[],[]
    build_path=lab/'runs/pressure-methods-v4/streamed-build-01/build-receipt.json'
    for path in sorted((native_root/'long-drift').glob('*.json')):
        result=load(path,report['evidence'][os.path.relpath(path,lab)])
        ref_path=native_root/'long-rollouts'/result['trajectory']/'receipt.json'
        ai_path=native_root/'long-rollouts'/result['policy']/'receipt.json'
        reference=load(ref_path,result['reference_receipt_sha256'])
        candidate=load(ai_path,result['candidate_receipt_sha256'])
        build=load(build_path,candidate['build_receipt_sha256'])
        if (reference['scene']!=candidate['scene'] or result['status']!='completed'
                or result['application_acceptance_claimed'] or not result['no_fallback']
                or not all(r['native_process']['reaped'] and r['native_process']['returncode']==0 for r in [reference,candidate])
                or len(result['rows'])!=result['matched_accepted_steps']):
            raise ValueError('Sixth-campaign independent drift custody differs')
        point=known[candidate['entry']['weights_sha256']] if candidate['neural'] else None
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
        label=result['policy'].replace(result['trajectory'].removesuffix('-spectral')+'-','')
        if point is None:
            drift_controls.append(dict(model=None,policy=label,scene=scene,scene_key=scene_key,points=points,source=relative(path),pressure_gauge='mean-free'))
            continue
        trace=dict(id=trace_id,scene_key=scene_key,scene=scene,model=point['name'],weights_sha256=point['weights_sha256'],
            policy_label=label+' · mean-free pressure vs spectral reference',
            policy=dict(learned_proposals_per_pressure_solve=candidate['max_passes'],proposal_network_evaluations=1,
                minimum_network_evaluations=0,adaptive_stopping={'equation_residual':1e-6,'predicted_divergence':1e-7},
                method='GCR + numerical hierarchy + local correction',reuse_capacity=candidate['reuse_capacity'],
                adaptive_reuse=candidate['adaptive_reuse'],retention=candidate['retention'],optimized=candidate['optimized'],
                execution=candidate['execution'],local_execution=candidate['local_execution'],variant=candidate['variant'],
                pressure_gauge='mean-free',reference_solver='spectral',timing_scope='Independent trajectory diagnostic; not single-track timing'),
            points=points,completed=len(points)==result['requested_steps'],hard_stop=any(f['quality']['hard_stop'] for f in candidate['frames']),
            qualified_steps=len(points),source=relative(path),frames_source=relative(path),receipt_source=relative(ai_path),
            reference_receipt_source=relative(ref_path),build_receipt_source=relative(build_path),chart_study=STUDY,
            native_binary_sha256=build['binary_sha256'],recording_format='independent-full-field-v6',pressure_gauge='mean-free')
        if trace_id not in trace_ids:
            data['drift_traces'].append(trace);trace_ids.add(trace_id)
        drift.append(dict(model=point['id'],policy=label,source=relative(path),scene_key=scene_key,steps=len(points),time_s=points[-1][1]))
    data['methods_v6']=dict(id=STUDY,source=STUDY+'/results.json',models=[m['cohort_id'] for m in models],
        controls=sorted({r['model'] for r in quiet if r['model'] and r['model'] not in IDENTIFIERS.values()}),
        rows=rows,summaries=summaries,quiet=quiet,timing=timing,drift=drift,drift_controls=drift_controls,
        training=[dict(model=m['cohort_id'],name=m['name'],parameters=m['network_parameters'],optimizer_updates=2400,
            optimizer_work=m['optimizer_work'],teacher_work={},initialization_description=m['initialization_description'],
            solver_training_iterations=m['solver_training_iterations']) for m in models],runtime_breakdown=report['runtime_breakdown'])
