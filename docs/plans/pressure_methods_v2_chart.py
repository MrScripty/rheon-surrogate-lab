"""Read the sealed second pressure-method campaign for the offline dashboard."""
from datetime import datetime, timezone
import hashlib
import json
import os
import statistics
from pathlib import Path

STUDY = 'pressure-methods-v2'
ARMS = ['v2-small24', 'v2-mixed16', 'v2-retain24', 'v2-headbase', 'v2-headshape',
        'v2-num4', 'v2-num6', 'v2-headshape-s25', 'v2-headshape-s26', 'v2-num6-s25', 'v2-num6-s26']
IDENTIFIERS = dict(zip(ARMS, (f'C{n}' for n in range(89, 100))))


def supplemental_drift(base, lab, load, known):
    """Admit chart-owned C98 measurements separately from the sealed campaign."""
    directory = Path(base) / 'chart-diagnostics/pressure-methods-v2/c98-drift-01'
    path = directory / 'manifest.json'
    if not path.exists():
        path = directory / 'short-manifest.json'
        if not path.exists():
            return []
    manifest = load(path)
    specification = load(directory / 'specification.json', manifest['specification_sha256'])
    weight = specification['weights_sha256']
    point = known[weight]
    if (manifest['schema'] != 'chart-supplemental-drift-v1' or manifest['status'] != 'completed'
            or manifest['model'] != 'C98' or point['id'] != 'C98'
            or manifest['weights_sha256'] != weight or not manifest['owned_children_reaped']
            or specification['kind'] != 'neural' or specification['passes'] != 7):
        raise ValueError('C98 supplemental drift identity/custody differs')
    source = directory / 'controller-source.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != specification['source_sha256']:
        raise ValueError('C98 diagnostic controller changed')
    evaluation_path = (Path(base) / specification['evaluation']).resolve()
    evaluation = load(evaluation_path, specification['evaluation_sha256'])
    if evaluation['entry']['weights_sha256'] != weight:
        raise ValueError('C98 diagnostic uses a different checkpoint')
    build = load(Path(base) / specification['build'] / 'build-receipt.json', specification['build_receipt_sha256'])
    expected = {(0,512), (1,512), (4,512)}
    if path.name == 'manifest.json':
        expected.add((0,7200))
    observed = {(r['scene_id'], r['steps']) for r in manifest['drift']}
    if len(manifest['drift']) != len(expected) or observed != expected or build['status'] != 'qualified_build':
        raise ValueError('C98 diagnostic scene coverage differs')
    rows = []
    for row in manifest['drift']:
        result = load(Path(lab) / row['path'], row['sha256'])
        reservation = load(Path(base) / row['reservation'], result['reservation_sha256'])
        reserved = next(r for r in reservation['scenes'] if r['id'] == row['scene_id'])
        scene = load(Path(lab) / reserved['path'], reserved['sha256'])
        if row['steps'] > scene['steps']:
            raise ValueError('C98 diagnostic extends beyond reserved horizon')
        scene = dict(scene, steps=row['steps'])
        comparison = Path(lab) / 'runs/pressure-methods-v2' / (
            'finalist-native-long-01' if row['steps'] == 7200 else 'finalist-native-01') / (
            f'v2-num6-scene{row["scene_id"]}-7/results.json')
        matched = load(comparison)
        if (result['scene'] != scene or scene != matched['scene']
                or result['kind'] != 'neural' or result['passes'] != 7
                or Path(result['evaluation']).resolve() != evaluation_path
                or result['source_sha256'] != specification['runner_sha256']
                or result['build_receipt_sha256'] != specification['build_receipt_sha256']
                or not row['reaped'] or row['returncode'] != 0):
            raise ValueError('C98 diagnostic differs from the matched C95 protocol')
        rows.append(dict(path=row['path'], status=result['status'], supplemental=True,
                         matched_source=os.path.relpath(comparison, base)))
    return rows


def family(arm):
    if arm.startswith('v2-headshape'):
        return 'Shape-informed coarse correction'
    if arm.startswith('v2-num'):
        return 'Numerical coarse correction'
    if arm == 'v2-headbase':
        return 'Learned coarse correction'
    return 'Training domain expansion'


def completed_methods_v2(base, lab, source_lab):
    base, lab = Path(base), Path(lab)
    report_path = base / STUDY / 'results.json'
    if not report_path.exists():
        return [], None
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = read(report_path)
    if report['status'] != 'completed':
        return [], None
    if (report['missing'] or report['promoted'] or report['application_acceptance_claimed']
            or report['successful_fits'] != 12 or len(set(report['managed_run_ids'])) != 24
            or any(not c['drained'] or c['status'] != 'completed' for c in report['custody'])):
        raise ValueError('Completed second-campaign custody required')
    if sha(lab / 'experiments' / STUDY / 'build_report.py') != report['source_sha256']:
        raise ValueError('Second-campaign report source changed')
    for source, expected in report['evidence'].items():
        if sha(lab / source) != expected:
            raise ValueError(f'Second-campaign evidence changed: {source}')
    rows = {r['arm']: r for r in report['training'] if r['arm'] != 'C53'}
    if set(rows) != set(ARMS) | {'v2-small16'}:
        raise ValueError('All twelve second-campaign fits required')
    if rows['v2-small16']['weights_sha256'] != read(lab / 'runs/pressure-methods-v1/domain-small-control-study-01/receipt.json')['weights_sha256']:
        raise ValueError('Repeated small control no longer matches C78')
    models = []
    for arm, row in rows.items():
        repeat = row['repeat']
        receipt_path = lab / 'runs' / STUDY / (arm + ('-study-01' if repeat else '-study-02')) / 'receipt.json'
        evaluation_path = lab / 'runs' / STUDY / ('replicate-evaluation-01' if repeat else 'evaluation-01') / arm / 'results.json'
        receipt, evaluation = read(receipt_path), read(evaluation_path)
        artifacts = receipt_path.parent / 'restore'
        project = lab / evaluation['entry']['project']
        history = read(artifacts / 'history.json')
        objective = read(project / 'fine-tuning.json')
        updates = row['optimizer_updates']
        if (receipt['status'] != 'completed' or updates not in (1600, 2400)
                or receipt['optimizer_updates'] != updates or not all(receipt['artifact_parity'].values())
                or len(receipt['artifact_parity']) != 5
                or receipt['training_receipt']['status'] != 'complete'
                or receipt['restore_receipt']['status'] != 'complete'
                or receipt['restore_receipt']['optimizer_updates'] != 0
                or receipt['restore_receipt']['execution']['cuda_initialized']
                or read(artifacts / 'receipt.json') != receipt['restore_receipt']
                or sha(artifacts / 'weights.npz') != row['weights_sha256']
                or sha(artifacts / 'history.json') != receipt['training_receipt']['outputs']['history.json']['sha256']
                or history['optimizer_work']['network_forwards'] != 2*updates
                or history['optimizer_work']['forward_cells'] != row['fine_forward_cells']
                or evaluation['status'] != 'completed' or evaluation['optimizer_updates']
                or evaluation['blind_scenes_consumed'] or evaluation['measured_cases'] != 156):
            raise ValueError(f'Incomplete second-campaign fit: {arm}')
        for source, expected in evaluation['entry']['identities'].items():
            if sha(lab / source) != expected:
                raise ValueError(f'Second-campaign checkpoint/source changed: {source}')
        for source, expected in read(project / 'source-witness.json')['sources'].items():
            if sha(project / source) != expected:
                raise ValueError(f'Captured second-campaign source changed: {arm}/{source}')
        if arm == 'v2-small16':
            continue
        models.append(dict(name=arm, cohort_id=IDENTIFIERS[arm], study_id=STUDY, family=family(arm),
            project=os.path.relpath(project, lab), artifacts=os.path.relpath(artifacts, lab),
            weights_sha256=row['weights_sha256'], source_lab=os.path.relpath(source_lab, lab),
            started_at=datetime.fromtimestamp(history['training_started_unix'], timezone.utc).isoformat(),
            validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
            training_source=os.path.relpath(receipt_path, base), training_sha256=sha(receipt_path),
            study_source=os.path.relpath(report_path, base), study_sha256=sha(report_path),
            config_source=os.path.relpath(project / 'config.json', base),
            initialized_from='C53', initialization_sha256=objective['initial_weights_sha256'],
            optimizer_updates=updates, optimizer_examples=updates, optimizer_work=history['optimizer_work'],
            physical_training_stage=True, sampling_repeat=repeat,
            evaluation_source=os.path.relpath(evaluation_path, base), evaluation_sha256=sha(evaluation_path)))
    return sorted(models, key=lambda m: m['started_at']), report


def enrich_methods_v2(data, base, lab, load):
    """Keep development, blind, quiet-policy and application scopes separate."""
    from current_cohort_records import SOURCE_LAB
    models, report = completed_methods_v2(base, lab, SOURCE_LAB)
    if report is None:
        return
    rel = lambda p: os.path.relpath(p, base)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    load(base / STUDY / 'results.json')
    for source, expected in report['evidence'].items():
        load(lab / source, expected)
    known = {p['weights_sha256']: p for p in data['cohort_points']}
    by_id = {p['id']: p for p in data['cohort_points']}
    arm_ids = {**IDENTIFIERS, 'v2-small16': 'C78', 'C53': 'C53'}
    summaries, evaluation_sources = [], []
    for row in report['training']:
        arm = row['arm']
        if arm == 'C53':
            path = lab / 'runs' / STUDY / 'evaluation-01/C53-extended/results.json'
        else:
            path = lab / 'runs' / STUDY / ('replicate-evaluation-01' if row['repeat'] else 'evaluation-01') / arm / 'results.json'
        evaluation = load(path)
        evaluation_sources.append(rel(path))
        for summary in evaluation['summary']:
            summaries.append(dict(summary, passes=summary.get('passes', summary.get('policy', {}).get('passes')),
                                  model=arm_ids[arm], arm=arm, source=rel(path)))
    quiet = []
    for row in report['chart']:
        if row['label'].startswith('Full spectral') or row['label'].startswith('Numerical two-level'):
            model, source = None, lab / 'runs' / STUDY / 'quiet-confirmation-01/results.json'
        elif row['label'].endswith(' adaptive'):
            arm = row['label'].removesuffix(' adaptive')
            model = arm_ids[arm]
            source = lab / 'runs' / STUDY / ('quiet-small-adaptive-01/results.json' if arm == 'v2-small16' else 'quiet-confirmation-01/results.json')
        else:
            arm = 'C53' if row['label'].startswith('C53 extended') else row['label'].split()[0]
            model, source = arm_ids[arm], lab / 'runs' / STUDY / 'quiet-assessment-01' / arm / 'results.json'
        quiet.append(dict(row, model=model, source=rel(source)))
    blind = []
    for row in report['blind']:
        source = lab / 'runs' / STUDY / 'blind-assessment-01' / row['arm'] / 'results.json'
        blind.append(dict(row, model=arm_ids[row['arm']], source=rel(source)))
    application = []
    for row in report['application']:
        method = row['method']
        arm = method.removesuffix('-adaptive')
        application.append(dict(row, model=arm_ids.get(arm),
            source=rel(lab / 'runs' / STUDY / 'single-final-quiet-01' / f'n{row["size"]}-{method}-rep0/results.json')))
    scaling = []
    for row in report['scaling']:
        model = (next((arm_ids[arm] for arm in ['v2-headshape','v2-num6','C53'] if arm in row['case']), None)
                 if row['method'] in ('fine','learned-coarse','numerical-coarse') else None)
        scaling.append(dict(row, model=model, source=rel(lab / row['path'])))
    trace_ids = {t['id'] for t in data['drift_traces']}
    native_summary = []
    for row in report['drift'] + supplemental_drift(base, lab, load, known):
        path = lab / row['path']
        result = load(path)
        native = path.parent / 'native'
        receipt = load(native / 'receipt.json')
        if row.get('supplemental'):
            build = load(lab / 'runs/pressure-methods-v2/harder-drift-build-02/build-receipt.json',
                         result['build_receipt_sha256'])
            if (receipt['binary_sha256'] != build['binary_sha256']
                    or receipt['learned_proposals_per_pressure_solve'] != 7
                    or receipt['cuda_initialized'] or receipt['cpu_threads'] != 1):
                raise ValueError('C98 native build/policy changed')
        manifest = (load(Path(receipt['adapter_manifest']), receipt['adapter_manifest_sha256'])
                    if receipt['adapter_manifest'] else {})
        if manifest and sha(path.parent / 'model_adapter.py') != manifest['adapter_sha256']:
            raise ValueError('Second-campaign native adapter changed')
        if row.get('supplemental') and (
                manifest.get('kind') != result['kind'] or manifest.get('passes') != result['passes']
                or manifest.get('evaluation') != result['evaluation']
                or manifest.get('adapter_sha256') != result['adapter_sha256']):
            raise ValueError('C98 diagnostic adapter binding differs')
        evaluation_path = result.get('evaluation')
        if evaluation_path:
            evaluation = load(Path(evaluation_path), manifest['evaluation_sha256'])
            weight = evaluation['entry']['weights_sha256']
        else:
            material = load(lab / 'runs/pressure-pass-sweep-v1/cpu-01/material.json')
            weight = next(m['weights_sha256'] for m in material['models'] if m['id'] == 'C53')
        point = known[weight]
        raw_path = native / 'frames.jsonl'
        if (sha(raw_path) != result['frames_sha256'] or not result['native_process']['reaped']
                or result['native_process']['returncode'] != 0 or receipt['scene'] != result['scene']):
            raise ValueError(f'Second-campaign native custody changed: {path}')
        data['history_sources'][rel(raw_path)] = result['frames_sha256']
        frames = [json.loads(line) for line in raw_path.read_text().splitlines()]
        ai = [f for f in frames if f.get('ai_quality')]
        if len(ai) != len(result['points']) or len(ai) != result['completed_post_warmup_steps']:
            raise ValueError(f'Second-campaign native population changed: {path}')
        kind = result['kind']
        numerical = kind in ('native','spectral')
        label = ('Rheon native control' if kind == 'native' else 'Full spectral numerical control' if kind == 'spectral'
                 else f'{kind} · {result["passes"]} neural passes' +
                 (f' · numerical every {result["interval"]} AI steps' if kind == 'periodic' else ''))
        native_summary.append(dict(row, model=point['id'], policy_label=label, source=rel(path)))
        if not ai:
            continue
        points = []
        for frame, recorded in zip(ai, result['points']):
            if (frame['step'] != recorded['step'] or frame['time_s'] != recorded['time_s']
                    or frame['relative_pressure_difference'] != recorded['pressure_trajectory_relative_l2']
                    or frame['ai_quality']['relative_residual'] != recorded['equation_residual']
                    or not frame['reference_quality']['qualified']
                    or (row.get('supplemental') and not frame['ai_quality']['qualified'])):
                raise ValueError(f'Second-campaign native point changed: {path}')
            points.append([frame['step'], frame['time_s'], frame['relative_pressure_difference'],
                           frame['ai_quality']['qualified'], frame['ai_quality']['relative_residual']])
        adaptive = kind in ('triggered','cost-hybrid')
        policy = dict(learned_proposals_per_pressure_solve=0 if numerical else result['passes'],
            adaptive_stopping={'equation_residual':1e-6,'predicted_divergence':1e-7} if adaptive else None,
            proposal_network_evaluations=1, minimum_network_evaluations=0, method=kind,
            numerical_interval=result['interval'] if kind == 'periodic' else None,
            coarse_nodes_per_axis=6 if kind == 'cost-hybrid' else None,
            captured_project=point.get('replay_project'), timing_scope=result['timing_scope'])
        scene_key = hashlib.sha256(json.dumps(dict(scene=result['scene'], binary=receipt['binary_sha256']), sort_keys=True).encode()).hexdigest()[:20]
        trace_id = hashlib.sha256((STUDY + ':' + rel(path)).encode()).hexdigest()[:20]
        if trace_id not in trace_ids:
            data['drift_traces'].append(dict(id=trace_id, scene_key=scene_key, scene=result['scene'],
                model=point['name'], policy_label=label, weights_sha256=weight, policy=policy, points=points,
                completed=result['status'] == 'completed', hard_stop=receipt['result']['hard_stop'],
                qualified_steps=sum(p[3] for p in points), source=rel(path), frames_source=rel(raw_path),
                receipt_source=rel(native / 'receipt.json'), chart_study=STUDY, native_binary_sha256=receipt['binary_sha256']))
            trace_ids.add(trace_id)
    data['methods_v2'] = dict(id=STUDY, source=STUDY+'/results.json', models=[m['cohort_id'] for m in models],
        controls=['C53','C78'], arm_ids=arm_ids, summaries=summaries, quiet=quiet, blind=blind,
        application=application, scaling=scaling, native=native_summary, evaluation_sources=evaluation_sources,
        training=[dict(r, model=arm_ids[r['arm']]) for r in report['training']],
        repeated_control=dict(model='C78', arm='v2-small16'), direction_coverage=report['direction_coverage'])


def enrich_methods_v2_passes(data, base):
    """Append previously unmeasured model/pass pairs, preserving older timings."""
    study, result = data.get('methods_v2'), data.get('pass_sweep')
    if not study or not result:
        return
    known = {p['id']:p for p in data['cohort_points']}
    existing = {(p['model'],p['passes']) for p in result['points']}
    models = {m['id']:m for m in result['models']}
    for identifier in study['controls']+study['models']:
        point = known[identifier]
        if identifier not in models:
            model = dict(id=identifier,name=point['name'],tags=[],weights_sha256=point['weights_sha256'],
                original_worst_error=point['worst_error'],original_inference_median_ms=point['inference_median_ms'])
            result['models'].append(model)
            models[identifier] = model
        models[identifier]['tags'].append('methods-v2-study')
    paths = [base / source for source in study['evaluation_sources'] if 'C53-extended' not in source]
    paths += [base.parent.parent / 'runs' / STUDY / 'quiet-assessment-01' / arm / 'results.json'
              for arm in ['C53','v2-small16','v2-headshape','v2-num6']]
    added = 0
    for path in paths:
        evaluation = json.loads(path.read_text())
        point = next(p for p in known.values() if p['weights_sha256'] == evaluation['entry']['weights_sha256'])
        identifier = point['id']
        for n in sorted({r.get('passes',r.get('policy',{}).get('passes')) for r in evaluation['rows']}):
            if (identifier,n) in existing:
                continue
            rows = [r for r in evaluation['rows'] if r['population']=='consumed128' and r.get('passes',r.get('policy',{}).get('passes'))==n]
            identity = lambda r:(r['trajectory_id'],r.get('native_frame',r.get('frame')),tuple(r['counts_xyz']),r.get('cell_size_m') or 1/r['cells_per_metre'])
            if len(rows)!=128 or sorted(map(identity,rows))!=sorted(map(identity,point['timing_cases'])):
                raise ValueError('Second-campaign pass-sweep population differs')
            errors = [r['relative_residual'] for r in rows]
            timings = [r.get('median_policy_seconds',r.get('policy_seconds')) for r in rows]
            source = os.path.relpath(path,base)
            result['sources'][source] = hashlib.sha256(path.read_bytes()).hexdigest()
            result['points'].append(dict(model=identifier,passes=n,cases=128,worst_error=max(errors),
                median_error=statistics.median(errors),inference_median_ms=statistics.median(timings)*1000,
                inference_best_ms=min(timings)*1000,inference_worst_ms=max(timings)*1000,
                equation_passes=sum(r['qualified'] for r in rows),mean_actual_calls=statistics.mean(r['network_calls'] for r in rows),
                source=source,timing_source=source,study=STUDY))
            existing.add((identifier,n))
            added += 1
    result['passes'] = sorted(set(result['passes']) | {p['passes'] for p in result['points']})
    result['additional_snapshot_settings'] += added
    result['timing_scope'] += ' C89–C99 add saved 2/4/6/8-pass development measurements; seven-pass quiet follow-ups fill missing settings. Existing C53/C78 measurements are retained, and C78 is the exact repeated small control. Quiet matched policy timing and single-track application timing are shown separately on Pressure methods II.'
