"""Import the completed pressure-method campaign without altering its evidence.

Training IDs denote distinct restored checkpoints. Teacher-coarse exactly
reproduces domain-small-control and remains an alias of that checkpoint.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
import statistics
from pathlib import Path

STUDY = 'pressure-methods-v1'
ARMS = ['domain-small-control', 'domain-expanded', 'teacher-half', 'teacher-quarter',
        'spacing-data-control', 'numerical-coarse', 'sparse-anchors', 'learned-coarse',
        'spacing-zero-feature', 'spacing-conditioned', 'coarse-head-rate']
IDENTIFIERS = dict(zip(ARMS, (f'C{n}' for n in range(78, 89))))
FAMILIES = dict(zip(ARMS, ['Training domain expansion', 'Training domain expansion',
    'Teacher timestep refinement', 'Teacher timestep refinement', 'Spacing training',
    'Numerical coarse correction', 'Sparse numerical anchors', 'Learned coarse correction',
    'Explicit spacing conditioning', 'Explicit spacing conditioning', 'Learned coarse correction']))


def completed_methods_study(base, lab, source_lab):
    base, lab = Path(base), Path(lab)
    report_path = base / STUDY / 'results.json'
    if not report_path.exists():
        return [], None
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(Path(p).read_text())
    report = read(report_path)
    if report['status'] != 'completed_bounded_experiments':
        return [], None
    if report['model_promotion'] or report['untouched_scenes_consumed']:
        raise ValueError('Unexpected pressure-method campaign disposition')
    for path, expected in report['controller_evidence'].items():
        controller = read(lab / path)
        if sha(lab / path) != expected or controller['status'] != 'completed' or controller.get('active'):
            raise ValueError(f'Incomplete pressure-method controller: {path}')
        if any(not r.get('reaped', True) or r.get('returncode', 0) != 0 for r in controller.get('workers', [])):
            raise ValueError(f'Unfinished pressure-method worker: {path}')
    if sha(lab / 'experiments' / STUDY / 'report_campaign.py') != report['source_sha256']:
        raise ValueError('Pressure-method report generator changed')
    rows = {r['arm']: r for r in report['managed_fits']}
    if set(rows) != set(ARMS) | {'teacher-coarse'}:
        raise ValueError('All twelve managed fits required')
    if rows['teacher-coarse']['weights_sha256'] != rows['domain-small-control']['weights_sha256']:
        raise ValueError('Repeated coarse-teacher control no longer reproduces the small control')
    models = []
    for arm, row in rows.items():
        receipt_path = lab / row['receipt']
        receipt = read(receipt_path)
        artifacts = receipt_path.parent / 'restore'
        evaluation_path = lab / row['evaluation']
        evaluation = read(evaluation_path)
        project = lab / evaluation['entry']['project']
        history = read(artifacts / 'history.json')
        objective = read(project / 'fine-tuning.json')
        if (sha(receipt_path) != row['receipt_sha256'] or sha(evaluation_path) != row['evaluation_sha256']
                or receipt['status'] != 'completed' or receipt['optimizer_updates'] != 1600
                or len(receipt['artifact_parity']) != 5 or not all(receipt['artifact_parity'].values())
                or sha(artifacts / 'weights.npz') != row['weights_sha256']
                or read(artifacts / 'receipt.json') != receipt['restore_receipt']
                or receipt['restore_receipt']['execution']['cuda_initialized']
                or receipt['restore_receipt']['optimizer_updates'] != 0
                or sha(artifacts / 'history.json') != receipt['training_receipt']['outputs']['history.json']['sha256']
                or evaluation['status'] != 'completed' or len(evaluation['rows']) != 700
                or evaluation['training_receipt_sha256'] != row['receipt_sha256']
                or evaluation['optimizer_updates'] or evaluation['cuda_initialized'] or evaluation['threads'] != 1):
            raise ValueError(f'Pressure-method fit/evaluation custody differs: {arm}')
        for path, expected in evaluation['entry']['identities'].items():
            if sha(lab / path) != expected:
                raise ValueError(f'Pressure-method checkpoint/source changed: {path}')
        witness = read(project / 'source-witness.json')
        for path, expected in witness['sources'].items():
            if sha(project / path) != expected:
                raise ValueError(f'Captured model source changed: {arm}/{path}')
        if arm == 'teacher-coarse':
            continue
        models.append(dict(name=arm, cohort_id=IDENTIFIERS[arm], study_id=STUDY, family=FAMILIES[arm],
            project=os.path.relpath(project, lab), artifacts=os.path.relpath(artifacts, lab),
            weights_sha256=row['weights_sha256'], source_lab=os.path.relpath(source_lab, lab),
            started_at=datetime.fromtimestamp(history['training_started_unix'], timezone.utc).isoformat(),
            validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
            training_source=os.path.relpath(receipt_path, base), training_sha256=row['receipt_sha256'],
            study_source=os.path.relpath(report_path, base), study_sha256=sha(report_path),
            config_source=os.path.relpath(project / 'config.json', base),
            initialized_from='C53', initialization_sha256=objective['initial_weights_sha256'],
            optimizer_updates=1600, optimizer_examples=1600, optimizer_work=history['optimizer_work'],
            physical_training_stage=True, evaluation_source=os.path.relpath(evaluation_path, base),
            evaluation_sha256=row['evaluation_sha256']))
    return sorted(models, key=lambda m: m['started_at']), report


def enrich_methods_study(data, base, lab, load):
    """Attach sealed population summaries and independent native trajectories."""
    from current_cohort_records import SOURCE_LAB
    models, report = completed_methods_study(base, lab, SOURCE_LAB)
    if report is None:
        return
    rel = lambda p: os.path.relpath(p, base)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report_path = base / STUDY / 'results.json'
    load(report_path)
    known = {p['weights_sha256']: p for p in data['cohort_points']}
    by_id = {p['id']: p for p in data['cohort_points']}
    by_arm = {m['name']: m for m in models}
    by_arm['teacher-coarse'] = by_arm['domain-small-control']
    summaries = []
    for row in report['managed_fits']:
        evaluation = load(lab / row['evaluation'], row['evaluation_sha256'])
        point = known[row['weights_sha256']]
        summaries.extend(dict(model=point['id'], arm=row['arm'], **s, source=rel(lab / row['evaluation']))
                         for s in evaluation['summary'])
    quiet = []
    for row in report['quiet_timing']:
        load(lab / row['source'], row['source_sha256'])
        identifier = IDENTIFIERS.get(row['model'], row['model'])
        quiet.append(dict(row, model=identifier, arm=row['model'], source=rel(lab / row['source'])))
    spacing = []
    for arm, evidence in report['spacing'].items():
        result = load(lab / evidence['source'], evidence['sha256'])
        spacing.extend(dict(s, model=IDENTIFIERS.get(arm, 'C78' if arm == 'teacher-coarse' else arm),
                            arm=arm, source=rel(lab / evidence['source'])) for s in result['summary'])
    trace_ids = {t['id'] for t in data['drift_traces']}
    dispositions = []
    for row in report['native']:
        path = lab / row['results']
        result = load(path, row['results_sha256'])
        native = path.parent / 'native'
        receipt = load(native / 'receipt.json')
        manifest = (load(Path(receipt['adapter_manifest']), receipt['adapter_manifest_sha256'])
                    if receipt['adapter_manifest'] else {})
        raw_path = native / 'frames.jsonl'
        if sha(raw_path) != result['frames_sha256'] or not result['native_process']['reaped']:
            raise ValueError(f'Native methods frames/custody changed: {path}')
        data['history_sources'][rel(raw_path)] = result['frames_sha256']
        frames = [json.loads(line) for line in raw_path.read_text().splitlines()]
        ai = [f for f in frames if f.get('ai_quality')]
        evaluation_source = result.get('fit_evaluation')
        if evaluation_source:
            evaluation = load(Path(evaluation_source), manifest['evaluation_sha256'])
            weight = evaluation['entry']['weights_sha256']
        else:
            material_path = Path(manifest.get('material') or lab / 'runs/pressure-pass-sweep-v1/cpu-01/material.json')
            material = load(material_path, result['material_sha256'])
            weight = next(m['weights_sha256'] for m in material['models'] if m['id'] == result['checkpoint'])
            if weight != by_id[result['checkpoint']]['weights_sha256']:
                raise ValueError('Frozen-policy checkpoint differs from the historical cohort')
        point = known[weight]
        policy_label = result['kind'] + ' · ' + str(result['passes']) + ' neural passes'
        if result['kind'] == 'full-numerical':
            policy_label = 'Full numerical control · 0 neural passes'
        if result['kind'] == 'native-control':
            policy_label = 'Rheon native control · 0 neural passes'
        if result['kind'] == 'periodic':
            policy_label += ' · numerical every ' + str(result['interval']) + ' AI steps'
        dispositions.append(dict(model=point['id'], policy_label=policy_label, source=rel(path), **row))
        if (len(ai) != result['completed_post_warmup_steps'] or len(ai) != len(result['points'])
                or receipt['scene'] != result['scene']
                or any(not f['reference_quality']['qualified'] for f in frames)):
            raise ValueError(f'Native methods trajectory population changed: {path}')
        if not ai:
            continue  # Rejected attempts have no measured drift to plot.
        points = []
        for frame, recorded in zip(ai, result['points']):
            if (frame['step'] != recorded['step'] or frame['time_s'] != recorded['time_s']
                    or frame['relative_pressure_difference'] != recorded['pressure_trajectory_relative_l2']
                    or frame['ai_quality']['relative_residual'] != recorded['equation_residual']):
                raise ValueError(f'Native methods pressure points changed: {path}')
            points.append([frame['step'], frame['time_s'], frame['relative_pressure_difference'],
                           frame['ai_quality']['qualified'], frame['ai_quality']['relative_residual']])
        policy = dict(learned_proposals_per_pressure_solve=0 if result['kind'] in ('full-numerical','native-control') else result['passes'],
            adaptive_stopping=None, proposal_network_evaluations=1, minimum_network_evaluations=0,
            method=result['kind'], numerical_interval=result['interval'] if result['kind'] == 'periodic' else None,
            captured_project=point.get('replay_project'), timing_scope=result['timing_scope'])
        scene_key = hashlib.sha256(json.dumps(dict(scene=result['scene'], binary=receipt['binary_sha256']), sort_keys=True).encode()).hexdigest()[:20]
        trace_id = hashlib.sha256((STUDY + ':' + rel(path)).encode()).hexdigest()[:20]
        if trace_id not in trace_ids:
            data['drift_traces'].append(dict(id=trace_id, scene_key=scene_key, scene=result['scene'],
                model=point['name'], policy_label=policy_label, weights_sha256=weight, policy=policy,
                points=points, completed=result['status'] == 'completed', hard_stop=receipt['result']['hard_stop'],
                qualified_steps=sum(p[3] for p in points), source=rel(path), frames_source=rel(raw_path),
                receipt_source=rel(native / 'receipt.json'), chart_study=STUDY,
                native_binary_sha256=receipt['binary_sha256']))
            trace_ids.add(trace_id)
    data['methods_study'] = dict(id=STUDY, source=rel(report_path), models=[m['cohort_id'] for m in models],
        cohorts=report['cohorts'], summaries=summaries, quiet=quiet, spacing=spacing, native=dispositions,
        repeated_control=dict(model='C78', arm='teacher-coarse'), timing_scope=report['runtime_scope'],
        limits=report['limits'], tolerance=report['declared_equation_tolerance'])


def enrich_methods_passes(data, base):
    """Use the saved 1/2/4/8-pass results; keep quiet timings on their own view."""
    study = data.get('methods_study')
    if not study or not data.get('pass_sweep'):
        return
    result = data['pass_sweep']
    known = {p['id']: p for p in data['cohort_points']}
    for identifier in study['models']:
        point = known[identifier]
        # Summary policies retain the pass count in their explicit policy name.
        summaries = [s for s in study['summaries'] if s['model'] == identifier and s['arm'] != 'teacher-coarse'
                     and s['population'] == 'consumed128' and s['policy'].startswith('neural-')]
        result['models'].append(dict(id=identifier, name=point['name'], tags=['methods-study'],
            weights_sha256=point['weights_sha256'], original_worst_error=point['worst_error'],
            original_inference_median_ms=point['inference_median_ms']))
        for summary in summaries:
            path = base / summary['source']
            snapshot = json.loads(path.read_text())
            passes = int(summary['policy'].split('-')[-1])
            cases = [r for r in snapshot['rows'] if r['population'] == 'consumed128' and r['passes'] == passes]
            identity = lambda r: (r['trajectory_id'], r.get('native_frame', r.get('frame')), tuple(r['counts_xyz']), r['cell_size_m'])
            errors = [r['relative_equation_residual'] for r in cases]
            if (len(cases) != 128 or sorted(map(identity, cases)) != sorted(map(identity, point['timing_cases']))
                    or max(errors) != summary['worst_relative_equation_residual']):
                raise ValueError('Pressure-method pass-sweep population/statistics differ')
            result['sources'][summary['source']] = hashlib.sha256(path.read_bytes()).hexdigest()
            result['points'].append(dict(model=identifier, passes=passes, cases=128,
                worst_error=max(errors), median_error=statistics.median(errors),
                inference_median_ms=summary['median_policy_ms'],
                inference_best_ms=min(r['median_policy_s'] for r in cases)*1000,
                inference_worst_ms=max(r['median_policy_s'] for r in cases)*1000,
                equation_passes=summary['qualified'], mean_actual_calls=statistics.mean(r['network_calls'] for r in cases),
                source=summary['source'], timing_source=summary['source'], study=STUDY))
    result['additional_snapshot_settings'] += 4 * len(study['models'])
    result['timing_scope'] += ' C78–C88 use saved campaign measurements with warm geometry caches, all declared numerical/coarse stages and final residual included. These initial campaign timings are separate from the quiet follow-up on the Pressure methods tab. The teacher-coarse control exactly reproduces C78.'
