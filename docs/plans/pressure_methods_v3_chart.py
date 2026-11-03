"""Import the completed large-grid and reduced-width pressure campaign."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

STUDY = 'pressure-methods-v3'
IDENTIFIERS = {'v3-replay':'C100', 'v3-native':'C101', 'v3-c37half':'C102', 'v3-c44half':'C103'}


def completed_methods_v3(base, lab, source_lab):
    base, lab = Path(base), Path(lab)
    report_path = base / STUDY / 'results.json'
    if not report_path.exists():
        return [], None
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = read(report_path)
    if report['status'] != 'complete':
        return [], None
    publication = read(base / STUDY / 'publication.json')
    final_path, completion_path = base / STUDY / 'final-audit.json', base / STUDY / 'completion-audit.json'
    final, completion = read(final_path), read(completion_path)
    if (publication['status'] != 'complete' or publication['promotion'] or report['promotion']
            or sha(lab / 'experiments' / STUDY / 'report.py') != report['source_sha256']
            or sha(final_path) != report['final_audit_sha256']
            or sha(completion_path) != report['completion_audit_sha256']
            or final['status'] != 'passed' or completion['status'] != 'complete'
            or final['successful_fits'] != 4 or final['successful_managed_runs'] != 8
            or final['live_owned_workers'] or final['promotion'] or final['application_drift_acceptance_claimed']
            or any(r['state'] != 'completed' or r['custody_phase'] != 'drained' for r in final['managed_runs'])):
        raise ValueError('Completed, drained third-campaign evidence required')
    for source, expected in {**publication['artifacts'], **final['evidence']}.items():
        if sha(lab / source) != expected:
            raise ValueError(f'Third-campaign evidence changed: {source}')
    fits = {r['arm']:r for r in report['fits']}
    if set(fits) != set(IDENTIFIERS):
        raise ValueError('All four third-campaign fits required, including negative results')
    models = []
    for arm, fit in fits.items():
        evaluation = next(r for r in report['evaluations'] if r['entry']['id'] == arm)
        evaluation_path = lab / evaluation['path']
        if sha(evaluation_path) != evaluation['sha256']:
            raise ValueError('Third-campaign evaluation changed')
        entry = evaluation['entry']
        project, artifacts = lab / entry['project'], lab / entry['restore']
        receipt_path = lab / fit['path']
        if sha(receipt_path) != fit['sha256']:
            raise ValueError('Third-campaign fit changed')
        receipt, restore = read(receipt_path), read(artifacts / 'receipt.json')
        history, objective = read(artifacts / 'history.json'), read(project / 'fine-tuning.json')
        if (fit['status'] != 'completed' or fit['updates'] != 2400 or receipt['optimizer_updates'] != 2400
                or not all(receipt['artifact_parity'].values()) or len(receipt['artifact_parity']) != 5
                or receipt['restore_receipt'] != restore or restore['mode'] != 'restore'
                or restore['optimizer_updates'] != 0 or restore['execution']['cuda_initialized']
                or sha(artifacts / 'weights.npz') != entry['weights_sha256']
                or history['optimizer_work'] != fit['optimizer_work']
                or history['optimizer_work']['network_forwards'] != 4800
                or evaluation['status'] != 'completed' or evaluation['optimizer_updates'] != 0
                or not evaluation['unchanged_inventory']):
            raise ValueError(f'Incomplete third-campaign checkpoint: {arm}')
        for source, expected in entry['identities'].items():
            if sha(lab / source) != expected:
                raise ValueError(f'Third-campaign checkpoint identity changed: {source}')
        for source, expected in read(project / 'source-witness.json')['sources'].items():
            if sha(project / source) != expected:
                raise ValueError(f'Captured third-campaign source changed: {arm}/{source}')
        half = 'half' in arm
        initialized = 'C37' if arm == 'v3-c37half' else 'C44' if half else 'C93'
        initialization = f'Channel-sliced {initialized} initialization' if half else f'{initialized} initialization'
        scope = (f'Additional 2,400-update physical-request fitting stage only; {initialization}. '
                 'Counts describe available training snapshots, not optimizer draw frequencies. '
                 'The original 1,600 small-sample draws are retained; 800 larger draws balance the available size tiers. '
                 'Inherited training is not counted again; optimizer-forward cell budgets differ between replay and native fits.')
        models.append(dict(name=arm, cohort_id=IDENTIFIERS[arm], study_id=STUDY,
            family='Reduced-width numerical correction' if half else 'Shape-informed coarse correction',
            project=os.path.relpath(project, lab), artifacts=os.path.relpath(artifacts, lab),
            weights_sha256=entry['weights_sha256'], source_lab=os.path.relpath(source_lab, lab),
            started_at=datetime.fromtimestamp(history['training_started_unix'], timezone.utc).isoformat(),
            validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
            training_source=os.path.relpath(receipt_path, base), training_sha256=sha(receipt_path),
            study_source=os.path.relpath(report_path, base), study_sha256=sha(report_path),
            config_source=os.path.relpath(project / 'config.json', base),
            initialized_from=initialized, initialization_sha256=objective['initial_weights_sha256'],
            initialization_description=initialization, network_parameters=evaluation['parameters'],
            optimizer_updates=2400, optimizer_examples=2400, optimizer_work=history['optimizer_work'],
            physical_training_stage=True, training_exposure_scope=scope,
            evaluation_source=os.path.relpath(evaluation_path, base), evaluation_sha256=sha(evaluation_path)))
    return sorted(models, key=lambda m:m['started_at']), report


def enrich_methods_v3(data, base, lab, load):
    """Keep original proposals, adaptive policies and coupled checks distinct."""
    from current_cohort_records import SOURCE_LAB
    models, report = completed_methods_v3(base, lab, SOURCE_LAB)
    if report is None:
        return
    base, lab = Path(base), Path(lab)
    relative = lambda p: os.path.relpath(p, base)
    load(base / STUDY / 'results.json')
    load(base / STUDY / 'publication.json')
    final = load(base / STUDY / 'final-audit.json', report['final_audit_sha256'])
    load(base / STUDY / 'completion-audit.json', report['completion_audit_sha256'])
    for source, expected in final['evidence'].items():
        if Path(source).suffix == '.json':
            load(lab / source, expected)
        else:
            data['history_sources'][relative(lab / source)] = expected
    known = {p['weights_sha256']:p for p in data['cohort_points']}
    for model in models:
        known[model['weights_sha256']]['training_stage'] = (
            'Recorded physical pressure requests; '+model['initialization_description']+
            '; inherited training excluded')
    rows, summaries = [], []
    marker = 'pressure-methods-v3-chart'
    data['history'] = [r for r in data['history'] if r.get('chart_importer') != marker]
    for section in ['evaluations','blind_evaluations']:
        for record in report[section]:
            source = relative(lab / record['path'])
            evaluation = load(lab / record['path'], record['sha256'])
            point = known[evaluation['entry']['weights_sha256']]
            for row in evaluation['rows']:
                rows.append(dict(row, model=point['id'], stage='blind' if section=='blind_evaluations' else 'development',
                                 kind=evaluation['kind'], source=source))
            for summary in evaluation['summary']:
                summary = dict(summary, model=point['id'], stage='blind' if section=='blind_evaluations' else 'development',
                               kind=evaluation['kind'], source=source)
                summaries.append(summary)
                group = [r for r in evaluation['rows'] if r['population']==summary['population'] and r['max_passes']==summary['max_passes']]
                data['history'].append(dict(chart_importer=marker, group='18 · Pressure methods III · adaptive policies',
                    name=point['name'], weights_sha256=point['weights_sha256'], source=source,
                    median=statistics.median(r['relative_residual'] for r in group), worst=summary['worst_relative_residual'],
                    qualified=summary['qualified'], cases=summary['cases'],
                    population=summary['stage']+' · '+summary['population'],
                    policy=f'{evaluation["kind"]} correction · adaptive, up to {summary["max_passes"]} CNN calls',
                    field_worst=summary['worst_pressure_relative_l2'], updates=0))
    quiet = []
    for record in report['quiet_benchmarks']:
        evaluation = load(lab / record['path'], record['sha256'])
        point = known[evaluation['entry']['weights_sha256']]
        quiet.append(dict(evaluation['summary'], identifier=evaluation['identifier'],
            model=None if evaluation['kind']=='spectral' else point['id'], kind=evaluation['kind'],
            source=relative(lab / record['path']), population=evaluation['population']))
    coupled = []
    for path in sorted((lab / 'runs' / STUDY / 'coupled').glob('*/receipt.json')):
        relative_lab = os.path.relpath(path, lab)
        if relative_lab not in final['evidence']:
            continue
        receipt = load(path, final['evidence'][relative_lab])
        if not receipt['native_process']['reaped'] or receipt['native_process']['returncode'] != 0 or receipt['optimizer_updates']:
            raise ValueError('Third-campaign coupled child/source changed')
        entry = receipt.get('entry')
        point = known[entry['weights_sha256']] if entry else None
        frames = receipt['frames']
        advanced = sum(f['advanced'] for f in frames)
        complete = receipt['status']=='completed' and advanced==receipt['scene']['steps']
        coupled.append(dict(model=point['id'] if point else None, identifier=path.parent.name,
            kind=receipt.get('policy','spectral'), counts_xyz=receipt['grid_dimensions'],
            cells_per_metre=receipt['cells_per_metre'], status=receipt['status'], advanced=advanced,
            requested=receipt['scene']['steps'], complete=complete,
            median_frame_ms=1000*statistics.median(f['frame_seconds'] for f in frames),
            max_passes=receipt.get('max_passes',0),
            median_network_calls=statistics.median(c.get('work',{}).get('network_calls',0) for c in receipt['callbacks']),
            source=relative(path)))
    data['methods_v3'] = dict(id=STUDY, source=STUDY+'/results.json', models=[m['cohort_id'] for m in models],
        controls=sorted({s['model'] for s in summaries}-set(IDENTIFIERS.values())), rows=rows,
        summaries=summaries, quiet=quiet, coupled=coupled,
        training=[dict(model=m['cohort_id'], name=m['name'], initialized_from=m['initialized_from'],
            initialization_description=m['initialization_description'], parameters=m['network_parameters'],
            optimizer_updates=m['optimizer_updates'], optimizer_work=m['optimizer_work']) for m in models],
        drift_scope='Four-step coupled checks have no independent reference trajectory; no long-horizon drift curves are inferred.')


def enrich_methods_v3_passes(data, base, study_key='methods_v3', tag='methods-v3-study', timing_note=None, passes=2):
    """Expose measured default two-stage proposals without relabelling adaptive ceilings."""
    study, result = data.get(study_key), data.get('pass_sweep')
    if not study or not result:
        return
    by_id = {p['id']:p for p in data['cohort_points']}
    existing_models = {m['id'] for m in result['models']}
    existing_points = {(p['model'],p['passes']) for p in result['points']}
    added = 0
    for identifier in study['models']:
        point = by_id[identifier]
        cases = [json.loads(line) for line in (Path(base) / point['case_source']).read_text().splitlines()]
        errors = [c['raw_learned']['quality']['relative_residual'] for c in cases]
        if identifier not in existing_models:
            result['models'].append(dict(id=identifier, name=point['name'], tags=[tag],
                weights_sha256=point['weights_sha256'], original_worst_error=point['worst_error'],
                original_inference_median_ms=point['inference_median_ms']))
        if (identifier,passes) in existing_points:
            continue
        result['points'].append(dict(model=identifier, passes=passes, worst_error=point['worst_error'],
            median_error=point['median_error'], inference_median_ms=point['inference_median_ms'],
            inference_best_ms=point['inference_best_ms'], inference_worst_ms=point['inference_worst_ms'],
            cases=point['cases'], equation_passes=sum(error<=1e-6 for error in errors),
            mean_actual_calls=passes, source=point['source'], timing_source=point['timing_source'],
            timing_scope=point['timing_scope']))
        added += 1
    result['additional_snapshot_settings'] += added
    result['timing_scope'] += timing_note or ' C100–C103 add only their measured original two-stage proposals. Adaptive eight/sixteen-call ceilings are shown separately on Pressure methods III and are not fixed-pass measurements.'
