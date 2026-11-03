"""Admit completed detached-correction fits from their frozen campaign evidence.

The C53/U control reproduces C66 exactly. Only eight distinct checkpoints receive
new cohort IDs; the repeated control remains evidence for C66.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

STUDY = 'pressure-detached-correction-v1'
REPORT = 'flow-reasoning-assessment/execution/results.json'
IDENTIFIERS = dict(zip(['c53-a','c53-d','c44-u','c44-a','c44-d','c37-u','c37-a','c37-d'],
                       [f'C{n}' for n in range(70,78)]))
FAMILIES = {'U':'Unmodified fundamental fitting', 'A':'Attached correction feedback',
            'D':'Detached correction feedback'}


def completed_detached_study(base, lab, source_lab):
    base, lab = Path(base), Path(lab)
    report_path = base / REPORT
    if not report_path.is_file():
        return [], None
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(Path(p).read_text())
    report = read(report_path)
    if report['status'] != 'completed':
        return [], None
    campaign_root = lab / 'runs' / STUDY
    campaign = read(campaign_root / 'campaign.json')
    execution = read(campaign_root / 'execution.json')
    evaluation = read(report_path.with_name('evaluation.json'))
    if (execution['status'] != 'training_completed' or len(execution['completed']) != 9
            or evaluation['status'] != 'completed' or len(evaluation['completed']) != 12
            or any(not r['reaped'] or r['returncode'] != 0 for r in evaluation['workers'])
            or report['model_promoted'] or report['unused_holdouts_consumed']):
        raise ValueError('Completed managed fits and drained CPU evaluations required')
    for source, expected in report['evidence'].items():
        if sha(lab / source) != expected:
            raise ValueError(f'Detached-correction evidence changed: {source}')
    rows = {r['key']: r for r in report['rows']}
    control = read(lab / 'runs/pressure-fundamental-bank-fit-v1/shape-isolation-study-01/receipt.json')
    models, snapshots = [], {}
    for arm in campaign['rows']:
        key, mode, initial = arm['key'], arm['mode'], arm['initialization']
        row = rows[key]
        project = lab / arm['project']
        receipt_path = campaign_root / (key + '-study-01/receipt.json')
        receipt = read(receipt_path)
        artifacts = receipt_path.parent / 'restore'
        history = read(artifacts / 'history.json')
        objective = read(project / 'fine-tuning.json')
        bank = receipt['training_receipt']['fine_tuning_statistics']['fundamental_bank']
        statistics = receipt['training_receipt']['fine_tuning_statistics']
        feedback = statistics['rollout_feedback']
        forwards = 6400 if mode == 'U' else 8000
        cells = sum(math.prod(c['counts_xyz']) for c in bank['cases']) * 200 * (4 if mode == 'U' else 5)
        if (row['initialization'] != initial or row['mode'] != mode
                or sha(project / 'assembly.json') != arm['assembly_sha256']
                or sha(lab / arm['config']) != arm['config_sha256']
                or sha(lab / arm['qualification']) != arm['qualification_sha256']
                or receipt['assembly_sha256'] != arm['assembly_sha256']
                or receipt['status'] != 'completed' or receipt['optimizer_updates'] != 1600
                or not all(receipt['artifact_parity'].values())
                or sha(artifacts / 'weights.npz') != receipt['weights_sha256']
                or read(artifacts / 'receipt.json') != receipt['restore_receipt']
                or sha(artifacts / 'history.json') != receipt['training_receipt']['outputs']['history.json']['sha256']
                or row['training']['optimizer_work'] != history['optimizer_work']
                or bank['optimizer_examples'] != 1600 or bank['optimizer_exposures'] != [200]*8
                or bank['expected_exposures'] != [200]*8 or len(bank['cases']) != 8
                or feedback['mode'] != mode or feedback['depth_exposures'] != [[50]*4 for _ in range(8)]
                or feedback['auxiliary_losses'] != (0 if mode == 'U' else 1600)
                or history['optimizer_work']['network_forwards'] != forwards
                or history['optimizer_work']['forward_cells'] != cells
                or objective['rollout_feedback']['mode'] != mode or objective['training_passes'] != 4
                or objective['teacher_weight'] != 0 or statistics['teacher_forwards'] != 0
                or any(history.get(k) is not None for k in ['analytical_feedback','supplemental_feedback','current_feedback'])):
            raise ValueError(f'Detached-correction identity or optimizer exposure differs: {key}')
        for phase in ['training','restore']:
            terminal = read(receipt_path.parent / ('terminal-' + receipt[phase+'_run_id'] + '.json'))
            if terminal['state'] != 'completed' or terminal['custody_phase'] != 'drained':
                raise ValueError(f'Unfinished managed custody: {key}')
        controller = campaign_root / (key + '-controller-' + ('02' if key == 'c53-u' else '01')) / 'execution.json'
        finished = read(controller)
        if finished['status'] != 'completed' or not finished['owned_controller_reaped'] or finished['returncode'] != 0:
            raise ValueError(f'Unreaped training controller: {key}')
        if key == 'c53-u':
            if receipt['weights_sha256'] != control['weights_sha256']:
                raise ValueError('The repeated C66 control no longer matches C66')
            continue
        identifier = IDENTIFIERS[key]
        path = campaign_root / 'cpu-01' / key / 'snapshots.json'
        snapshot = read(path)
        if snapshot['weights_sha256'] != receipt['weights_sha256'] or snapshot['status'] != 'passed':
            raise ValueError(f'Detached-correction snapshot identity differs: {key}')
        snapshots[identifier] = dict(source=os.path.relpath(path,lab),sha256=sha(path))
        models.append(dict(name=key,cohort_id=identifier,study_id=STUDY,family=FAMILIES[mode],
            project=os.path.relpath(project,lab),artifacts=os.path.relpath(artifacts,lab),
            weights_sha256=receipt['weights_sha256'],source_lab=os.path.relpath(source_lab,lab),
            started_at=datetime.fromtimestamp(history['training_started_unix'],timezone.utc).isoformat(),
            validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
            training_source=os.path.relpath(receipt_path,base),training_sha256=sha(receipt_path),
            study_source=REPORT,study_sha256=sha(report_path),config_source=os.path.relpath(project/'config.json',base),
            optimizer_updates=1600,optimizer_examples=1600,optimizer_work=history['optimizer_work'],
            initialized_from=initial,initialization_sha256=objective['initial_weights_sha256'],
            auxiliary_feedback_mode=mode,auxiliary_losses=feedback['auxiliary_losses'],
            manufactured_training_bank=[dict(counts_xyz=c['counts_xyz'],cell_size_m=1/c['cells_per_metre'],
                mode_zyx=c['mode_zyx'],presentations=n)for c,n in zip(bank['cases'],bank['optimizer_exposures'])]))
    if {m['cohort_id'] for m in models} != set(IDENTIFIERS.values()):
        raise ValueError('All eight distinct detached-correction fits required')
    return models,dict(schema='rheon-detached-correction-chart-sources-v1',status='completed',
        training=[dict(id=m['cohort_id'])for m in models],snapshots=snapshots,
        repeated_control=dict(model_id='C66',arm='c53-u',source=REPORT))
