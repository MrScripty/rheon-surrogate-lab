"""Admit finished fits from the ongoing full-AI campaign, per checkpoint.

Assemblies and execution variants are inference policies, not training cohorts.
Temporal full-state experiments retain their own populations and error measures.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from full_ai_execution_chart import append_viscosity_executions,temporal_accuracy_rows
from full_ai_replica_chart import append_replica_evaluations

STUDY = 'full-ai-next-priorities-v1'
ARMS = ['a-control','a-head','a-tail','a-combined','a-redistributed','a-one-stage',
        'a-head-calibration','a-head-calibration-s2','a-head-calibration-s3',
        'a-tail8','a-tail8-s2','a-tail8-s3']
IDENTIFIERS = dict(zip(ARMS, [f'C{i}' for i in range(119,131)]))


def audit_passed(audit):
    if isinstance(audit,bool):
        return audit
    if not isinstance(audit,dict) or not audit:
        return False
    if 'status' in audit and audit['status']!='passed':
        return False
    return all(audit_passed(v) for k,v in audit.items() if k!='status')


def completed_temporal_specs(base, lab, load):
    """Admit completed full-state fits from their sealed development reports.

    Pressure cohort numbers belong to pressure checkpoints. Temporal recipes
    retain their own identifiers, including untimed fitting follow-ups.
    """
    specs = [
        (f'K={k}', f'b{k}-pair-02', f'temporal-development-evaluation-01/b{k}/summary.json')
        for k in (1, 2, 3, 4)
    ] + [
        ('K=1 relative MSE', 'b1-relative-mse-pair-01', 'temporal-mse-development-evaluation-01/summary.json'),
        ('K=1 shared global', 'b1-global-pair-01', 'temporal-global-development-evaluation-01/b1/summary.json'),
    ]
    followups = [
        ('temporal-sharing-results.json', [
            (f'K={k} {architecture}', f'b{k}-{architecture}',
             f'temporal-sharing-development-evaluation-01/b{k}-{architecture}/summary.json')
            for k in (2, 3, 4) for architecture in ('endpoint', 'reuse')]),
        ('temporal-step-scaled-results.json', [
            (f'K=2 {architecture} step-scaled', f'b2-{architecture}-step-scaled',
             f'temporal-step-scaled-development-evaluation-01/b2-{architecture}-step-scaled/summary.json')
            for architecture in ('endpoint', 'reuse')]),
        ('temporal-increment-supervision-results.json', [
            ('K=2 endpoint full-state MSE', 'b2-endpoint-mse-fullstate',
             'temporal-increment-supervision-development-evaluation-01/b2-endpoint-mse-fullstate/summary.json'),
            ('K=2 endpoint increment MSE', 'b2-endpoint-mse-increment',
             'temporal-increment-supervision-development-evaluation-01/b2-endpoint-mse-increment/summary.json')]),
        ('temporal-curl-results.json', [
            ('K=2 endpoint curl increment', 'b2-endpoint-curl-increment',
             'temporal-curl-development-evaluation-01/b2-endpoint-curl-increment/summary.json')]),
        ('temporal-pressure-expert-results.json', [
            ('K=2 pressure expert', 'b2-endpoint-pressure-expert',
             'temporal-pressure-expert-evaluation-01/b2-endpoint-pressure-expert/summary.json')]),
        ('temporal-decoder-results.json', [
            ('K=2 decoder', 'b2-endpoint-decoder',
             'temporal-decoder-evaluation-01/b2-endpoint-decoder/summary.json')]),
        ('temporal-coupled-results.json', [
            ('K=2 coupled', 'b2-coupled',
             'temporal-coupled-evaluation-01/b2-coupled/summary.json')]),
        ('temporal-provisional-results.json', [
            ('K=2 continuation control', 'b2-provisional-control',
             'temporal-provisional-evaluation-01/b2-provisional-control/summary.json'),
            ('K=2 provisional supervision', 'b2-provisional-aux',
             'temporal-provisional-evaluation-01/b2-provisional-aux/summary.json')]),
        ('temporal-displacement-results.json', [
            ('K=2 displacement transport', 'b2-displacement',
             'temporal-displacement-evaluation-01/b2-displacement/summary.json')]),
        ('fixed-viscosity-family-results.json', [
            (f'K={k} fixed viscosity',f'b{k}-fixed-viscosity',
             f'fixed-viscosity-evaluation-01/b{k}-fixed-viscosity/summary.json')
            for k in (1,2,3,4)]),
    ]
    for name, entries in followups:
        path = Path(base) / STUDY / name
        if not path.exists():
            continue
        report = load(path)
        if report['status'] != 'completed':
            continue
        if report.get('model_promotion') or report.get('application_model_promoted') or report.get('reserved_scenes_consumed'):
            raise ValueError('Unexpected temporal follow-up disposition: ' + name)
        audit = report.get('control_audit')
        if audit and not audit_passed(audit):
            raise ValueError('Temporal follow-up control audit failed: ' + name)
        evidence = report.get('source_evidence', report.get('sources', {}))
        for source, expected in evidence.items():
            path=Path(lab)/source
            if path.suffix=='.json':
                load(path,expected)
            elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Temporal follow-up artifact changed: '+source)
        for label, identifier, evaluation in entries:
            source = f'runs/{STUDY}/{evaluation}'
            if source not in evidence:
                raise ValueError('Unbound temporal follow-up evaluation: ' + source)
            specs.append((label, identifier + '-pair-01', evaluation))
    # Admit a finished replica only after its actual evaluation and zero-update
    # wrapper receipt are complete; training progress alone supplies no score.
    root=Path(lab)/'runs'/STUDY
    for pair_path in sorted(root.glob('b*-fixed-viscosity-s*-pair-01/receipt.json')):
        pair=load(pair_path)
        identifier=pair_path.parent.name.removesuffix('-pair-01')
        evaluation=f'fixed-viscosity-replication-evaluation-01/{identifier}/summary.json'
        path=root/evaluation
        wrapper_path=path.parent/'wrapper-receipt.json'
        if pair['status']!='completed' or not path.exists() or not wrapper_path.exists():
            continue
        value=load(path)
        wrapper=load(wrapper_path)
        if value['status']!='completed' or wrapper['status']!='completed':
            continue
        if (wrapper['optimizer_updates'] or not wrapper['training_teacher_disabled']
                or wrapper['core_summary_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError('Replica evaluation wrapper differs: '+identifier)
        k=pair['horizons']
        seed=identifier.rsplit('-s',1)[1]
        specs.append((f'K={k} fixed viscosity seed {seed}',pair_path.parent.name,evaluation))
    return specs


def completed_full_ai(base, lab, source_lab):
    base, lab = Path(base), Path(lab)
    root = lab/'runs'/STUDY
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    models = []
    for arm in ARMS:
        entry_path = root/'entries'/(arm+'.json')
        if not entry_path.exists():
            continue
        entry = read(entry_path)
        project, artifacts = lab/entry['project'], lab/entry['restore']
        pair_path = artifacts.parent/'receipt.json'
        if not pair_path.exists() or read(pair_path)['status'] != 'completed':
            continue
        pair, restore = read(pair_path), read(artifacts/'receipt.json')
        train, history = read(artifacts.parent/'train/receipt.json'), read(artifacts/'history.json')
        config, objective = read(project/'config.json'), read(project/'fine-tuning.json')
        updates = config['training']['updates']
        if (pair['optimizer_updates'] != updates or pair.get('model_promotion')
                or not pair['artifact_parity'] or not all(pair['artifact_parity'].values())
                or restore['mode'] != 'restore' or restore['optimizer_updates']
                or restore['execution']['cuda_initialized'] or train['mode'] != 'train'
                or train['optimizer_updates'] != updates
                or sha(artifacts/'weights.npz') != entry['weights_sha256']):
            raise ValueError(f'Incomplete full-AI checkpoint: {arm}')
        if 'restore_receipt' in pair:
            if pair['restore_receipt'] != restore or pair['training_receipt'] != train:
                raise ValueError('Full-AI pair receipt differs')
        elif (sha(artifacts/'receipt.json') != pair['restore_receipt_sha256']
                or sha(artifacts.parent/'train/receipt.json') != pair['training_receipt_sha256']
                or not restore['full_cpu_probe_bitwise_verified'] or not restore['fine_tensors_bitwise_unchanged']
                or restore['candidate_numerical_pressure_solve']):
            raise ValueError('Full-AI head/tail CPU restore differs')
        for source, expected in entry['identities'].items():
            if sha(lab/source) != expected:
                raise ValueError(f'Full-AI identity changed: {source}')
        for source, expected in read(project/'source-witness.json')['sources'].items():
            if sha(project/source) != expected:
                raise ValueError(f'Captured full-AI source changed: {arm}/{source}')
        for name in pair['artifact_parity']:
            if sha(artifacts/name) != sha(artifacts.parent/'train'/name):
                raise ValueError(f'Full-AI CPU restore parity failed: {arm}/{name}')
        modal = 'head_forwards' in history['optimizer_work']
        tail = arm.startswith('a-tail8')
        architecture = restore['architecture']
        parameters = restore.get('parameters')
        if parameters is None:
            inventory = history['checkpoints'][f'checkpoint-{updates:03d}.npz']['inventory']
            parameters = sum(math.prod(v['shape']) for v in inventory.values())
        family = ('Learned eight-mode tail' if tail else 'Learned head calibration' if modal
                  else 'Compact pressure architecture' if arm in ('a-one-stage','a-redistributed') else 'Full AI pressure continuation')
        init = ('Frozen routed calibrated head/C107 parent plus seeded learned tail' if tail else 'C107 learned pressure checkpoint')
        if (project/'architecture-initialization.json').exists():
            mapped=read(project/'architecture-initialization.json')
            if mapped['mapped_weights_sha256']!=objective['initial_weights_sha256']:
                raise ValueError('Architecture initialization mapping differs')
            init='Architecture-specific tensor mapping from C107; new seeded tensors; no function-equivalence claim'
        scope = (f'Additional {updates:,} updates only; inherited training excluded. '
                 + ('Offline whole-geometry modal teacher labels; no physical simulation frames used. '
                    'Only the learned head is fitted; fine CNN tensors remain bitwise unchanged. '
                    f"{history['optimizer_work']['teacher_examples']:,} sampled teacher examples, not independent simulations."
                    if modal else 'Recorded physical pressure requests; available snapshots differ from balanced optimizer draw frequencies.'))
        models.append(dict(name=arm,cohort_id=IDENTIFIERS[arm],study_id=STUDY,family=family,
            project=entry['project'],artifacts=entry['restore'],weights_sha256=entry['weights_sha256'],
            source_lab=os.path.relpath(source_lab,lab),
            started_at=datetime.fromtimestamp(history['training_started_unix'],timezone.utc).isoformat(),
            validation_inputs='runs/water-early-pressure-v1/control/project/inputs',
            training_source=os.path.relpath(pair_path,base),training_sha256=sha(pair_path),
            study_source=os.path.relpath(entry_path,base),study_sha256=sha(entry_path),
            evaluation_source=os.path.relpath(entry_path,base),evaluation_sha256=sha(entry_path),
            config_source=os.path.relpath(project/'config.json',base),
            initialization_sha256=sha(artifacts/'checkpoint-000.npz'),initialization_description=init,
            network_parameters=parameters,optimizer_updates=updates,
            optimizer_examples=history['optimizer_work'].get('teacher_examples',updates),
            optimizer_work=history['optimizer_work'],physical_training_stage=True,
            modal_teacher_training=modal,captured_pressure_probe=True,
            proposal_network_evaluations=architecture['correction']['steps'],
            training_exposure_scope=scope))
    return sorted(models,key=lambda m:m['started_at'])


def bind_temporal_timing(row, model, base, lab, load, reports):
    """Resolve derived cost points to a witnessed complete-block capture.

    Tensor-only and profiling measurements never substitute for full prediction
    cost. The published cost report can witness timing separately from accuracy.
    """
    if row.get('timing_source'):
        return row
    report_source = row['source']
    if report_source not in reports:
        report = load(lab/report_source)
        if (report['status'] != 'completed' or not report['CPU_latency_claim']
                or report.get('training_on_reserved') or report.get('model_promotion')):
            raise ValueError('Unqualified temporal cost report: '+report_source)
        captures = []
        for source, expected in report['source_evidence'].items():
            target=lab/source
            if target.suffix!='.json':
                if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                    raise ValueError('Temporal cost evidence changed: '+source)
                continue
            value = load(target,expected)
            if Path(source).name == 'summary.json' and value.get('rows'):
                captures.append((source,value))
        reports[report_source] = report,captures
    report,captures = reports[report_source]
    if bool(report.get('reserved_scenes_consumed')) != row['regime'].startswith('reserved_'):
        raise ValueError('Temporal cost report population differs')
    if {k:v for k,v in row.items() if k!='source'} not in report['points']+report.get('comparison',[]):
        raise ValueError('Temporal cost point differs from its report')
    if os.path.relpath(base/model['evaluation_source'],lab) not in report['source_evidence']:
        raise ValueError('Temporal cost accuracy source is not witnessed')
    matches = []
    for source,capture in captures:
        for measured in capture['rows']:
            if (measured['identifier']==row['identifier'] and measured['regime']==row['regime'].removeprefix('reserved_')
                    and measured['variant']=='full' and measured['horizons']==row['horizons']
                    and measured['repeats']>=3
                    and measured['median_block_ms']==row['full_block_ms']
                    and measured['median_ms_per_generated_step']==row['median_ms']):
                matches.append(source)
    if len(matches)!=1:
        raise ValueError('No unique complete-block timing capture: '+row['identifier'])
    return dict(row,timing_source=matches[0],
                accuracy_source=os.path.relpath(base/model['evaluation_source'],lab),execution_variant='full')


def enrich_full_ai(data, base, lab, load):
    from current_cohort_records import SOURCE_LAB
    base, lab = Path(base), Path(lab)
    models = completed_full_ai(base,lab,SOURCE_LAB)
    report_path = base/STUDY/'frontiers.json'
    if not models or not report_path.exists():
        return
    report = load(report_path)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not any(sha(path)==report['source_sha256']
               for path in (lab/'experiments'/STUDY).glob('build_report*.py')):
        raise ValueError('Full-AI report builder changed')
    # Batch execution manifests remain live while the campaign continues. They
    # are status only; numerical evidence is checked against the report snapshot.
    batches = {}
    for source, expected in report['evidence_sha256'].items():
        path = lab/source
        if path.name == 'execution.json':
            value = load(path)
            batches[path.parent.name] = dict(status=value['status'],source=os.path.relpath(path,base))
        elif path.suffix == '.json':
            load(path,expected)
        elif sha(path) != expected:
            raise ValueError('Full-AI report evidence changed: '+source)
    by_name = {m['name']:m['cohort_id'] for m in models}
    temporal, temporal_rows = [], []
    for label, pair_name, evaluation in completed_temporal_specs(base, lab, load):
        pair_root = lab/'runs'/STUDY/pair_name
        pair = load(pair_root/'receipt.json')
        restore = load(pair_root/'restore/receipt.json',pair['restore_receipt_sha256'])
        load(pair_root/'train/receipt.json',pair['training_receipt_sha256'])
        history = load(pair_root/'restore/history.json',restore['outputs']['history.json']['sha256'])
        value = load(lab/'runs'/STUDY/evaluation)
        continuation=pair_name in ('b2-provisional-control-pair-01','b2-provisional-aux-pair-01')
        expected_updates=4000 if continuation else 10000
        fixed_viscosity='-fixed-viscosity' in pair_name
        if fixed_viscosity:
            family_report=load(base/STUDY/'fixed-viscosity-family-results.json')
            registered=next(r for r in family_report['training_work'] if r['horizons']==pair['horizons'])
            expected_updates=registered['optimizer_updates']
        if (value['status'] != 'completed' or pair['status'] != 'completed' or pair['optimizer_updates'] != expected_updates
                or not all(pair['artifact_parity'].values()) or restore['mode'] != 'restore'
                or restore['optimizer_updates'] or restore['execution']['cuda_initialized']
                or not restore['full_cpu_probe_bitwise_verified'] or value['cuda_initialized']
                or value['candidate_numerical_pressure_solve'] or value['optimizer_updates']
                or value['model']['pair_receipt_sha256'] != sha(pair_root/'receipt.json')
                or value['model']['weights_sha256'] != sha(pair_root/'restore/weights.npz')):
            raise ValueError('Temporal fit/restore/evaluation mismatch: '+pair_name)
        for name in pair['artifact_parity']:
            if sha(pair_root/'restore'/name) != sha(pair_root/'train'/name):
                raise ValueError('Temporal CPU restore output mismatch')
        temporal.append(dict(label=label,pair=pair_name,parameters=pair['parameters'],horizons=pair['horizons'],
            weights_sha256=value['model']['weights_sha256'],optimizer_updates=pair['optimizer_updates'],
            trained_at=datetime.fromtimestamp(history['training_started_unix'],timezone.utc).isoformat(),
            identifier=(label if pair_name in [f'b{k}-pair-02' for k in (1,2,3,4)] else pair_name.removesuffix('-pair-01')),
            family=value['model'].get('family','full-state-temporal'),
            architecture=value['model'].get('architecture','original'),bank_sha256=value['bank_sha256'],
            evaluation_source=os.path.relpath(lab/'runs'/STUDY/evaluation,base),
            optimizer_work=history['optimizer_work'],source=os.path.relpath(pair_root/'receipt.json',base)))
        followup = {'b2-endpoint-pressure-expert-pair-01':'temporal-pressure-expert-results.json',
                    'b2-endpoint-decoder-pair-01':'temporal-decoder-results.json'}.get(pair_name)
        if followup:
            context = load(base/STUDY/followup)
            inherited = context['inherited_request_training']
            load(lab/inherited['source_pair']/'receipt.json',inherited['source_pair_receipt_sha256'])
            temporal[-1].update(trainable_parameters=value['model']['trainable_parameters'],
                pressure_expert_calls_per_block=value['model']['pressure_expert_calls_per_block'],
                inherited_request_training=inherited,training_context_source=STUDY+'/'+followup,
                training_scope='10,000 additional full-state updates; inherited factored-request training and frozen learned pressure assets are additional. Architecture and initialization differ from generic controls; no single-factor attribution.')
        if pair_name=='b2-coupled-pair-01' or continuation:
            context_source='temporal-provisional-results.json' if continuation else 'temporal-coupled-results.json'
            context=load(base/STUDY/context_source)
            parent=next((m for m in temporal if m['identifier']=='b2-coupled'),None) if continuation else None
            temporal[-1].update(trainable_parameters=value['model']['trainable_parameters'],
                pressure_expert_calls_per_block=value['model']['pressure_expert_calls_per_block'],
                inherited_full_state_updates=context['parent_updates'] if continuation else 0,
                training_context_source=STUDY+'/'+context_source,
                training_scope=('4,000 additional full-state updates from the shared 10,000-update coupled parent; inherited frozen pressure expert and flux calibration are additional. Control/auxiliary runs share initialization and sampling.' if continuation else
                    '10,000 full-state updates; inherited frozen pressure expert and 1,000-update flux calibration are additional. Architecture and widths differ from decoder; no matched-FLOPs or single-factor attribution.'))
            if continuation:
                if parent is None or context['additional_updates_per_arm']!=expected_updates:
                    raise ValueError('Unbound provisional continuation parent')
                temporal[-1].update(initialized_from=parent['label'],parent_weights_sha256=parent['weights_sha256'])
        if pair_name=='b2-displacement-pair-01':
            context=load(base/STUDY/'temporal-displacement-results.json')
            calibration=load(base/STUDY/'point-sampling-results.json')
            temporal[-1].update(trainable_parameters=value['model']['trainable_parameters'],
                pressure_expert_calls_per_block=value['model']['pressure_expert_calls_per_block'],
                training_context_source=STUDY+'/temporal-displacement-results.json',
                changes_vs_coupled=context['changes_vs_coupled'],
                inherited_point_sampling=dict(parameters=calibration['parameters'],
                    optimizer_updates=calibration['optimizer_updates'],source=STUDY+'/point-sampling-results.json'),
                training_scope='10,000 full-state updates; frozen pressure/flux assets and a 972-parameter point sampler calibrated for 1,000 updates are inherited. Displacement representation, curvature supervision, seed and learning-rate schedule differ from coupled; no single-factor attribution.')
        if fixed_viscosity:
            if history['optimizer_work']['predicted_physical_states']!=36000:
                raise ValueError('Fixed-viscosity fit does not match registered physical-state work')
            temporal[-1].update(record_kind='fit',trainable_parameters=value['model']['trainable_parameters'],
                pressure_expert_calls_per_block=value['model']['pressure_expert_calls_per_block'],
                training_context_source=STUDY+'/fixed-viscosity-family-results.json',
                training_scope=f"{expected_updates:,} full-state updates; 36,000 predicted physical-state targets. Learned viscosity is frozen from the start. Frozen pressure/flux/sampling/viscosity assets are inherited; equal target counts do not imply equal FLOPs.")
        temporal_rows.extend(temporal_accuracy_rows(label,pair['horizons'],value,os.path.relpath(lab/'runs'/STUDY/evaluation,base)))
    temporal_executions=append_viscosity_executions(temporal,temporal_rows,base,lab,load)
    append_replica_evaluations(temporal,temporal_executions,temporal_rows,base,lab,load)
    # Timings must name an admitted recipe and its evaluation population. The
    # newer conditioning/loss/curl fits have no matched timing; do not borrow
    # latency from an earlier fit with the same architecture or step count.
    temporal_by_identifier = {m['identifier']: m for m in temporal+temporal_executions}
    temporal_timing = []
    temporal_captures = {}
    timing_reports = {}
    timing_inputs=list(report['points'])
    timing_identity=lambda r:(r['track'],r['identifier'],r['regime'],r['interval'],r.get('source',r.get('timing_source')))
    known_timing_inputs={timing_identity(r) for r in timing_inputs}
    # Candidate frontiers can intentionally omit re-timed baseline points.
    # Retain the sealed matched-comparison rows as distinct capture records.
    for path in sorted((base/STUDY).glob('*cost-results.json')):
        cost=load(path)
        if cost['status']!='completed' or not cost.get('CPU_latency_claim'):
            continue
        for point in cost.get('points',[])+cost.get('comparison',[]):
            row=dict(point,source=os.path.relpath(path,lab))
            identity=timing_identity(row)
            if identity not in known_timing_inputs:
                timing_inputs.append(row)
                known_timing_inputs.add(identity)
    for row in timing_inputs:
        if row['track'] != 'B':
            continue
        # The frontier's seed-confirmation aliases duplicate the sealed cost
        # points ingested above; they are not separate executions or captures.
        if row['identifier'].endswith('/seed-confirmation'):
            continue
        model = temporal_by_identifier.get(row['identifier'])
        if model is None:
            continue
        reserved=row['regime'].startswith('reserved_')
        population='reserved' if reserved else 'development'
        regime=row['regime'].removeprefix('reserved_')
        if reserved:
            evaluation=model['reserved_evaluation']
            model=dict(model,evaluation_source=evaluation['source'],bank_sha256=evaluation['bank_sha256'])
        row=bind_temporal_timing(row,model,base,lab,load,timing_reports)
        actual=next(r for r in temporal_rows if r['label']==model['label']
                    and r['regime']==regime and r['interval']==row['interval']
                    and r.get('population','development')==population)
        if row['field_worst']!=actual['field_worst'] or row['accuracy_pass']!=actual['accuracy_pass']:
            raise ValueError('Temporal timing accuracy belongs to a different execution: '+row['identifier'])
        if row.get('weights_sha256', model['weights_sha256']) != model['weights_sha256']:
            raise ValueError('Temporal timing checkpoint differs: ' + row['identifier'])
        if (row['horizons'] != model['horizons']
                or os.path.relpath(lab/row['accuracy_source'],base) != model['evaluation_source']
                or not math.isclose(row['full_block_ms']/row['horizons'],row['median_ms'],rel_tol=1e-12)):
            raise ValueError('Temporal timing task/population differs: ' + row['identifier'])
        source = os.path.relpath(lab/row['timing_source'],base)
        if source not in temporal_captures:
            capture = load(lab/row['timing_source'])
            if capture['status'] != 'completed' or capture['bank_sha256'] != model['bank_sha256']:
                raise ValueError('Temporal timing input bank differs: ' + source)
            temporal_captures[source] = dict(source=source,
                label=Path(row['timing_source']).parent.name,
                **{k:capture[k] for k in ['cpu','threads','batch_size','warmup','balanced_serial']})
            if reserved:
                temporal_captures[source]['population']='reserved'
        normalized=dict(row,identifier=model['label'],recipe_identifier=row['identifier'],
            timing_source=source,accuracy_source=model['evaluation_source'])
        if reserved:
            normalized.update(regime=regime,population='reserved')
        temporal_timing.append(normalized)
    pressure=[]
    intermediate=[]
    for row in report['points']:
        if row['track']!='A':
            continue
        identifier=row['identifier'].split('/')[0]
        parent=identifier.removesuffix('-routed')
        if '@' in parent:
            parent,update=parent.rsplit('@',1)
            fit=next((m for m in models if m['name']==parent),None)
            if fit is None:
                raise ValueError('Unknown full-AI intermediate fit')
            if int(update)!=fit['optimizer_updates']:
                intermediate.append(dict(row,reason='Intermediate checkpoint, not the final admitted weights'))
                continue
        model=identifier if identifier in ('C106','C107') else by_name.get(parent)
        if model is None:
            raise ValueError('Unknown full-AI pressure execution: '+identifier)
        pressure.append(dict(row,model=model,source=os.path.relpath(lab/row.get('accuracy_source',row['timing_source']),base),
            timing_source=os.path.relpath(lab/row['timing_source'],base)))
    data['full_ai_study']=dict(id=STUDY,source=STUDY+'/frontiers.json',status=report['status'],goal_complete=report['goal_complete'],
        models=[m['cohort_id'] for m in models],controls=['C106','C107'],pressure=pressure,
        temporal_models=temporal,temporal_executions=temporal_executions,temporal=temporal_rows,intermediate_pressure=intermediate,
        temporal_timing=temporal_timing,temporal_timing_captures=list(temporal_captures.values()),batches=batches,
        excluded_temporal=[dict(pair='b1-pair-01',reason='Superseded pressure-gauge feedback; diagnostic only'),
            dict(pair='b4-cost-pilot-pair-01',reason='16-update cost pilot; not a 10,000-update candidate')],
        scopes=report['scopes'],precise_execution=report.get('precise_routed_followup'))
    # Native rejected requests do not supply an accepted drift curve. Retain
    # their measured disposition alongside the learned pressure diagnostics.
    known={p['id']:p for p in data['cohort_points']}
    for m in models:
        p=known[m['cohort_id']]
        p['native_drift_scope']='No accepted independent long-horizon native drift trajectory recorded for this fit. Teacher-state spectrogram replay is not model-driven drift.'
    enrich_causal_component(data['full_ai_study'],base,lab,load)
    enrich_causal_component(data['full_ai_study'],base,lab,load,
        identifier='b1-factored-request',evaluation='factored-request-evaluation-01',expected_updates=11000)
    data['full_ai_one_stage']=dict(models=[m['cohort_id'] for m in models if m['proposal_network_evaluations']==1])
    data['full_ai_two_stage']=dict(models=[m['cohort_id'] for m in models if m['proposal_network_evaluations']==2])
    from full_ai_tight_chart import enrich_tight_targets
    enrich_tight_targets(data['full_ai_study'],base,lab,load)
    from expanded_models_chart import enrich_expanded_models
    enrich_expanded_models(data['full_ai_study'],base,lab,load)


def full_ai_markup(text, base):
    start = text.index('// FULL AI STUDY START')
    end = text.index('// FULL AI STUDY END',start)
    return text[:start]+'// FULL AI STUDY START\n'+(Path(base)/'full_ai_study_ui.js').read_text()+'\n'+text[end:]


def enrich_causal_component(study, base, lab, load, *,
                           identifier='b1-causal-request',evaluation='causal-request-evaluation-01',expected_updates=10000):
    root=Path(lab)/'runs'/STUDY
    entry_path=root/'entries'/(identifier+'.json')
    result_path=root/evaluation/'summary.json'
    if not entry_path.exists() or not result_path.exists():
        return
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    entry=load(entry_path)
    artifacts=Path(lab)/entry['restore']
    pair=load(artifacts.parent/'receipt.json',entry['originating_pair_sha256'])
    restore=load(artifacts/'receipt.json',pair['restore_receipt_sha256'])
    load(artifacts.parent/'train/receipt.json',pair['training_receipt_sha256'])
    history=load(artifacts/'history.json',restore['outputs']['history.json']['sha256'])
    result=load(result_path)
    if (pair['status']!='completed' or pair['optimizer_updates']!=expected_updates
            or not all(pair['artifact_parity'].values()) or restore['mode']!='restore'
            or restore['optimizer_updates'] or restore['execution']['cuda_initialized']
            or not restore['full_cpu_probe_bitwise_verified'] or not result['component_only']
            or result['full_state_temporal_candidate'] or result['own_history_rollout']
            or result['optimizer_updates'] or result['candidate_numerical_pressure_solve']
            or result['component_entry_sha256']!=sha(entry_path)):
        raise ValueError('Causal request component custody/scope differs')
    for source,expected in entry['identities'].items():
        if sha(Path(lab)/source)!=expected:
            raise ValueError('Causal request checkpoint identity differs')
    for name in pair['artifact_parity']:
        if sha(artifacts/name)!=sha(artifacts.parent/'train'/name):
            raise ValueError('Causal request CPU restore differs')
    for source,expected in result['sources'].items():
        load(Path(lab)/source,expected)
    rows=[]
    for source in result['traces']:
        trace=load(Path(lab)/source['path'],source['sha256'])
        if trace['status']!='completed' or not trace['component_only'] or trace['own_history_rollout']:
            raise ValueError('Causal request trace scope differs')
        rows.extend(trace['rows'])
    component=dict(name=entry['id'],parameters=pair['parameters'],optimizer_updates=pair['optimizer_updates'],
        pressure_core='C128',groups=result['groups'],rows=rows,optimizer_work=history['optimizer_work'],
        component_only=True,own_history_rollout=False,source=os.path.relpath(result_path,base),
        training_source=os.path.relpath(artifacts.parent/'receipt.json',base))
    study.setdefault('causal_components',[]).append(component)
    if identifier=='b1-causal-request':
        study['causal_component']=component
