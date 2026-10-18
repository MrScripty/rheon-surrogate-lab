"""Extend the retained chart snapshot with completed current-checkout studies.

Historical source branches stay untouched. Reads sealed managed artifacts and
recorded CPU campaigns; executes no training or simulation.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from current_cohort_records import ROOT, LAB, discover, discover_completed_models, digest, read, relative
from drift_pressure_metrics import enrich_drift_pressure
from pass_sweep_chart import enrich_pass_sweep
from precision_chart import enrich_precision, precision_markup
from four_pass_study import four_pass_link, reserved_four_pass_ids

PAGE = ROOT / 'cohort-progress.html'
MARKER = 'current-managed-pressure-v1'


def main():
    text = PAGE.read_text()
    match = re.search(r'(<script id="chart-data" type="application/json">)(.*?)(</script>)', text, re.S)
    data = json.loads(match[2])
    sources = data['history_sources']
    studies = discover()
    if not studies:
        print('No completed current-checkout pressure studies found.')
        return

    def load(path, expected=None):
        path = Path(path)
        value = read(path, expected)
        sources[relative(path)] = digest(path)
        return value

    def upsert(key, row, identity):
        rows = data[key]
        old = next((i for i, r in enumerate(rows) if r[identity] == row[identity]), None)
        if old is None:
            rows.append(row)
        else:
            rows[old] = row

    # Rebuild only these imported study views; preserve all historical records.
    for key in ('history', 'policy_studies', 'trajectory_studies'):
        data[key] = [r for r in data[key] if r.get('chart_importer') != MARKER]
    points = {p['weights_sha256']: p for p in data['cohort_points']}
    reserved_ids = reserved_four_pass_ids(ROOT)
    new_entries = discover_completed_models()
    reference_probe = data['frequency_probes'][0]
    for entry in new_entries:
        name, study, weight = entry['name'], entry['study_id'], entry['weights_sha256']
        directory = ROOT / 'chart-diagnostics' / study / name
        diagnostic_path = directory / 'validation.json'
        diagnostic = load(diagnostic_path)
        load(ROOT / diagnostic['admission_source'], diagnostic['admission_sha256'])
        load(ROOT / entry['training_source'], entry['training_sha256'])
        project, artifacts = LAB / entry['project'], LAB / entry['artifacts']
        training_receipt = load(artifacts.parent / 'train/receipt.json')
        history = load(artifacts / 'history.json', training_receipt['outputs']['history.json']['sha256'])
        splits = load(artifacts / 'splits.json', training_receipt['outputs']['splits.json']['sha256'])
        if (diagnostic['weights_sha256'] != weight or digest(artifacts / 'weights.npz') != weight
                or diagnostic['optimizer_updates'] or diagnostic['simulation_steps']):
            raise ValueError(f'Unbound chart diagnostic: {name}')
        raw = (ROOT / diagnostic['case_source']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != diagnostic['cases_sha256']:
            raise ValueError(f'Changed per-case diagnostic: {name}')
        cases = [json.loads(line) for line in raw.decode().splitlines()]
        errors = [c['raw_learned']['quality']['relative_residual'] for c in cases]
        residual = diagnostic['relative_residual']
        if (len(cases) != 128 or max(errors) != residual['maximum'] or min(errors) != residual['minimum']
                or statistics.median(errors) != residual['median']):
            raise ValueError(f'Incomplete validation cases: {name}')
        sources[diagnostic['case_source']] = diagnostic['cases_sha256']
        timings = diagnostic['per_case_inference_s']
        if weight not in points:
            number = max(int(p['id'][1:]) for p in points.values()) + 1
            while f'C{number:02d}' in reserved_ids:
                number += 1
            point = dict(id=f'C{number:02d}', order=number,
                name=name, family=entry['family'], weights_sha256=weight,
                trained_at=datetime.fromtimestamp(history['training_started_unix'], timezone.utc).isoformat(),
                best_error=min(errors), median_error=statistics.median(errors), worst_error=max(errors),
                cases=128, qualified=None, inference_median_ms=statistics.median(timings)*1000,
                inference_best_ms=min(timings)*1000, inference_worst_ms=max(timings)*1000,
                timed_cases=len(timings), timing_cases=diagnostic['timing_cases'],
                source=relative(diagnostic_path), timing_source=relative(diagnostic_path),
                training_source=relative(artifacts / 'history.json'), case_source=diagnostic['case_source'],
                validation_summary_source=relative(diagnostic_path), diagnostic_scope=diagnostic['scope'],
                policy='Original model proposal, before adaptive repetition or native refinement',
                timing_scope='CPU pressure prediction; excludes setup, sample assembly, native verification and fallback',
                **{k: diagnostic[k] for k in ['run_root','replay_project','replay_artifacts','replay_source_lab']})
            points[weight] = point
            data['cohort_points'].append(point)
        point = points[weight]
        identity = lambda c: json.dumps([c['trajectory_id'], c.get('native_frame', c['frame']), c['counts_xyz'], c['cell_size_m']])
        baseline = points[data['resolution_evaluations'][0]['weights_sha256']]
        if sorted(map(identity, cases)) != sorted(map(identity, baseline['timing_cases'])):
            raise ValueError(f'New cohort validation population differs: {name}')
        grouped = {}
        for case, error in zip(cases, errors):
            grouped.setdefault((tuple(case['counts_xyz']),case['cell_size_m']), []).append((case,error))
        domains = []
        for (counts, spacing), rows in sorted(grouped.items()):
            worst, error = max(rows, key=lambda r: r[1])
            domains.append(dict(counts_xyz=list(counts), cell_size_m=spacing, cases=len(rows),
                worst_error=error, best_error=min(r[1] for r in rows), qualified=0,
                worst_case={k:worst[k] for k in ['trajectory_id','frame','native_frame']}))
        upsert('resolution_evaluations',dict(weights_sha256=weight,source=diagnostic['case_source'],
            summary_source=relative(diagnostic_path),cases=128,domains=domains),'weights_sha256')
        # Count complete real training splits and the original analytical streams.
        training_domains = {}
        def domain(counts, spacing):
            return training_domains.setdefault((tuple(counts),spacing),dict(counts_xyz=list(counts),cell_size_m=spacing,
                snapshots=set(),simulations=set(),manufactured_presentations=0))
        for parent in splits['parents']:
            if parent['entry']['split'] != 'train':
                continue
            d = domain(parent['counts_xyz'],parent['cell_size_m'])
            parent_id = (parent.get('capture_manifest_sha256'),parent['entry']['trajectory_id'])
            d['simulations'].add(parent_id)
            d['snapshots'].update((*parent_id,frame['index']) for frame in parent['frames'])
        for key in ('analytical_feedback','supplemental_feedback'):
            feedback = history[key]
            if sum(feedback['grid_exposures']) != feedback['additional_examples']:
                raise ValueError('Training grid exposures disagree')
            for grid, count in zip(feedback['grids'],feedback['grid_exposures']):
                grid = grid.get('grid',grid)
                domain(grid['counts_xyz'],grid.get('cell_size_m') or 1/grid['cells_per_metre'])['manufactured_presentations'] += count
        counted = [dict(counts_xyz=d['counts_xyz'],cell_size_m=d['cell_size_m'],unique_real_snapshots=len(d['snapshots']),
            real_simulations=len(d['simulations']),manufactured_presentations=d['manufactured_presentations']) for d in training_domains.values()]
        upsert('training_exposures',dict(weights_sha256=weight,source=relative(artifacts/'splits.json'),
            receipt_source=relative(artifacts.parent/'train/receipt.json'),history_source=relative(artifacts/'history.json'),domains=counted,
            **{key:sum(d[key] for d in counted) for key in ['unique_real_snapshots','real_simulations','manufactured_presentations']},
            current_policy_presentations=history.get('current_feedback',{}).get('additional_examples',0),
            scope='Original real teacher snapshots and analytical training draws. Current-policy residual replays are counted separately; they are not additional CFD simulations.'),'weights_sha256')
        # New probes replay the exact saved benchmark, so physical coordinates are shared.
        probe_path = directory / 'mode-probe.json'
        probe = load(probe_path)
        if probe['weights_sha256'] != weight or len(probe['cases']) != len(reference_probe['cases']):
            raise ValueError(f'Incomplete frequency probe: {name}')
        frequency_cases, sample_keys = [], []
        for i,(case,reference) in enumerate(zip(probe['cases'],reference_probe['cases'])):
            sample = {k:case[k] for k in ['counts_xyz','cell_size_m','population','components_xyz','amplitude_pa']}
            if any(sample[k] != reference[k] for k in sample):
                raise ValueError('Frequency input specifications differ')
            sample_keys.append(json.dumps(sample,sort_keys=True))
            trace = [s['relative_residual'] for s in case['trace']]
            if len(trace) != case['network_evaluations'] or trace[0] != case['first_relative_residual'] or trace[-1] != case['final_relative_residual']:
                raise ValueError('Frequency trace does not match source')
            frequency_cases.append(dict(**sample,id=hashlib.sha256(f'{relative(probe_path)}:{i}'.encode()).hexdigest()[:20],
                rms_cycles_per_m=reference['rms_cycles_per_m'],rms_cycles_per_cell=reference['rms_cycles_per_cell'],
                first_error=trace[0],final_error=trace[-1],trace=trace,network_evaluations=len(trace),equation_passed=case['equation_passed']))
        benchmark = hashlib.sha256(json.dumps([probe['seed'],probe['source_sha256'],sorted(sample_keys)]).encode()).hexdigest()
        upsert('frequency_probes',dict(weights_sha256=weight,source=relative(probe_path),benchmark_sha256=benchmark,
            seed=probe['seed'],cases=frequency_cases),'weights_sha256')

    traces_by_receipt = {sources.get(t['receipt_source']):t for t in data['drift_traces']}
    trace_keys = {t['id'] for t in data['drift_traces']}
    def measurement(directory, weight, name, source, expected=None):
        directory = Path(directory)
        native = directory / 'native'
        receipt = load(native/'receipt.json',expected)
        raw = (native/'frames.jsonl').read_bytes()
        sources[relative(native/'frames.jsonl')] = hashlib.sha256(raw).hexdigest()
        frames = [json.loads(line) for line in raw.decode().splitlines()]
        callbacks = [json.loads(line) for line in (native/'callbacks.jsonl').read_text().splitlines()]
        sources[relative(native/'callbacks.jsonl')] = digest(native/'callbacks.jsonl')
        ai = [f for f in frames if f.get('ai_quality')]
        if (not ai or len(ai)!=receipt['result']['ai_steps'] or len(callbacks)!=len(ai)
                or receipt['result']['hard_stop'] or not receipt['native_process']['reaped']
                or not all(f['reference_quality']['qualified'] for f in frames)
                or not all(f['ai_quality']['qualified'] for f in ai)):
            raise ValueError(f'Incomplete trajectory: {directory}')
        manifest = load(receipt['adapter_manifest'],receipt['adapter_manifest_sha256'])
        if manifest['files']['weights.npz']['sha256'] != weight:
            raise ValueError('Trajectory adapter checkpoint differs')
        points_ = [[f['step'],f['time_s'],f['relative_pressure_difference'],True,f['ai_quality']['relative_residual']] for f in ai]
        policy = {k:receipt.get(k) for k in ['learned_proposals_per_pressure_solve','adaptive_stopping','proposal_network_evaluations','minimum_network_evaluations','pressure_bound']}
        scene_key = hashlib.sha256(json.dumps(dict(scene=receipt['scene'],binary=receipt['binary_sha256']),sort_keys=True).encode()).hexdigest()[:20]
        trace_key = hashlib.sha256(json.dumps([scene_key,weight,policy,points_],sort_keys=True).encode()).hexdigest()[:20]
        if trace_key not in trace_keys:
            data['drift_traces'].append(dict(id=trace_key,scene_key=scene_key,scene=receipt['scene'],model=name,policy_label=name,
                weights_sha256=weight,policy=policy,points=points_,completed=ai[-1]['step']==receipt['scene']['steps'],hard_stop=False,
                qualified_steps=len(ai),source=source,frames_source=relative(native/'frames.jsonl'),receipt_source=relative(native/'receipt.json')))
            trace_keys.add(trace_key)
        sparse_path = directory/'diagnostics.json'
        sparse = load(sparse_path)['samples'] if sparse_path.exists() else []
        return dict(errors=[f['relative_pressure_difference'] for f in ai],calls=[c['network_passes'] for c in callbacks],sparse=sparse,
            times=[f['ai_seconds'] for f in ai],native_times=[f['reference_seconds'] for f in ai],receipt=receipt)

    historical_weights = next(s for s in data['policy_studies'] if s['id']=='water-early-pressure-v1')['model_weights']
    group_number = max(int(r['group'].split(' · ')[0]) for r in data['history'])+1
    for study in studies:
        regression = study['regression']
        load(ROOT/study['regression_source'])
        historical = regression['sources']['historical_summary']
        load(historical['path'],historical['sha256'])
        model_weights = {name:historical_weights[name] for name in ['reference-current','late-pressure','control']}
        campaign_measurements = {}
        for name,evidence in regression['sources']['campaigns'].items():
            campaign_path = Path(evidence['path'])
            campaign = load(campaign_path,evidence['sha256'])
            if campaign['status'] not in ('passed','completed') or len(campaign['records'])!=74 or not campaign['all_worker_children_reaped']:
                raise ValueError(f'Incomplete campaign: {campaign_path}')
            if evidence.get('equivalence'):
                load(evidence['equivalence']['path'],evidence['equivalence']['sha256'])
            for row in campaign['records']:
                command = row['command'];directory=Path(command[command.index('--output')+1])
                config = load(command[command.index('--config')+1],campaign['config_sha256'])
                manifest = load(config['manifest'],campaign['manifest_sha256'])
                weight = manifest['files']['weights.npz']['sha256']
                model_weights[name]=weight
                item=measurement(directory,weight,name,study['regression_source'],row['metrics']['receipt_sha256'])
                if item['calls']!=row['network_calls'] or not math.isclose(max(item['errors']),row['metrics']['worst_paired_pressure_relative_l2'],rel_tol=1e-12):
                    raise ValueError('Campaign measurements disagree with retained frames')
                campaign_measurements[name,row['scene']]=item
        all_measured = {}
        for row in regression['cases']:
            name,index = row['model'],row['scene']
            if (name,index) in campaign_measurements:
                item=campaign_measurements[name,index]
            else:
                trace=traces_by_receipt.get(row['source_receipt_sha256'])
                if trace is not None:
                    if trace['weights_sha256'] != model_weights[name]:
                        raise ValueError('Historical checkpoint differs')
                    item=dict(errors=[p[2] for p in trace['points']],calls=row['calls'],sparse=[],times=[],native_times=[])
                else:
                    # Identical old trajectories may have been deduplicated under
                    # another receipt. Recover this exact sealed observation.
                    source=next((p for p,sha in sources.items() if sha==row['source_receipt_sha256'] and p.endswith('/receipt.json')),None)
                    if source is None:
                        raise ValueError(f'Historical trace not retained: {name}/{index}')
                    native=(ROOT/source).parent
                    receipt=load(native/'receipt.json',row['source_receipt_sha256'])
                    frames_source=relative(native/'frames.jsonl')
                    if sources.get(frames_source) and digest(native/'frames.jsonl')!=sources[frames_source]:
                        raise ValueError('Historical trajectory frames changed')
                    frames=[json.loads(line) for line in (native/'frames.jsonl').read_text().splitlines()]
                    ai=[f for f in frames if f.get('ai_quality')]
                    callbacks=[json.loads(line) for line in (native/'callbacks.jsonl').read_text().splitlines()]
                    if len(ai)!=receipt['result']['ai_steps'] or not all(f['ai_quality']['qualified'] for f in ai):
                        raise ValueError('Incomplete historical trajectory')
                    item=dict(errors=[f['relative_pressure_difference'] for f in ai],calls=[c['network_passes'] for c in callbacks],sparse=[],times=[],native_times=[])
            if item['calls']!=row['calls'] or not math.isclose(max(item['errors']),row['worst_paired_pressure_relative_l2'],rel_tol=1e-12):
                raise ValueError('Regression summary differs from original observations')
            all_measured[name,index]=item
        candidate=study['models'][0]['name']
        phases,populations=[],{}
        def phase(phase_id,title,selected,timed=False):
            models=[]
            for name,items in selected.items():
                errors=[e for item in items for e in item['errors']];calls=[c for item in items for c in item['calls']];sparse=[s for item in items for s in item['sparse']]
                models.append(dict(name=name,weights_sha256=model_weights[name],episodes=len(items),qualified_episodes=len(items),ai_steps=len(errors),
                    mean_network_evaluations=statistics.mean(calls),maximum_network_evaluations=max(calls),minimum_network_evaluations=min(calls),
                    worst_paired_pressure_relative_l2=max(errors),median_paired_pressure_relative_l2=statistics.median(errors),
                    inference_median_ms=statistics.median([t for item in items for t in item['times']])*1000 if timed else None,
                    sparse_worst_pressure_relative_l2=max((s['pressure_relative_l2'] for s in sparse),default=None),
                    sparse_worst_low_band_relative_l2=max((s['mean_free_pressure_relative_l2']*math.sqrt(s['low_mode_error_energy_fraction']) for s in sparse),default=None),
                    sparse_worst_other_band_relative_l2=max((s['mean_free_pressure_relative_l2']*math.sqrt(max(0,1-s['low_mode_error_energy_fraction'])) for s in sparse),default=None)))
            native_ms=statistics.median([t for items in selected.values() for item in items for t in item['native_times']])*1000 if timed else None
            gates={}
            if not timed:
                expected=regression['groups'][phase_id.removeprefix('group-')]
                for m in models:
                    r=expected['models'][m['name']]
                    if (r['scenes']!=m['episodes'] or r['ai_steps']!=m['ai_steps'] or r['maximum_network_evaluations']!=m['maximum_network_evaluations']
                            or not math.isclose(r['mean_network_evaluations'],m['mean_network_evaluations'],rel_tol=1e-12)
                            or not math.isclose(r['worst_paired_pressure_relative_l2'],m['worst_paired_pressure_relative_l2'],rel_tol=1e-12)):
                        raise ValueError('Population aggregate does not match recorded summary')
                gates={c+' vs '+b+': '+k:v for c,baselines in expected['gates'].items() for b,checks in baselines.items() for k,v in checks.items()}
            reference=next(name for name in ['reference01','reference003','cuda-control','control'] if name in selected)
            source=study['timing_source'] if timed else study['regression_source']
            phases.append(dict(id=phase_id,title=title,timed=timed,models=models,native_median_ms=native_ms,native_label='Independent native reference branch',
                source=source,records_source=source,timing_scope=study['timing']['scope'] if timed else 'Instrumented consumed correctness campaign; not an inference latency measurement.',
                gates=gates,passed=all(gates.values()) if gates else None,reference_policy=reference,candidate_policy=candidate))
            populations[phase_id]=dict(reference=reference,candidate=candidate,models={m['name']:dict(scenes=m['episodes'],qualified_scenes=m['qualified_episodes'],
                attempted_ai_steps=m['ai_steps'],qualified_ai_steps=m['ai_steps'],mean_network_evaluations=m['mean_network_evaluations'],worst_pressure_relative_l2=m['worst_paired_pressure_relative_l2']) for m in models})
            for m in models:
                data['history'].append(dict(chart_importer=MARKER,group=f'{group_number:02d} · '+study['title'],name=m['name']+' · '+phase_id,source=source,
                    median=m['median_paired_pressure_relative_l2'],worst=m['worst_paired_pressure_relative_l2'],qualified=m['episodes'],cases=m['episodes'],metric='field_l2',
                    population=title,policy='Pressure-aware iterative solve'+(f" · diagnostic AI branch {m['inference_median_ms']:.3f} ms" if timed else ''),
                    weights_sha256=m['weights_sha256'],evaluations=m['mean_network_evaluations'],speed=None,updates=0))
        for group in regression['groups']:
            selected={name:[all_measured[name,r['scene']] for r in regression['cases'] if r['group']==group and r['model']==name] for name in model_weights}
            phase('group-'+group,'Consumed '+group,selected)
        # Timing rows are paired diagnostic branches, never standalone solver speed.
        timed={}
        for label,evidence in study['timing']['sources'].items():
            campaign=load(evidence['path'],evidence['sha256'])
            name=label.rsplit('-rep',1)[0]
            if name=='control':
                model_weights[name]=historical_weights['control']
            for row in campaign['records']:
                command=row['command'];directory=Path(command[command.index('--output')+1])
                item=measurement(directory,model_weights[name],name,study['timing_source'])
                scene=row['scene'];timed.setdefault(scene,{}).setdefault(name,[]).append(item)
                summary=next(r for r in study['timing']['rows'] if r['model']==label and r['scene']==scene)
                if (digest(directory/'native/frames.jsonl')!=summary['frames_sha256'] or digest(directory/'native/callbacks.jsonl')!=summary['callbacks_sha256']
                        or not math.isclose(statistics.median(item['times']),summary['median_ai_step_s'],rel_tol=1e-12)):
                    raise ValueError('Serial diagnostic timing source differs')
        for scene,selected in sorted(timed.items()):
            phase('serial-'+str(scene),'Serial paired diagnostic · grid '+str(scene)+' · two repeats',selected,True)
        frozen=min(m['started_at'] for m in study['models'])
        note='All new models physically qualified all 74 consumed scenes. No candidate passed every comparative gate or was promoted; eight unused holdouts remain untouched. The highlighted model is a display comparison only. Serial AI-branch timings include an additional same-request native reference solve, capture, learned correction, IPC and logging; they do not measure standalone solver or model speed.'
        if 'reference-band003' in model_weights:
            note+=' Reference01 reuses the retained current-feedback 0.1 model. Reference-band003 is historical context with a different training-work budget, not a matched third arm.'
        data['policy_studies'].append(dict(chart_importer=MARKER,id=study['id'],title=study['title'],phases=phases,populations=populations,
            selected_policy=None,comparison_policy=candidate,model_weights=model_weights,source=study['regression_source'],selection_source=study['decision_source'],
            frozen_at_utc=frozen,candidate_promoted=False,fresh_numeric_gates_passed=None,default_phase='group-original58',default_population='group-original58',
            population_labels={p['id']:p['title'] for p in phases},training_runs=len(study['models']),new_weights=[m['weights_sha256'] for m in study['models']],
            outcome_note=note,source_scope=study['timing']['scope'],highlight_label='Amber ring: displayed comparison model; no candidate was selected for promotion.'))
        data['trajectory_studies'].append(dict(chart_importer=MARKER,id=study['id'],title=study['title']+' · trained checkpoint rollouts',kind='stopping-policy',
            populations=populations,candidate=candidate,reference=populations['group-original58']['reference'],model_weights=model_weights,
            frozen_at_utc=frozen,source=study['regression_source'],selection_source=study['decision_source'],decision=dict(candidate_promoted=False),
            population_labels={p['id']:p['title'] for p in phases},default_population='group-original58',outcome_note=note))
        group_number+=1
    data['policy_studies'].sort(key=lambda s:s['frozen_at_utc'])
    data['trajectory_studies'].sort(key=lambda s:s['frozen_at_utc'])
    # A completed checkpoint remains chartable when its rollout qualification
    # failed. Retain observed prefixes without filling unstarted/cancelled cases.
    data['rollout_campaign_status'] = []
    for entry in new_entries:
        if not entry.get('qualification_source'):
            continue
        qualification = load(ROOT / entry['qualification_source'])
        for path, expected in qualification['sources'].items():
            if digest(path) != expected:
                raise ValueError(f'Partial campaign evidence changed: {path}')
            sources[relative(path)] = expected
        study = LAB / 'runs' / entry['study_id']
        campaign = study / 'cpu-02/regression'
        weight = entry['weights_sha256']
        observed = 0
        for row in qualification['rows']:
            folders = list(campaign.glob(f"case-*-scene-{row['scene']:02d}"))
            if not folders or not (folders[0] / 'native/receipt.json').exists():
                continue
            folder = folders[0]
            episode = load(folder / 'episode.json')
            receipt = load(folder / 'native/receipt.json', episode.get('metrics', {}).get('receipt_sha256'))
            frames_path = folder / 'native/frames.jsonl'
            raw = frames_path.read_bytes()
            sources[relative(frames_path)] = hashlib.sha256(raw).hexdigest()
            frames = [json.loads(line) for line in raw.decode().splitlines()]
            ai = [f for f in frames if f.get('ai_quality')]
            if not ai:
                continue
            manifest = load(receipt['adapter_manifest'], receipt['adapter_manifest_sha256'])
            if manifest['files']['weights.npz']['sha256'] != weight or not receipt['native_process']['reaped']:
                raise ValueError('Partial rollout checkpoint/custody mismatch')
            values = [[f['step'], f['time_s'], f['relative_pressure_difference'], f['ai_quality']['qualified'],
                       f['ai_quality']['relative_residual']] for f in ai]
            policy = {k: receipt.get(k) for k in ['learned_proposals_per_pressure_solve','adaptive_stopping',
                      'proposal_network_evaluations','minimum_network_evaluations','pressure_bound']}
            scene_key = hashlib.sha256(json.dumps(dict(scene=receipt['scene'],binary=receipt['binary_sha256']),sort_keys=True).encode()).hexdigest()[:20]
            trace_id = hashlib.sha256(json.dumps([scene_key,weight,policy,values],sort_keys=True).encode()).hexdigest()[:20]
            if trace_id not in trace_keys:
                data['drift_traces'].append(dict(id=trace_id,scene_key=scene_key,scene=receipt['scene'],model=entry['name'],
                    policy_label=entry['name'],weights_sha256=weight,policy=policy,points=values,
                    completed=ai[-1]['step']==receipt['scene']['steps'],hard_stop=bool((receipt['result'] or {}).get('hard_stop')),
                    qualified_steps=sum(f['ai_quality']['qualified'] for f in ai),source=entry['qualification_source'],
                    frames_source=relative(frames_path),receipt_source=relative(folder/'native/receipt.json'),
                    campaign_disposition=row['disposition'],campaign_complete=False,
                    independent_certificate_failed=row['disposition']=='independent_pressure_bound_failure'))
                trace_keys.add(trace_id)
            observed += 1
        status = dict(weights_sha256=weight,model=entry['name'],source=entry['qualification_source'],
            campaign_complete=qualification['campaign_complete'],qualification=qualification['qualification'],
            scheduled_scenes=qualification['scheduled_scenes'],dispositions=qualification['dispositions'],
            observed_traces=observed,completed_independent_checks=qualification['completed_independent_checks'],
            scope=qualification['scope'])
        data['rollout_campaign_status'].append(status)
        points[weight]['rollout_qualification'] = status
    # Include the new raw-proposal measurements as their own scientific metric.
    for entry in new_entries:
        point=points[entry['weights_sha256']]
        data['history'].append(dict(chart_importer=MARKER,group=f'{group_number:02d} · Current original-proposal snapshots',
            name=point['name'],source=point['source'],median=point['median_error'],worst=point['worst_error'],qualified=None,cases=128,
            population='Same 128 consumed validation snapshots',policy='Original proposal · chart CPU diagnostic',weights_sha256=point['weights_sha256'],updates=5000))
    # Expose the build already bound into each setup's identity so the UI can
    # distinguish otherwise identical simulations before and after cutover.
    for trace in data['drift_traces']:
        receipt=load(ROOT/trace['receipt_source'],sources.get(trace['receipt_source']))
        trace['native_binary_sha256']=receipt['binary_sha256']
    spectral=ROOT/'pressure-spectrogram-data.json'
    if spectral.exists():
        psd=load(spectral)
        if psd['source_sha256']==digest(ROOT/'build-pressure-spectrogram.py'):
            for model in psd['models']:
                if digest(ROOT/model['weights_source'])!=model['weights_sha256']:
                    raise ValueError('Spectral replay checkpoint changed')
            data['pressure_spectrogram']=psd
    enrich_drift_pressure(data, ROOT)
    enrich_pass_sweep(data, ROOT)
    enrich_precision(data, ROOT)
    now=datetime.now(timezone.utc);data['history_refreshed_utc']=now.isoformat();data['snapshot_date']=now.astimezone(ZoneInfo('America/Vancouver')).date().isoformat()
    text=text[:match.start(2)]+json.dumps(data,separators=(',',':'),allow_nan=False)+text[match.end(2):]
    text=re.sub(r'Evidence snapshot · [^<]+','Evidence snapshot · '+now.astimezone(ZoneInfo('America/Vancouver')).strftime('%d %b %Y'),text)
    PAGE.write_text(four_pass_link(precision_markup(text, ROOT), ROOT))
    print(f"Updated chart: {len(data['cohort_points'])} checkpoints, {len(data['history'])} history rows, {len(data['drift_traces'])} trajectories.")


if __name__=='__main__':
    main()
