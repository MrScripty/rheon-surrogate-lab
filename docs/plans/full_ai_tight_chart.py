"""Import completed tight-target component fits and full-state sampler evaluations.

Only reads recorded artifacts. Configurable recurrent horizons share one trained
fit; pressure-only diagnostics and offline oracle controls never become solvers.
"""
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import statistics

STUDY = 'full-ai-tight-targets-v1'
FIELDS = ('equation', 'pressure', 'velocity', 'tracer')


def sampler_accuracy_rows(value, label, horizons, source, bank, population='development'):
    """Aggregate every scored state, requiring the full per-scene interval."""
    rollout = value['own_history_without_teacher_resets']
    intervals = [('rollout_12', 12), ('rollout_48', 48), ('rollout_240', 240)] if rollout else [('single_block', horizons)]
    result = []
    for regime in ('small', '32cubed'):
        available = [r for r in value['rows'] if r['horizons'] == horizons
                     and ('small' if max(r['counts_xyz']) <= 16 else '32cubed') == regime]
        scenes = {r['identifier'] for r in available}
        if not scenes:
            raise ValueError('Missing sampler grid population')
        for interval, steps in intervals:
            selected = [r for r in available if not rollout or r['step'] <= steps]
            for scene in scenes:
                actual = {(r['start'], r['step']) for r in selected if r['identifier'] == scene}
                starts = {0} if rollout else {0, 120}
                expected = {(start, start + step) for start in starts for step in range(1, steps + 1)}
                if actual != expected or len([r for r in selected if r['identifier'] == scene]) != len(expected):
                    raise ValueError('Incomplete or duplicated sampler interval: ' + scene)
            errors = {f: max(r['metrics']['relative'][f] for r in selected) for f in FIELDS}
            if not all(math.isfinite(e) and e >= 0 for e in errors.values()):
                raise ValueError('Nonfinite sampler accuracy')
            result.append(dict(label=label, horizons=horizons, regime=regime, interval=interval,
                field_worst=errors, scenes=len(scenes), complete=True,
                accuracy_pass=all(r['metrics']['all_accuracy_criteria_pass'] for r in selected),
                source=source, population=population, bank_sha256=bank, study_id=STUDY,
                scored_states=len(selected), accuracy_limit=1e-5,
                sampling_scope='Every intermediate state; '+('own history from step 0' if rollout else 'reference-state starts 0 and 120')))
    return result


def enrich_tight_targets(study, base, lab, load):
    base, lab = Path(base), Path(lab)
    docs, root = base / STUDY, lab / 'runs' / STUDY
    if not (docs / 'recurrent-sampler-summary-01.json').exists():
        return
    hashes, values = {}, {}

    def sha(path):
        path = Path(path)
        if path not in hashes:
            with path.open('rb') as stream:
                hashes[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return hashes[path]

    def read(path, expected=None):
        path = Path(path)
        if expected is not None and sha(path) != expected:
            raise ValueError('Tight-target evidence changed: ' + str(path))
        if path not in values:
            values[path] = load(path, expected)
        return values[path]

    def witness(value):
        for field in ('source_evidence', 'source_sha256'):
            for source, expected in value.get(field, {}).items():
                path = lab / source
                if path.suffix == '.json':
                    read(path, expected)
                elif sha(path) != expected:
                    raise ValueError('Tight-target source changed: ' + source)

    def qualified(path, status, protocol):
        value = read(path)
        if value['status'] != status or value.get('optimizer_updates') or value.get('cuda_initialized') or not value['all_plan_owners_closed']:
            raise ValueError('Incomplete tight-target evaluation: ' + str(path))
        reg = read(docs / protocol, value['protocol_sha256'])
        witness(reg)
        witness(value)
        return value, reg

    def pair(name, expected_updates):
        directory = root / name
        receipt = read(directory / 'receipt.json')
        train = read(directory / 'train/receipt.json', receipt['training_receipt_sha256'])
        restore = read(directory / 'restore/receipt.json', receipt['restore_receipt_sha256'])
        if (receipt['status'] != 'completed' or receipt['model_promotion']
                or receipt['optimizer_updates'] != expected_updates or not all(receipt['artifact_parity'].values())
                or train['mode'] != 'train' or train['optimizer_updates'] != expected_updates
                or restore['mode'] != 'restore' or restore['optimizer_updates']
                or restore['execution']['cuda_initialized'] or not restore['full_cpu_probe_bitwise_verified']):
            raise ValueError('Managed sampler/component restore differs: ' + name)
        for filename in receipt['artifact_parity']:
            if sha(directory / 'train' / filename) != sha(directory / 'restore' / filename):
                raise ValueError('Tight-target train/CPU restore artifact differs')
        for filename, spec in restore['outputs'].items():
            if sha(directory / 'restore' / filename) != spec['sha256']:
                raise ValueError('Tight-target restored output changed')
        history = read(directory / 'restore/history.json')
        if not history['train_only']:
            raise ValueError('Tight-target training population differs')
        return dict(pair=name, weights_sha256=sha(directory / 'restore/weights.npz'),
            optimizer_updates=expected_updates, parameters=receipt['parameters'],
            source=os.path.relpath(directory / 'receipt.json', base),
            trained_at=datetime.fromtimestamp(history['training_started_unix'], timezone.utc).isoformat())

    publication = read(docs / 'recurrent-sampler-summary-01.json')
    if publication['status'] != 'completed_trained_prototype_evaluation_targets_unmet':
        raise ValueError('Sampler publication incomplete')
    for source, expected in publication['evidence'].items():
        read(lab / source, expected)
    transport = pair('sampler-head-pair-01', 440)
    pressure = pair('modal-pressure-pair-01', 800)
    components = []
    for name, updates, folder, protocol in [
        ('modal-pressure-pair-01', 800, 'modal-component-evaluation-01', 'modal-component-evaluation-protocol-01.json'),
        ('modal-precision-pair-02', 80, 'modal-precision-component-evaluation-02', 'modal-precision-component-evaluation-protocol-02.json'),
    ]:
        path = root / folder / 'results.json'
        if not path.exists():
            continue
        fitted = pair(name, updates)
        result, reg = qualified(path, 'completed_offline_learned_pressure_component_evaluation', protocol)
        if (not result['teacher_rhs_used'] or result['causal_full_state_generator'] or result['own_history_rollout']
                or result['weights_sha256'] != fitted['weights_sha256'] or not result['weights_unchanged']
                or len(result['rows']) != reg['expected_pressure_states']):
            raise ValueError('Pressure-only component scope differs')
        components.append(dict(fitted, name='Learned modal pressure' if updates == 800 else 'Modal pressure precision refinement',
            component_only=True, inherited_optimizer_updates=0 if updates == 800 else 800,
            source_evaluation=os.path.relpath(path, base), groups=result['summaries']))
    study['tight_components'] = components

    def evaluation(folder, protocol, label, k, population='development', mode='rollout'):
        path = root / folder / 'results.json'
        value, reg = qualified(path, 'completed_trained_full_state_' + mode, protocol)
        if (not value['every_intermediate_state_scored'] or not value['every_output_typed_finite_and_hashed']
                or not value['truth_used_only_after_causal_prediction']
                or value['own_history_without_teacher_resets'] != (mode == 'rollout')
                or value['fresh_holdout_consumed'] != (population == 'sampler_fresh')
                or value['model_identity']['transport_weights_sha256'] != transport['weights_sha256']
                or value.get('native_pressure_solver_called') or value.get('analytic_inverse_constructed')
                or reg['maximum_error'] != 1e-5):
            raise ValueError('Sampler prediction provenance/scope differs')
        source = os.path.relpath(path, base)
        bank = sha(lab / reg['bank'])
        expected_scenes={r['identifier'] for r in read(lab/reg['bank'])['trajectories']}
        if {r['identifier'] for r in value['rows'] if r['horizons']==k} != expected_scenes:
            raise ValueError('Sampler evaluation omits registered scenes')
        rows = sampler_accuracy_rows(value, label, k, source, bank, population)
        study['temporal'].extend(rows)
        return value, rows

    model_paths = {}
    for k in (1, 2, 3, 4):
        label = f'K={k} recurrent sampler'
        short, rows = evaluation('trained-sampler-development-blocks-01', 'trained-sampler-development-evaluation-protocol-01.json', label, k, mode='blocks')
        evaluation('trained-sampler-development-rollout-01', 'trained-sampler-development-evaluation-protocol-01.json', label, k)
        evaluation('trained-sampler-fresh-blocks-01', 'trained-sampler-fresh-evaluation-protocol-01.json', label, k, 'sampler_fresh', 'blocks')
        evaluation('trained-sampler-fresh-rollout-01', 'trained-sampler-fresh-evaluation-protocol-01.json', label, k, 'sampler_fresh')
        item = dict(transport, label=label, identifier=f'tight-sampler-k{k}', horizons=k, parameters=publication['total_learned_parameters'],
            record_kind='fit' if k == 1 else 'execution', optimizer_updates=440 if k == 1 else 0,
            weights_sha256=transport['weights_sha256'], trainable_parameters=48, inherited_pressure_weights_sha256=pressure['weights_sha256'],
            bank_sha256=rows[0]['bank_sha256'], study_id=STUDY, evaluation_format='sampler-state-rows',
            learned_inventory=short['learned_inventory'],
            evaluation_source=rows[0]['source'], architecture='Internally recurrent learned sampler',
            training_scope='One 440-call managed transport fit (48 parameters); inherits an 800-update modal pressure fit and frozen learned point/viscosity/flux assets. K=1–4 reuse the same fitted weights. All intermediate states are generated recurrently; K does not imply parallel prediction.')
        if k > 1:
            item.update(parent_label='K=1 recurrent sampler', parent_weights_sha256=transport['weights_sha256'])
        (study['temporal_models'] if k == 1 else study['temporal_executions']).append(item)
        model_paths[k] = item

    # Only complete-block captures can provide latency. Keep fresh scenes and
    # the new strict normalization separate from the earlier reserved bank.
    def timing(name, protocol, labels, population='development', accuracy_rows=None):
        value, reg = qualified(root / name, 'completed_trained_complete_block_cost_gate', protocol)
        if (value['fresh_holdout_consumed'] != (population == 'sampler_fresh') or value['threads'] != 1 or value['batch_size'] != 1
                or sha(lab / reg['bank']) != next(r['bank_sha256'] for r in study['temporal'] if r['label'] == labels[1] and r['population'] == population)
                or len(value['models'])!=1 or value['models'][0]['inventory']!=model_paths[1]['learned_inventory']):
            raise ValueError('Sampler timing bank/settings differ')
        source = os.path.relpath(root / name, base)
        study['temporal_timing_captures'].append(dict(source=source, label=Path(name).stem,
            cpu=value['cpu'], threads=1, batch_size=1, population=population, study_id=STUDY,
            warmup='First call separately recorded for each case/horizon; five subsequent complete calls', balanced_serial=False,
            clock_scope=value['clock_scope']))
        for summary in value['summaries']:
            k, regime = summary['horizons'], summary['regime']
            selected = [r for r in value['rows'] if r['horizons'] == k and ('small' if max(r['counts_xyz']) <= 16 else '32cubed') == regime]
            samples = [ms for r in selected for ms in r['raw_complete_block_ms']]
            if (any(len(r['raw_complete_block_ms']) != 5 or not r['output_shape_dtype_and_finiteness_verified'] for r in selected)
                    or statistics.median(samples) != summary['median_block_ms']
                    or not math.isclose(summary['median_block_ms'] / k, summary['median_ms_per_generated_state'], rel_tol=1e-12)):
                raise ValueError('Sampler full-block raw timing differs')
            for row in (accuracy_rows if accuracy_rows is not None else study['temporal']):
                if row['label'] != labels[k] or row['population'] != population or row['regime'] != regime:
                    continue
                study['temporal_timing'].append(dict(identifier=labels[k], recipe_identifier=f'tight-sampler-k{k}',
                    horizons=k, regime=regime, interval=row['interval'], population=population, study_id=STUDY,
                    median_ms=summary['median_ms_per_generated_state'], full_block_ms=summary['median_block_ms'],
                    timing_source=source, accuracy_source=row['source'], field_worst=row['field_worst'], accuracy_pass=row['accuracy_pass'],
                    clock_scope=value['clock_scope']))
    labels = {k: m['label'] for k, m in model_paths.items()}
    timing('trained-sampler-cpu23-cost-01.json', 'trained-sampler-cost-protocol-01.json', labels)
    timing('trained-sampler-fresh-cpu23-cost-01.json', 'trained-sampler-fresh-cost-protocol-01.json', labels, 'sampler_fresh')

    # The contiguous-load version has a paired all-intermediate byte proof;
    # inherit only development accuracy, with a separately labelled CPU capture.
    path = root / 'paired-sampler-load-cost-03.json'
    if path.exists():
        proof, _ = qualified(path, 'completed_paired_complete_sampler_comparison', 'paired-sampler-load-cost-protocol-03.json')
        if not proof['all_intermediate_outputs_bitwise_equal'] or not proof['learned_tensors_unchanged']:
            raise ValueError('Sampler load-path accuracy inheritance not established')
        inherited = []
        contiguous = {}
        for k, parent in model_paths.items():
            label = parent['label'] + ' · contiguous'
            item = dict(parent, label=label, identifier=parent['identifier']+'-contiguous', record_kind='execution',
                optimizer_updates=0, parent_label=parent['label'], parent_weights_sha256=parent['weights_sha256'],
                qualification_source=os.path.relpath(path, base), accuracy_inheritance='paired complete-output byte proof')
            study['temporal_executions'].append(item)
            inherited.extend(dict(r, label=label) for r in study['temporal'] if r['label'] == parent['label'] and r['population'] == 'development')
            contiguous[k] = label
        study['temporal'].extend(inherited)
        timing('trained-sampler-contiguous-cpu23-cost-03.json', 'trained-sampler-contiguous-cost-protocol-03.json', contiguous, accuracy_rows=inherited)

    # A component refinement is an assembly, not another transport fit. Its
    # actual rollouts cannot borrow the original sampler's timings or replay.
    refined = {}
    for k in (1, 2, 3, 4):
        folder = root / 'refined-pressure-sampler-development-rollout-02'
        if not (folder / 'results.json').exists():
            break
        label = f'K={k} recurrent sampler · refined pressure'
        value, rows = evaluation(folder.name, 'refined-pressure-sampler-development-protocol-02.json', label, k)
        refinement = value['model_identity']['pressure_precision_refinement']
        fitted = pair(Path(refinement['pair']).name, 80)
        if refinement['weights_sha256'] != fitted['weights_sha256'] or refinement['receipt_sha256'] != sha(lab / refinement['pair'] / 'receipt.json'):
            raise ValueError('Refined pressure component checkpoint differs')
        item = dict(model_paths[k], label=label, identifier=f'tight-refined-k{k}', record_kind='composition' if k == 1 else 'execution',
            optimizer_updates=0, component_optimizer_updates=80, pressure_component_weights_sha256=fitted['weights_sha256'],
            learned_inventory=value['learned_inventory'],
            parent_label='K=1 recurrent sampler' if k == 1 else 'K=1 recurrent sampler · refined pressure',
            parent_weights_sha256=transport['weights_sha256'], evaluation_source=rows[0]['source'],
            training_scope='Assembly reuses the 440-call sampler fit and replaces the 800-update learned pressure component with its 80-call managed precision refinement. No new transport fit; own-history diagnostics only, no matched full-block timing or fresh confirmation.')
        (study['temporal_models'] if k == 1 else study['temporal_executions']).append(item)
        refined[k] = item

    for folder, protocol, suffix, parent in [
        ('trained-sampler-physical-rollout-04', 'trained-sampler-physical-evaluation-protocol-04.json', 'physical arithmetic', model_paths[1]),
        ('sampler-head-quantization-m30-rollout-01', 'sampler-head-quantization-m30-protocol-01.json', 'transport quantization m30', model_paths[1]),
        ('sampler-head-quantization-m40-rollout-01', 'sampler-head-quantization-m40-protocol-01.json', 'transport quantization m40', model_paths[1]),
        *[(f'sampler-precision-{tag}-rollout-01', f'sampler-precision-{tag}-protocol-01.json', tag.replace('-', ' '), refined.get(1))
          for tag in ('tensor-dct', 'quantized-locals', 'tensor-dct-quantized-locals')],
    ]:
        if parent is None or not (root / folder / 'results.json').exists():
            continue
        label = parent['label'] + ' · ' + suffix
        # Uniform head quantization records use a distinct diagnostic status.
        if suffix.startswith('transport quantization'):
            path = root / folder / 'results.json'
            value, reg = qualified(path, 'completed_uniform_head_precision_diagnostic', protocol)
            if not value['raw_managed_restore_unchanged'] or value['parameter_learning_executed']:
                raise ValueError('Precision diagnostic performed training')
            rows = sampler_accuracy_rows(value, label, 1, os.path.relpath(path, base), sha(lab / reg['bank']))
            study['temporal'].extend(rows)
        else:
            value, rows = evaluation(folder, protocol, label, 1)
        study['temporal_executions'].append(dict(parent, label=label, identifier=folder, record_kind='execution',
            optimizer_updates=0, parent_label=parent['label'], parent_weights_sha256=parent['weights_sha256'],
            evaluation_source=rows[0]['source'], execution_inventory=value['learned_inventory'],
            training_scope=parent['training_scope']+' Changed arithmetic/precision policy has its own scored rollout. No matched timing or independent confirmation; not a new training fit.'))
    study['tight_source'] = STUDY + '/recurrent-sampler-results-01.md'

    from full_ai_tight_followup_chart import enrich_tight_followups
    enrich_tight_followups(study,base,lab,read=read,qualified=qualified,pair=pair,sha=sha)
