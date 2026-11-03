"""Read subsequent sampler CPU captures, decoder refit and bounded diagnostics."""
import math
import os
from pathlib import Path
import statistics

from full_ai_tight_chart import STUDY, sampler_accuracy_rows


def enrich_tight_followups(study, base, lab, *, read, qualified, pair, sha):
    base, lab = Path(base), Path(lab)
    docs, root = base / STUDY, lab / 'runs' / STUDY
    records = {m['label']: m for m in study['temporal_models'] + study['temporal_executions']}
    transport = records['K=1 recurrent sampler']['weights_sha256']

    def publication(name, status):
        path = docs / name
        if not path.exists():
            return None
        value = read(path)
        if value['status'] != status:
            raise ValueError('Incomplete sampler follow-up publication: ' + name)
        for source, expected in value['evidence'].items():
            target = lab / source
            if target.suffix == '.json':
                read(target, expected)
            elif sha(target) != expected:
                raise ValueError('Follow-up publication evidence changed: ' + source)
        return value

    publications = [
        ('recurrent-sampler-followup-summary-02.json', 'completed_cpu_efficiency_and_scaling_targets_unmet'),
        ('frozen-point-execution-summary-07.json', 'completed_frozen_point_execution_single_core_improved_targets_unmet'),
        ('learned-pressure-representation-summary-08.json', 'completed_learned_pressure_controls_accuracy_improved_target_unmet'),
        ('decoder-compatible-pressure-summary-09.json', 'completed_decoder_modal_independent_confirmation_joint_target_unmet'),
        ('pressure-floor-and-head-precision-summary-10.json', 'completed_consumed_scene_precision_isolation_targets_unmet'),
    ]
    published = {name: publication(name, status) for name, status in publications}

    def accuracy(folder, protocol, label, k, population='development', mode='rollout', status=None, subset_bank=None):
        path = root / folder / 'results.json'
        value, reg = qualified(path, status or 'completed_trained_full_state_' + mode, protocol)
        if (value.get('numerical_pressure_oracle_used') or value.get('native_pressure_solver_called')
                or value.get('analytic_inverse_constructed') or value.get('analytic_eigenvalues_or_inverse_scales_used')
                or not value['every_intermediate_state_scored'] or not value['every_output_typed_finite_and_hashed']
                or not value['truth_used_only_after_causal_prediction']
                or value['own_history_without_teacher_resets'] != (mode == 'rollout')
                or value['fresh_holdout_consumed'] != reg['fresh_holdout']
                or reg['maximum_error'] != 1e-5
                or value['model_identity']['transport_weights_sha256'] != transport):
            raise ValueError('Follow-up accuracy is not a causal learned generator: ' + folder)
        bank = read(lab / reg['bank'])
        identifiers = {t['identifier'] for t in bank['trajectories']}
        if {r['identifier'] for r in value['rows'] if r['horizons'] == k} != identifiers:
            raise ValueError('Follow-up accuracy population incomplete')
        refinement = value['model_identity']['pressure_precision_refinement']
        fitted = pair(Path(refinement['pair']).name, refinement['training_optimizer_calls'])
        if fitted['weights_sha256'] != refinement['weights_sha256'] or sha(lab / refinement['pair'] / 'receipt.json') != refinement['receipt_sha256']:
            raise ValueError('Follow-up pressure checkpoint differs')
        scored=value
        if subset_bank:
            subset=read(lab/subset_bank)
            selected={t['identifier'] for t in subset['trajectories']}
            if not selected<=identifiers:
                raise ValueError('Timing subset is not contained in recorded accuracy')
            scored=dict(value,rows=[r for r in value['rows'] if r['identifier'] in selected])
        rows = sampler_accuracy_rows(scored, label, k, os.path.relpath(path, base), sha(lab / (subset_bank or reg['bank'])), population)
        if subset_bank:
            for row in rows:
                row['scene_identifiers']=sorted(selected)
                row['sampling_scope']+='; matched timing subset of the 40-scene consumed evaluation'
        study['temporal'].extend(rows)
        return value, rows

    def execution(label, parent, value, rows):
        item = dict(parent, label=label, identifier=label, record_kind='execution', optimizer_updates=0,
            parent_label=parent['label'], parent_weights_sha256=parent['weights_sha256'],
            bank_sha256=rows[0]['bank_sha256'],
            evaluation_source=rows[0]['source'], execution_inventory=value['learned_inventory'],
            training_scope=parent['training_scope']+' Execution diagnostic uses its own scored outputs; no new optimizer calls or matched latency claim.')
        study['temporal_executions'].append(item)
        records[label] = item
        return item

    # Representation controls preserve learned weights; their accuracy is actual,
    # rather than inherited from the multiplication baseline.
    if published['learned-pressure-representation-summary-08.json']:
        parent = records['K=1 recurrent sampler · refined pressure · tensor dct']
        for tag in ('reciprocal-division', 'negative-log-division'):
            label = 'K=1 recurrent sampler · refined pressure · ' + tag
            value, rows = accuracy(f'learned-pressure-{tag}-rollout-08', f'learned-pressure-{tag}-protocol-08.json', label, 1)
            execution(label, parent, value, rows)

    decoder = {}
    if published['decoder-compatible-pressure-summary-09.json']:
        component = pair('decoder-modal-pair-09', 80)
        result, reg = qualified(root / 'decoder-modal-component-09/results.json',
            'completed_offline_learned_pressure_component_evaluation', 'decoder-modal-component-protocol-09.json')
        if (result['weights_sha256'] != component['weights_sha256'] or not result['teacher_rhs_used']
                or result['causal_full_state_generator'] or result['own_history_rollout']
                or len(result['rows']) != reg['expected_pressure_states']):
            raise ValueError('Decoder component scope differs')
        study['tight_components'].append(dict(component, name='Decoder-compatible modal pressure', component_only=True,
            inherited_optimizer_updates=880, source_evaluation=os.path.relpath(root / 'decoder-modal-component-09/results.json', base),
            groups=result['summaries']))
        for k in (1, 2, 3, 4):
            label = f'K={k} recurrent sampler · decoder pressure'
            value, rows = accuracy('decoder-modal-all-horizon-development-09', 'decoder-modal-all-horizon-development-protocol-09.json', label, k)
            accuracy('decoder-modal-fresh-blocks-09', 'decoder-modal-fresh-evaluation-protocol-09.json', label, k, 'decoder_fresh', 'blocks')
            accuracy('decoder-modal-fresh-rollout-09', 'decoder-modal-fresh-evaluation-protocol-09.json', label, k, 'decoder_fresh')
            parent = records[f'K={k} recurrent sampler']
            item = dict(parent, label=label, identifier=f'tight-decoder-k{k}', record_kind='composition' if k == 1 else 'execution',
                optimizer_updates=0, component_optimizer_updates=80, pressure_component_weights_sha256=component['weights_sha256'],
                parent_label='K=1 recurrent sampler' if k == 1 else 'K=1 recurrent sampler · decoder pressure',
                parent_weights_sha256=transport, evaluation_source=rows[0]['source'], learned_inventory=value['learned_inventory'],
                training_scope='Existing 440-call sampler transport fit with an 80-call decoder-compatible pressure refinement; pressure inherits 800+80 earlier calls. K1–4 share weights. Independent 16-scene confirmation is distinct from prior banks. Timed implementations have complete-block byte checks; final timing-policy own-history qualification remains separate.')
            (study['temporal_models'] if k == 1 else study['temporal_executions']).append(item)
            records[label] = item
            decoder[k] = item
        for tag in ('tensor', 'division'):
            label = 'K=1 recurrent sampler · decoder pressure · ' + tag
            value, rows = accuracy(f'decoder-modal-{tag}-development-rollout-09', f'decoder-modal-{tag}-development-protocol-09.json', label, 1)
            execution(label, decoder[1], value, rows)

    # Two previously consumed failing scenes are attribution data. Their maxima
    # must never replace the frozen candidate's 16-scene confirmation scores.
    if published['pressure-floor-and-head-precision-summary-10.json'] and decoder:
        specs = [
            (f'conditioning-{tag}-learned-10', f'conditioning-{tag}-learned-protocol-10.json', 'conditioning ' + tag,
             'completed_consumed_scene_conditioning_diagnostic') for tag in ('gaussian', 'scalar')
        ] + [
            (f'decoder-head-{tag}-learned-10', f'decoder-head-{tag}-learned-protocol-10.json', 'head ' + tag,
             'completed_consumed_scene_head_precision_diagnostic') for tag in ('m30', 'm40')
        ] + [
            (f'decoder-pressure-{tag}-10', f'decoder-pressure-{tag}-protocol-10.json', tag,
             'completed_consumed_scene_learned_pressure_arithmetic_diagnostic') for tag in ('tensor-multiplier', 'tensor-division', 'fftw-division')
        ]
        for folder, protocol, suffix, status in specs:
            label = 'K=1 recurrent sampler · decoder pressure · ' + suffix
            value, rows = accuracy(folder, protocol, label, 1, 'decoder_consumed', status=status)
            item = execution(label, decoder[1], value, rows)
            item['training_scope'] += ' Only two consumed failure scenes; development diagnosis, not independent confirmation.'
            item['evaluation_population'] = 'decoder_consumed'

    def paired_timing(name, protocol, labels, population, expected_status, baseline_policy, proof=None):
        value, reg = qualified(root / name, expected_status, protocol)
        if (not value['all_intermediate_outputs_bitwise_equal'] or not value['learned_tensors_unchanged']
                or value['model_identity']['transport_weights_sha256'] != transport
                or value['execution_team']['observed_threads'] != len(value['cores'])
                or set(value['execution_team']['worker_cpus']) != set(value['cores'])):
            raise ValueError('Paired execution identity/resources differ: ' + name)
        bank = read(lab / reg['bank'])
        identifiers = {t['identifier'] for t in bank['trajectories']}
        if {r['identifier'] for r in value['rows']} != identifiers:
            raise ValueError('Paired timing scene population differs')
        for k, parent in labels.items():
            actual = [r for r in study['temporal'] if r['label'] == parent['label'] and r.get('population', 'development') == population]
            if not actual or any(r['bank_sha256'] != sha(lab / reg['bank']) for r in actual):
                raise ValueError('Timing has no matching accuracy bank: ' + name)
            if proof and value['model_identity'] != proof['model_identity']:
                raise ValueError('Timing and execution graph checkpoint differ')
            refinement = value['model_identity'].get('pressure_precision_refinement')
            if refinement and parent.get('pressure_component_weights_sha256') != refinement['weights_sha256']:
                raise ValueError('Timing borrows another pressure checkpoint')
        source = os.path.relpath(root / name, base)
        variants = list(value['summaries'][0]['median_block_ms'])
        for variant in variants:
            # serial_v5 performs serial work even within a larger allocated
            # OpenMP process; retain the allocation and worker distinction.
            active_cores = 1 if variant == 'serial_v5' else len(value['cores'])
            policy = baseline_policy if variant == 'baseline' else variant
            capture_id = source + '#' + variant
            label = Path(name).stem + ' · ' + policy + f' · {active_cores} core(s), CPU ' + ','.join(map(str, value['cores']))
            study['temporal_timing_captures'].append(dict(id=capture_id, source=source, label=label, cpu=value['cores'][0],
                threads=active_cores, core_count=active_cores, allocated_cpus=value['cores'], population=population, study_id=STUDY,
                warmup='One complete warm call; five alternating paired repeats per scene/start/horizon', balanced_serial=True,
                execution_policy=policy, clock_scope=value['clock_scope']))
            for summary in value['summaries']:
                k, regime = summary['horizons'], summary['regime']
                selected = [r for r in value['rows'] if r['horizons'] == k and ('small' if max(r['counts_xyz']) <= 16 else '32cubed') == regime]
                expected = {(scene, start) for scene in identifiers if any(r['identifier'] == scene for r in selected) for start in reg['starts']}
                if {(r['identifier'], r['start']) for r in selected} != expected or len(selected) != len(expected):
                    raise ValueError('Paired timing window coverage differs')
                samples = [ms for r in selected for ms in r['raw_complete_block_ms'][variant]]
                if (any(not r['all_intermediate_physical_outputs_bitwise_equal'] or len(r['raw_complete_block_ms'][variant]) != reg['repeats'] for r in selected)
                        or not all(math.isfinite(ms) and ms > 0 for ms in samples)
                        or statistics.median(ms/k for ms in samples) != summary['median_ms_per_state'][variant]
                        or not math.isclose(summary['median_block_ms'][variant] / k, summary['median_ms_per_state'][variant], rel_tol=1e-12)):
                    raise ValueError('Paired full-block raw timing differs')
                for row in study['temporal']:
                    if row['label'] != labels[k]['label'] or row.get('population', 'development') != population or row['regime'] != regime:
                        continue
                    study['temporal_timing'].append(dict(identifier=labels[k]['label'], recipe_identifier=labels[k]['identifier'],
                        horizons=k, regime=regime, interval=row['interval'], population=population, study_id=STUDY,
                        median_ms=summary['median_ms_per_state'][variant], full_block_ms=summary['median_block_ms'][variant],
                        p95_block_ms=summary['p95_block_ms'][variant], maximum_block_ms=summary['maximum_block_ms'][variant],
                        timing_source=source, capture_id=capture_id, accuracy_source=row['source'],
                        field_worst=row['field_worst'], accuracy_pass=row['accuracy_pass'],
                        execution_policy=policy, raw_variant=variant, core_count=active_cores, allocated_cpus=value['cores'],
                        accuracy_inheritance='Paired complete-block bytes; own-history scores measured with the recorded accuracy execution',
                        clock_scope=value['clock_scope']))

    original = {k: records[f'K={k} recurrent sampler'] for k in (1, 2, 3, 4)}
    if published['recurrent-sampler-followup-summary-02.json']:
        for tag in ('e1', 'p1', 'p2', 'p4'):
            proof, _ = qualified(root / f'sampler-cores-{tag}-graph-06.json',
                'passed_parallel_complete_graph_execution', f'sampler-cores-{tag}-graph-protocol-06.json')
            paired_timing(f'sampler-cores-{tag}-cost-06.json', f'sampler-cores-{tag}-cost-protocol-06.json', original,
                'development', 'completed_paired_parallel_sampler_comparison', 'serial_v5', proof)
    if published['frozen-point-execution-summary-07.json']:
        for tag in ('p1', 'p4'):
            proof, _ = qualified(root / f'frozen-sampler-{tag}-graph-07.json',
                'passed_parallel_complete_graph_execution', f'frozen-sampler-{tag}-graph-protocol-07.json')
            paired_timing(f'frozen-sampler-{tag}-cost-07.json', f'frozen-sampler-{tag}-cost-protocol-07.json', original,
                'development', 'completed_paired_frozen_sampler_comparison', 'serial_v5' if tag == 'p1' else 'parallel_v6', proof)
    if decoder:
        for tag in ('p1', 'p2', 'p4'):
            proof, _ = qualified(root / f'decoder-modal-{tag}-execution-qualification-09.json',
                'passed_parallel_complete_graph_execution', f'decoder-modal-{tag}-execution-qualification-protocol-09.json')
            paired_timing(f'decoder-modal-{tag}-fresh-cost-09.json', f'decoder-modal-{tag}-fresh-cost-protocol-09.json', decoder,
                'decoder_fresh', 'completed_paired_frozen_sampler_comparison', 'serial_v5' if tag == 'p1' else 'parallel_v6', proof)

    # Application clocks have distinct state exchange and verification costs.
    # Preserve their matched native controls without treating them as inference.
    study['temporal_applications'] = []
    if published['recurrent-sampler-followup-summary-02.json']:
        for tag in ('p1', 'p2', 'p4'):
            path = root / f'sampler-cores-{tag}-application-06/results.json'
            value = read(path)
            if value['status'] != 'completed_matched_application_comparison':
                raise ValueError('Incomplete matched sampler application')
            study['temporal_applications'].append(dict(source=os.path.relpath(path, base),
                core_count=int(tag[1:]), groups=value['summaries'], scope=value['application_clock_scope'] if 'application_clock_scope' in value else value.get('clock_scope')))
    from full_ai_history_chart import enrich_pressure_history
    enrich_pressure_history(study,base,lab,read=read,qualified=qualified,pair=pair,sha=sha,
        publication=publication,accuracy=accuracy,execution=execution,paired_timing=paired_timing)
