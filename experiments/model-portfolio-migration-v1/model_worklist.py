"""Build temporary model cards and benchmark work items from retained evidence.

Fits, checkpoints, aliases and execution implementations remain separate. This
is a migration input, not a replacement for Tuldok's model and lineage owners.
No weights are loaded, no inference runs and no source-only candidate is silently
discarded. Unknown scientific purpose and benchmark applicability remain explicit.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    'rheon-water-learning-training-v1': 'pressure_proposal',
    'rheon-sealed-water-training-v1': 'pressure_proposal',
    'rheon-cuda-pressure-training-v1': 'pressure_proposal',
    'rheon-variable-pressure-training-v1': 'pressure_proposal',
    'rheon-variable-pressure-training-v2': 'pressure_proposal',
    'rheon-local-pressure-training-receipt-v1': 'pressure_component',
    'rheon-learned-modal-pressure-result-v1': 'pressure_component',
    'rheon-shared-spectral-managed-result-v1': 'pressure_component',
    'rheon-point-sampling-fit-v1': 'transport_component',
    'rheon-viscosity-calibration-fit-v1': 'viscosity_component',
    'rheon-neural-head-training-result-v1': 'pressure_head_component',
    'rheon-temporal-training-result-v1': 'full_state_generator',
    'rheon-causal-request-training-result-v1': 'pressure_request_component',
    'rheon-learned-recurrent-sampler-head-result-v1': 'recurrent_transport_component',
    'rheon-macrostep-readout-managed-result-v1': 'full_state_readout_component',
    'rheon.guided-pressure-managed-fit.v1': 'guided_pressure_component',
    'rheon.guided-shared-managed-fit.v1': 'guided_shared_component',
    **{'rheon-internal-residual-managed-v' + str(i): 'full_state_correction_component' for i in range(1, 5)},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_outputs(outputs):
    if not isinstance(outputs, dict):
        return []
    rows = []
    for name, record in outputs.items():
        if not isinstance(record, dict) or not re.fullmatch('[a-f0-9]{64}', str(record.get('sha256', ''))):
            continue
        if record.get('role') == 'checkpoint' or re.search(r'weight|checkpoint', name, re.I):
            if Path(name).suffix.lower() not in ('.npz', '.pt', '.pth', '.safetensors', '.onnx'):
                continue
            stage = ('initial' if name.startswith(('initial-', 'checkpoint-000.')) else
                     'selected' if name in ('weights.npz', 'model.pt', 'model.pth') else 'retained_snapshot')
            rows.append(dict(name=name, sha256=record['sha256'], bytes=record.get('bytes'), stage=stage))
    return rows


def draft_fits(reconciled, runs):
    by_receipt = defaultdict(list)
    for record in runs['records']:
        for receipt in record['recipe_receipts']:
            by_receipt[receipt['sha256']].append(dict(
                catalog_id=record['namespace'], run_id=record['run']['id'],
                epoch=record['epoch'], observed_in=record['observed_in'],
                graph_name=record['run']['graph_name'], state=record['run']['state']))
    cards, nonfits = [], []
    for group in reconciled['training_receipt_groups']:
        path = ROOT / group['paths'][0]
        if sha(path) != group['receipt_sha256']:
            raise ValueError('Training receipt changed after reconciliation')
        raw = json.loads(path.read_text())
        if group['mode'] != 'train':
            nonfits.append(dict(receipt_sha256=group['receipt_sha256'], paths=group['paths'],
                               mode=group['mode'], classification='recovery_or_summary_not_a_new_fit',
                               runs=by_receipt[group['receipt_sha256']]))
            continue
        task = TASKS.get(group['schema'], 'unresolved_training_task')
        outputs = checkpoint_outputs(group['outputs'])
        architecture = group['architecture']
        runs_for_receipt = by_receipt[group['receipt_sha256']]
        name = (runs_for_receipt[0]['graph_name'] if runs_for_receipt else None) or str(path.parent.parent.name)
        method = ({k: architecture[k] for k in ('schema', 'input_channels', 'encoder_widths',
                  'bottleneck_width', 'decoder_widths', 'kernel', 'activation', 'dtype',
                  'pressure_dtype', 'correction', 'learned_coarse_revision', 'student_distillation',
                  'restrictions', 'counts_supported', 'max_cells') if k in architecture}
                  if isinstance(architecture, dict) else architecture)
        cards.append(dict(item_id='fit-' + group['receipt_sha256'][:16], kind='training_attempt',
                          name=name, task=task, schema=group['schema'], method=method,
                          receipt_sha256=group['receipt_sha256'], receipt_paths=group['paths'],
                          recorded_Run_observations=runs_for_receipt,
                          optimizer_updates=raw.get('optimizer_updates'), parameter_fits=raw.get('parameter_fits'),
                          declared_parameter_count=raw.get('parameter_count', raw.get('parameters')),
                          training_closure=group['closure'], checkpoints=outputs,
                          raw_receipt_status=raw.get('status'), declared_runtime=raw.get('runtime'),
                          named_entries=[],
                          purpose=dict(status='source_review_required',
                                       evidence='Captured architecture, training configuration and linked named entries.'),
                          outcome=dict(status='retained_training_evidence_only',
                                       fresh_common_benchmark='pending',
                                       scientific_promotion=raw.get('scientific_acceptance_or_model_promotion')),
                          lesson=dict(status='review_required_no_automatic_causal_claim'),
                          reproduction=dict(receipt=str(path.relative_to(ROOT)),
                                            source_and_data_closure='Retained hashes; clean reproduction not yet exercised.'),
                          applicability=dict(status='adapter_and_contract_review_required',
                                             task=task, fresh_result=None)))
    ids = [c['item_id'] for c in cards]
    if len(ids) != len(set(ids)):
        raise ValueError('Worklist key collision')
    return cards, nonfits


def path_evidence(record, fit):
    """Prefer a named Project/pair over mere equality with an inherited tensor."""
    if not isinstance(record, dict):
        return 0
    prefixes = []
    artifacts = record.get('replay_artifacts')
    if isinstance(artifacts, str):
        location = (ROOT / 'docs/plans' / artifacts).resolve()
        if location.is_relative_to(ROOT):
            prefixes.append((str(location.parent.relative_to(ROOT)) + '/', 4))
    source = record.get('source')
    if isinstance(source, str) and source.endswith('/receipt.json'):
        location = (ROOT / 'docs/plans' / source).resolve()
        if location.is_relative_to(ROOT):
            prefixes.append((str(location.parent.relative_to(ROOT)) + '/', 3))
    for field in ('training_source', 'run_root'):
        declared = record.get(field)
        if isinstance(declared, str):
            location = (ROOT / 'docs/plans' / declared).resolve()
            if location.is_relative_to(ROOT):
                if field == 'training_source':
                    location = location.parent
                    if location.name in ('restore', 'train'):
                        location = location.parent
                prefixes.append((str(location.relative_to(ROOT)) + '/', 3 if field == 'training_source' else 1))
    score = max((rank for prefix, rank in prefixes if any(p.startswith(prefix) for p in fit['receipt_paths'])), default=0)
    pair = record.get('pair')
    if isinstance(pair, str) and any('/' + pair + '/' in p for p in fit['receipt_paths']):
        score = max(score, 3)
    return score


def assembly_evidence(record, fits):
    if not isinstance(record, dict) or record.get('record_kind') not in ('composition', 'execution'):
        return None
    source = record.get('source')
    if not isinstance(source, str) or not source.endswith('/receipt.json'):
        return None
    path = (ROOT / 'docs/plans' / source).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size > 1024 * 1024:
        return None
    value = json.loads(path.read_text())
    if (value.get('status') != 'assembled_from_exact_managed_CPU_parents'
            or value.get('weights_sha256') != record.get('weights_sha256')):
        return None
    parents = []
    for role, parent in value['parents'].items():
        matches = [f['item_id'] for f in fits
                   if any(r['run_id'] == parent['training_run_id'] for r in f['recorded_Run_observations'])
                   and any(c['sha256'] == parent['weights_sha256'] for c in f['checkpoints'])]
        parents.append(dict(role=role, evidence=parent, fit_item_ids=matches))
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), parents=parents,
                all_parent_fits_located=all(len(p['fit_item_ids']) == 1 for p in parents),
                new_full_state_optimizer_updates=record.get('optimizer_updates'))


def link_entries(reconciled, fits):
    selected, snapshots = defaultdict(list), defaultdict(list)
    for fit in fits:
        for output in fit['checkpoints']:
            snapshots[output['sha256']].append(fit)
            if output['stage'] == 'selected':
                selected[output['sha256']].append(fit)
    entries = []
    for row in reconciled['declarations']:
        scope = row['discovery_key'].split('/')[0]
        record = row['record']
        metadata = row['metadata']
        weight = metadata.get('weights_sha256')
        candidates = (selected[weight] or snapshots[weight]) if weight else []
        scored = [(path_evidence(record, f), f) for f in fits
                  if not weight or any(c['sha256'] == weight for c in f['checkpoints'])]
        best_score = max((score for score, _ in scored), default=0)
        anchored = [f for score, f in scored if score == best_score] if best_score else []
        candidates = anchored or candidates
        unique = {f['item_id']: f for f in candidates}
        assembly = assembly_evidence(record, fits)
        if assembly and assembly['all_parent_fits_located']:
            disposition, fit_id = 'recorded_composition_of_exact_parent_checkpoints', None
        elif len(unique) == 1:
            disposition = 'linked_by_project_or_pair' if anchored else 'unique_declared_checkpoint_producer'
            fit_id = next(iter(unique))
        elif unique:
            disposition, fit_id = 'ambiguous_producers_require_source_review', None
        elif weight:
            disposition, fit_id = 'checkpoint_without_training_receipt_link', None
        else:
            disposition, fit_id = 'resolve_reference_components_or_execution_source', None
        kind = (record['record_kind'] if isinstance(record, dict) and record.get('record_kind') in ('composition', 'execution') else
                'model_alias' if scope in ('cohorts', 'legacy-models', 'temporal-models', 'tight-components') else
                'model_reference' if scope == 'pressure-models' else
                'control' if scope == 'controls' else 'execution_or_component')
        entry = dict(item_id=row['discovery_key'], kind=kind, source_pointer=row['source_pointer'],
                     label=(record.get('id') or record.get('identifier') or record.get('pair') or record.get('key') or
                            record.get('label') or record.get('name')) if isinstance(record, dict) else record,
                     record=record, declared_weights_sha256=weight,
                     checkpoint_disposition=row['checkpoint_disposition'], checkpoint_path=row['checkpoint_path'],
                     fit_link_disposition=disposition, fit_item_id=fit_id,
                     checkpoint_assembly=assembly,
                     candidate_fit_item_ids=sorted(unique),
                     applicable_benchmark='contract_and_adapter_review_required',
                     fresh_benchmark_status='pending')
        entries.append(entry)
        if fit_id:
            unique[fit_id]['named_entries'].append(row['discovery_key'])
    cohorts = {e['label']: e for e in entries if e['item_id'].startswith('cohorts/')}
    for entry in entries:
        scope = entry['item_id'].split('/')[0]
        if scope in ('pressure-models', 'controls') and isinstance(entry['record'], str):
            target = cohorts.get(entry['record'])
            if target is not None:
                entry.update(fit_link_disposition='reference_to_named_cohort',
                             referenced_item_id=target['item_id'], fit_item_id=target['fit_item_id'],
                             declared_weights_sha256=target['declared_weights_sha256'],
                             checkpoint_path=target['checkpoint_path'],
                             checkpoint_disposition=target['checkpoint_disposition'])
        elif scope == 'residual-fits':
            path = (ROOT / 'docs/plans' / entry['record']['source']).resolve()
            if path.is_relative_to(ROOT) and path.is_file() and path.stat().st_size <= 16 * 1024 * 1024:
                report = json.loads(path.read_text())
                matches = [f for f in fits if any(r['run_id'] == report.get('training_run_id')
                           for r in f['recorded_Run_observations'])]
                if len(matches) == 1:
                    fit = matches[0]
                    if report.get('optimizer_updates') != fit['optimizer_updates']:
                        raise ValueError('Reported residual training updates contradict the actual receipt')
                    entry.update(fit_link_disposition='reported_training_Run_identity', fit_item_id=fit['item_id'],
                                 fit_link_evidence=dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                                                        training_run_id=report['training_run_id']))
                    fit['named_entries'].append(entry['item_id'])
    return entries


def link_expanded_evidence(reconciled, entries, fits):
    """Read the retained producer's explicit variant/source associations."""
    local_path = ROOT / reconciled['inputs'][0]['path']
    local = json.loads(local_path.read_text())
    path = ROOT / local['dashboard']['path']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != local['dashboard']['sha256']:
        raise ValueError('Dashboard changed after source reconciliation')
    match = re.search(rb'<script id="chart-data" type="application/json">(.*?)</script>', raw, re.S)
    expanded = json.loads(match[1])['full_ai_study']['expanded_models']
    by_variant = {e['record']['id']: e for e in entries if e['item_id'].startswith('expanded-variants/')}
    routes = {key: dict(sources=set(), K=set(), grid_density=set(), populations=set(), scopes=set(), rows=Counter())
              for key in by_variant}
    unknown = Counter()
    for collection in ('accuracy', 'timing', 'traces'):
        for row in expanded[collection]:
            identifier = row['variant']
            if identifier not in routes:
                unknown[identifier] += 1
                continue
            target = routes[identifier]
            target['rows'][collection] += 1
            if isinstance(row.get('source'), str):
                target['sources'].add(row['source'])
            if isinstance(row.get('timing_source'), str):
                target['sources'].add(row['timing_source'])
            if isinstance(row.get('K'), int):
                target['K'].add(row['K'])
            if 'counts_xyz' in row and 'cells_per_metre' in row:
                target['grid_density'].add((tuple(row['counts_xyz']), row['cells_per_metre']))
            if isinstance(row.get('population'), str):
                target['populations'].add(row['population'])
            if isinstance(row.get('scope'), str):
                target['scopes'].add(row['scope'])
    residuals = {e['record']['id']: e for e in entries if e['item_id'].startswith('residual-fits/')}
    student_path = ROOT / 'docs/plans/full-ai-data-diversity-v1/macrostep-student-pilot-results-01.json'
    students = json.loads(student_path.read_text())['students']
    for identifier, entry in by_variant.items():
        route = routes[identifier]
        entry.update(kind='expanded_' + entry['record']['kind'],
                     fit_link_disposition='declared_' + entry['record']['kind'] + '_requires_execution_binding',
                     historical_evidence_routes=dict(
                         sources=sorted(route['sources']), observed_K=sorted(route['K']),
                         observed_grid_density=[dict(counts_xyz=list(grid), cells_per_metre=density)
                                                for grid, density in sorted(route['grid_density'])],
                         populations=sorted(route['populations']), timing_scopes=sorted(route['scopes']),
                         grouped_rows=dict(route['rows']),
                         provenance=dict(path=str(path.relative_to(ROOT)), sha256=local['dashboard']['sha256']),
                         scope='Historical producer associations only; exact implementation/parameter binding and fresh benchmark remain pending.'))
        if identifier in residuals:
            target = residuals[identifier]
            entry.update(fit_item_id=target['fit_item_id'], referenced_item_id=target['item_id'],
                         fit_link_disposition='reference_to_recorded_residual_fit')
        elif identifier in ('student3', 'student4'):
            student = next(s for s in students if s['stages'] == int(identifier[-1]))
            matches = [f for f in fits if any(r['run_id'] == student['training_run_id']
                       for r in f['recorded_Run_observations'])]
            if len(matches) != 1:
                raise ValueError('Declared temporal student fit not uniquely accounted')
            entry.update(fit_item_id=matches[0]['item_id'], fit_link_disposition='reported_temporal_readout_training_Run',
                         fit_link_evidence=dict(path=str(student_path.relative_to(ROOT)), sha256=sha(student_path),
                                                training_run_id=student['training_run_id']))
    for entry in entries:
        if entry['item_id'].startswith('pilot-captures/'):
            entry.update(kind='benchmark_capture', fit_link_disposition='capture_not_an_additional_fit')
    return dict(variants=len(routes), variants_with_source_routes=sum(bool(r['sources']) for r in routes.values()),
                undeclared_measurement_variant_ids=dict(unknown))


def build(reconciled_path, runs_path, output, summary_path, index_path):
    if output.exists() or summary_path.exists() or index_path.exists():
        raise ValueError('Fresh model worklist and index required')
    reconciled, runs = (json.loads(p.read_text()) for p in (reconciled_path, runs_path))
    if runs['issues']:
        raise ValueError('Resolve Run/publication contradictions before drafting cards')
    fits, nonfits = draft_fits(reconciled, runs)
    entries = link_entries(reconciled, fits)
    expanded_routes = link_expanded_evidence(reconciled, entries, fits)
    sources = []
    for candidate in reconciled['prototype_source_candidates']:
        path = candidate['path']
        if '/test_' in path:
            classification = 'test_fixture'
        elif any(any(b.endswith('.Module') for b in c['bases']) for c in candidate['classes']):
            classification = 'neural_module_source_requires_semantic_classification'
        elif candidate['classes']:
            classification = 'execution_or_helper_source_requires_semantic_classification'
        else:
            classification = 'diagnostic_or_function_model_source_requires_review'
        sources.append(dict(**candidate, structural_classification=classification))
    result = dict(schema='rheon.model-card-worklist.v1', created_unix=time.time(),
                  status='draft_cards_and_coverage_not_complete_portfolio',
                  fit_cards=fits, nonfit_receipts=nonfits, named_entries=entries,
                  source_candidates=sources,
                  expanded_evidence_route_accounting=expanded_routes,
                  uncovered_checkpoint_states='All 524 byte identities remain in the checkpoint inventory; intermediate/unmatched states need context classification.',
                  inputs=[dict(path=str(p), sha256=sha(p)) for p in (reconciled_path, runs_path)],
                  source_sha256=sha(Path(__file__)), new_inference_calls=0,
                  complete_model_count_established=False,
                  limitations=['Source-derived task labels are routing hints; exact applicability requires the captured implementation.',
                               'A linked producer establishes retained evidence, not performance equivalence or architecture ancestry.',
                               'Unknown purpose, lessons, component assemblies and reproduction remain explicit review work.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    summary = dict(schema=result['schema'], status=result['status'], output=str(output), output_sha256=sha(output),
                   fit_cards=len(fits), nonfit_receipts=len(nonfits), task_routing=dict(Counter(f['task'] for f in fits)),
                   fits_with_saved_Run_records=sum(bool(f['recorded_Run_observations']) for f in fits),
                   fits_with_checkpoint_outputs=sum(bool(f['checkpoints']) for f in fits),
                   named_entries=len(entries), fit_links=dict(Counter(e['fit_link_disposition'] for e in entries)),
                   expanded_evidence_route_accounting=expanded_routes,
                   source_classifications=dict(Counter(s['structural_classification'] for s in sources)),
                   new_inference_calls=0, complete_model_count_established=False)
    with summary_path.open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    lines = ['# Named model and execution worklist', '',
             'Generated migration index. Links locate retained evidence; all fresh benchmarks are pending.',
             'This table is not the native project or a complete model count.', '',
             '| Collection | Alias | Fit evidence | Checkpoint |', '| --- | --- | --- | --- |']
    for entry in entries:
        label = str(entry['label']).replace('|', '\\|').replace('\n', ' ')
        lines.append('| ' + ' | '.join((entry['item_id'].split('/')[0], label,
                                        entry['fit_item_id'] or entry['fit_link_disposition'],
                                        entry['checkpoint_disposition'])) + ' |')
    with index_path.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reconciled', 'runs', 'output', 'summary', 'index'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.reconciled, args.runs, args.output, args.summary, args.index)
