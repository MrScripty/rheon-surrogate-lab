"""Prepare scoped outcomes for the twenty remaining native component cards.

Only proposed notes are written. Historical receipts, names, ancestry, sources
and accepted benchmark authorities remain unchanged. Scalar source paths are
explicit lab bindings; this proposal does not claim portable source closure.
"""
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, PROJECT, header, read, save, binding, digest, encode, require
from adopt_matched_generator_outcome_notes_02 import target_records
from prepare_matched_generator_outcome_notes import replace_paragraph, TAIL, LABELS

OUTPUT = RUN/'native-component-outcome-notes-01'
GUIDE = 'notes/00-models.portfolio.json'
INDEX = 'notes/portfolio-index.json'
COVERAGE = DOC/'reports/common-fit-benchmark-coverage-16.json'
LEADS = DOC/'reports/multistep-lead-statistics-01.json'
MATERIAL = RUN/'remaining-component-material-01.json'


def outcome(row):
    a, task = row['accuracy_benchmark'], row['task']
    if task in ('full_state_correction_component', 'full_state_readout_component',
                'guided_shared_component', 'guided_pressure_component'):
        w = a['worst']['worst_relative']
        if task == 'full_state_correction_component':
            population = '783 supplied internal-stage requests; partial groups are not physical simulation outputs'
        elif task == 'full_state_readout_component':
            population = '58 active next-output windows from 29 consumed N8 parents; 119 N1 parents bypass this readout'
        else:
            population = '2,880 complete guided blocks / 7,200 physical outputs on 24 consumed scenes, at 30/60/120 outputs per second and K1–4'
        text = (f'CUDA evaluation: {population}. Worst accepted relative L2 errors are '
                f'pressure {w["pressure"]:.6g}, combined velocity {w["velocity"]:.6g}, '
                f'tracer {w["tracer"]:.6g} and equation {w["equation"]:.6g}. '
                'Original denominator floors remain part of these metric definitions. ')
    elif 'worst_native' in a:
        w = a['worst_native']
        text = (f'CUDA pressure evaluation on the matched 768-request native cohort union: '
                f'worst mean-free pressure relative L2 {w["pressure_mean_free_relative_l2"]["worst"]:.6g}; '
                f'worst equation relative L2 {w["equation_relative_l2"]["worst"]:.6g}. '
                'The two applicable water requests are a separate population; 126 other water requests '
                'are unsupported by the original shape guard. Native fluid density is 1.25 kg/m3, '
                'versus 1,000 kg/m3 for the water subset. Declared 32-cubed support has not been exercised. ')
    elif task == 'pressure_component':
        w = a['worst']
        text = (f'CUDA evaluation on 144 native RHS pressure requests: worst mean-free pressure '
                f'relative L2 {w["pressure_mean_free_relative_l2"]:.6g}; worst equation relative L2 '
                f'{w["common_equation_relative_l2"]:.6g}. This native-RHS interface differs from the '
                'common 128-request pressure CNN interface and performs no simulation advance. ')
    elif task == 'pressure_request_component':
        text = (f'CUDA evaluation on 2,048 original training-parent transitions: worst normalized '
                f'future-RHS q relative L2 {a["worst"]["worst_q_relative_l2_nonzero_targets"]:.6g}. '
                'The output is q, not pressure; pressure inversion and transport are outside this call. ')
    elif task == 'recurrent_transport_component':
        w = a['worst']
        text = (f'CUDA local-sampler evaluation on 576 groups / 77,998 consumed training points: '
                f'worst nonzero-target group relative L2 is {w["middle"]["worst_relative_l2_nonzero_target_groups"]:.6g} '
                f'for the middle velocity-equivalent vector and {w["end"]["worst_relative_l2_nonzero_target_groups"]:.6g} '
                'for the end negative sampled velocity. These are internal sampling targets, not two physical output steps. '
                'Zero-target points retain absolute errors separately. ')
    else:
        require(task in ('transport_component', 'viscosity_component'), 'Unreviewed component task')
        target = ('negative sampled XYZ velocity at staggered-face and tracer-cell destinations'
                  if task == 'transport_component' else 'dimensionless negative symmetric-strain stiffness response')
        text = (f'CUDA evaluation on 80 native-qualified synthetic local requests (64 fitting, 16 consumed development): '
                f'worst masked relative L2 {a["worst"]:.6g}. Output: {target}. '
                'This is an operator diagnostic, not full-grid or full-step accuracy. ')
    timing = row.get('default_benchmark')
    if timing:
        text += (f'Median of case GPU medians: {timing["case_median_median_ms"]:.6g} ms. '
                 f'Exact timing unit: {timing["timing_units"]}. ')
    else:
        text += 'Qualified GPU latency remains unavailable; instrumented accuracy replay duration is not deployment latency. '
    return text + 'No independent holdout, clean retraining or own-history drift qualification is implied.'


def lesson(row):
    task = row['task']
    if task.startswith('guided_'):
        return ('The inherited guide plus residual must be evaluated as a complete composition. '
                'Every intermediate output and endpoint retains separate statistics; a precise endpoint '
                'cannot conceal inaccurate intermediate frames. These selected weights reproduce their '
                'original physical scores, so the observed errors are not evidence of migration corruption. '
                'Keep best-worst, best-mean and last-evaluated parameter states distinct.')
    if task == 'full_state_correction_component':
        return ('Supplied fine-history stage context can measure a correction head but cannot establish '
                'causal trajectory accuracy. Groups of widths 2/2/3 are internal advances, not K=3 outputs. '
                'Compare the coupled state against its supplied zero-correction baseline and retain '
                'scaled head-delta errors separately; fitting direct deltas need not improve pressure.')
    if task == 'full_state_readout_component':
        return ('The raw coefficient/QR reconstruction check and final field accuracy answer different '
                'questions. Score the actual cast pressure and carriers, and include backbone stage '
                'generation in any complete-generator cost. Three/four internal readout stages do not '
                'mean three/four emitted physical frames.')
    if task == 'pressure_request_component':
        return ('A precise downstream pressure inverse cannot repair an inaccurate predicted RHS. '
                'Keep q error, original training-objective reproduction and fixed filtered-pressure '
                'diagnostics separate. Same-parent transitions outside the original sample support '
                'remain correlated diagnostics, not independent scenes.')
    if task in ('transport_component', 'viscosity_component', 'recurrent_transport_component'):
        return ('Near-exact local operator behavior does not establish full simulation accuracy or drift. '
                'Keep destination/mask, normalization, zero-target policy and units explicit; '
                'local request timings cannot be ranked as complete simulation-step latency.')
    if 'worst_native' in row['accuracy_benchmark']:
        return ('The original anisotropic pressure CNN and selected tensors now run on CUDA while '
                'fixed physical features and F64 reconstruction retain their original arithmetic. '
                'Historical CPU objectives agree within small floating-point differences, not bitwise. '
                'Keep native and water-subset results separate and expose unsupported geometry.')
    return ('A precise native-RHS pressure component does not by itself predict a future full state. '
            'Compare only matching RHS/geometry/reference populations and timing boundaries; '
            'basis preparation and cache policy remain explicit, and full-generator parity is not implied.')


def lead_table(fid, rows):
    selected = [r for r in rows if r['subject_id'] == fid]
    if not selected:
        return ''
    lines = ['## Accuracy at every physical output', '',
             'Each row scores one physical lead. Errors retain their original accepted metric definitions. '
             'Case populations pool the original grids/densities; joint filters use the retained case scalars. '
             'These are consumed diagnostic windows, not own-history drift. No per-lead latency is inferred.', '',
             '| Outputs/s | K | Lead | Role | Horizon (s) | Metric | N | Best | Worst | Mean | Median | p95 | Worst scene |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in selected:
        s = r['statistics']; case = s['worst_case']
        values = [f'{1/r["output_dt_s"]:g}', str(r['K']), str(r['lead']), r['output_role'],
                  f'{r["physical_lead_dt_s"]:.9g}', r['metric'], str(s['samples'])]
        values += [f'{s[k]:.9g}' for k in ('best', 'worst', 'mean', 'median', 'p95')]
        values += [case['identifier']+'; start '+str(case.get('start_fine_step',case.get('start','unavailable')))]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)+'\n\n'


def run():
    require(not OUTPUT.exists(), 'Preserve the original component outcome proposal')
    coverage, material, lead_source = read(COVERAGE), read(MATERIAL), read(LEADS)
    selected = [r for r in coverage['rows'] if r.get('accuracy_benchmark') is not None and r['task'] != 'pressure_proposal']
    require(len(selected) == 20 and len({r['fit_item_id'] for r in selected}) == 20, 'Exact twenty component outcomes required')
    sources = [COVERAGE, MATERIAL, LEADS, DOC/'reports/selected-scalar-storage-audit-01.json']
    source_specs = {str(p.relative_to(ROOT)):binding(p) for p in sources}
    for row in selected:
        spec = row['accuracy_benchmark']['independent_reduction']
        require(binding(ROOT/spec['path']) == spec, 'Accepted component reduction changed')
        source_specs[spec['path']] = spec
    _, snapshot = header(); current = target_records(snapshot)
    guide, index = read(PROJECT/GUIDE), read(PROJECT/INDEX)
    nodes = {n['id']:n for n in guide['models']}; fits = {f['item_id']:f for f in index['fits']}
    originals = {m['fit_item_id']:m for m in material['models']}
    require(len(nodes) == len(fits) == 205 and len(guide['relationships']) == 26, 'Original portfolio membership required')
    notes, staged = [], {}

    def put(path, raw, **extra):
        entry, doc = current[path]; old = doc['current']
        require((PROJECT/path).read_bytes() == old['text'].encode() and digest(old['text'].encode()) == old['sha256']
                and 0 < len(raw) <= 1048576, 'Current note or proposed note byte bound differs')
        staged[path] = raw
        notes.append(dict(kind='note', path=path, file_id=entry['id'], expected_revision=old['revision'],
                          revision=old['revision']+1, previous_sha256=old['sha256'], sha256=digest(raw), bytes=len(raw),
                          prepared_path=str((OUTPUT/path).relative_to(ROOT)), **extra))

    for row in selected:
        fid = row['fit_item_id']; node, fit = nodes[fid], fits[fid]; path = fit['card_path']
        original = (PROJECT/path).read_text(); previous = copy.deepcopy(node['summary'])
        updated = dict(previous, outcome=outcome(row), lesson=previous['lesson']+' '+lesson(row),
                       reproduction=previous['reproduction']+' Accepted CUDA evaluation and expanded scalar diagnostics '
                       'are now bound in the portfolio index under scoped_GPU_component_evaluation and retained_scalar_statistics. '
                       'Use the exact bound protocols, selected tensors and original reference population; no clean retraining '
                       'or new-Rheon equivalence is established. Raw prediction archives are not required for plotting. '
                       'Portable embedding of these external scalar/source bindings remains pending.')
        proposed = original
        for key in LABELS:
            proposed = replace_paragraph(proposed, key, previous[key], updated[key])
        intro = 'Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.'
        require(proposed.count(intro) == 1, 'Source-review introduction differs')
        proposed = proposed.replace(intro, 'Authored interpretation of captured source and receipts, updated with verified scoped CUDA evaluation. No new training, lineage or scientific promotion is implied.', 1)
        a, stats = row['accuracy_benchmark'], row['retained_field_statistics']
        details = ('## Scoped CUDA evaluation\n\nNVIDIA GeForce RTX 5090 Laptop GPU. '
                   'The outcome below describes the measured interface and population; required supplied context '
                   'and excluded work remain material parts of this result.\n\n'
                   f'Accuracy authority: `{a["independent_reduction"]["path"]}` · SHA256 `{a["independent_reduction"]["sha256"]}`.\n\n'
                   f'Expanded statistics authority: `{stats["source"]["path"]}` · SHA256 `{stats["source"]["sha256"]}`. '
                   'These are external lab bindings; native portable scalar/source closure remains pending.\n\n')
        if 'input_scope' in stats:
            details += f'Inputs: {stats["input_scope"]}\n\nOutputs: {stats["output_scope"]}\n\n'
        if row['task'].startswith('guided_'):
            details += lead_table(fid, lead_source['rows'])
        first = proposed.index('**Method.** ')
        proposed = proposed[:first]+details+proposed[first:]
        require(original.count(TAIL) == proposed.count(TAIL) == 1, 'Original retained-fact marker differs')
        tail = original.split(TAIL, 1)[1].encode(); receipt = originals[fid]['original_receipt']
        receipt_raw = (ROOT/receipt['path']).read_bytes()
        require(binding(ROOT/receipt['path']) == {k:receipt[k] for k in ('path','sha256')}
                and proposed.encode().endswith(tail) and original.encode().count(receipt_raw) == proposed.encode().count(receipt_raw) == 1,
                'Original receipt/configuration tail changed')
        put(path, proposed.encode(), fit_item_id=fid, original_tail_sha256=digest(tail), original_receipt=receipt)
        node['summary'] = updated; node['card'].update(revision=notes[-1]['revision'], sha256=notes[-1]['sha256'])
        fit.update(card_sha256=notes[-1]['sha256'], scoped_GPU_component_evaluation=dict(
            accuracy=row['accuracy_benchmark'], qualified_timing=row.get('default_benchmark'),
            authority=source_specs[str(COVERAGE.relative_to(ROOT))], native_chart_publication_pending=True),
            retained_scalar_statistics=stats)
    index['sources']['selected_fit_expanded_statistics'] = binding(DOC/'reports/plot-statistics-04.json')
    index['sources']['all205_selected_supported_accuracy'] = binding(COVERAGE)
    guide['description'] = ('205 source-reviewed recorded fits/components now have scoped CUDA accuracy outcomes. '
        '150 pressure fits share 128-request comparisons; 35 complete generators share 64-case comparisons. '
        'The remaining 20 components use their own admitted interfaces/populations. 195 fits have qualified GPU timing; '
        'ten timings remain. Expanded scalar diagnostics cover all205 selected fits; native charts and portable sources '
        'for the newer components remain pending. All524 states/variants, clean reproduction and long drift remain incomplete. '
        'Authored comparisons do not change original Trial ancestry.')
    put(INDEX, encode(index)); put(GUIDE, encode(guide))
    overview_path = 'notes/portfolio-overview.md'; overview = (PROJECT/overview_path).read_text(); title = '# Rheon model portfolio\n\n'
    require(overview.startswith(title), 'Original overview title differs')
    addition = ('## Current selected-fit evaluation: 205 accuracy outcomes, 195 qualified timings\n\n'
        'All twenty remaining component cards now explain their scoped CUDA accuracy outcomes and lessons. '
        'They include native-RHS pressure, future-RHS q, local sampling/viscosity, supplied-stage correction/readout '
        'and complete guided compositions. Their inputs, outputs and populations differ; do not rank all of their '
        'errors or request times as complete simulation-step results. Ten timings remain unavailable.\n\n'
        'Expanded scalar diagnostics cover all205 selected fits and49,700task-specific records. The47compressed '
        'scalar bodies total133,453,465bytes; they contain no raw prediction arrays. Earlier sections below record '
        'superseded publication stages. Current scoped figures do not complete all524states, intended policies, '
        'independent holdout, clean reproduction or600-output own-history drift.\n\n'
        'Every physical output retains its own accuracy. The two guided cards include per-lead best/worst/mean/'
        'median/p95/count at30/60/120outputs per second andK1–4. Internal groups and sampler middle/end targets '
        'are not physical output leads. Full block and block/K timing stay separate from independently measured '
        'stage costs. An accurate endpoint cannot qualify inaccurate intervening outputs.\n\n'
        'The new results and scalar bodies remain bound to exact external lab sources. Native chart publication '
        'for these twenty components and portable scalar/source closure remain pending; no portable-completion '
        'claim is made by this note update. Prediction deletion remains paused for the interface/work/consumer audit.\n\n')
    put(overview_path, (title+addition+overview[len(title):]).encode())
    require(len(notes) == 23 and sum(n['bytes'] for n in notes) <= 8*1048576, 'Native note batch bound exceeded')
    _, latest = header(); latest_entries = {(e['family'],e['id']):e for e in latest['inventory']}
    require(latest['checkpoint'] == snapshot['checkpoint'], 'Project changed during preparation')
    for note in notes:
        entry, _ = current[note['path']]
        require(latest_entries[('project_file',entry['id'])] == entry, 'Target note changed')
    OUTPUT.mkdir()
    for path, raw in staged.items():
        target = OUTPUT/path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    save(OUTPUT/'preparation.json', dict(status='twenty_scoped_component_outcomes_prepared_not_adopted',
        created_unix=time.time(), source_project_id=snapshot['project_id'], source_checkpoint_at_read=snapshot['checkpoint'],
        notes=notes, component_fits=20, models=205, authored_comparisons=26, guide_revision=current[GUIDE][1]['current']['revision']+1,
        original_receipt_configuration_tails_preserved=20, producer=binding(Path(__file__)), evidence=list(source_specs.values()),
        native_chart_publication_pending=True, portable_scalar_source_closure_complete=False,
        native_mutations=0, new_model_forwards=0, prediction_files_deleted=0, Trial_lineage_modified=False, full_portfolio_complete=False))
    print('Prepared23notes:20scoped component outcomes,guide,index,overview;no native changes.', flush=True)


if __name__ == '__main__':
    run()
