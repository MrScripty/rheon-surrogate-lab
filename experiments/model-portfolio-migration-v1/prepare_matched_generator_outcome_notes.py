"""Prepare38guarded note revisions for matched generator outcomes.

Only proposed notes are written. Original training receipt/configuration tails
and previous review statements remain exact. Guide identities, groups and
authored relationships stay unchanged; no Trial ancestry or new fit is created.
"""
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,header,read,digest,encode,binding,require

OUTPUT = RUN/'native-matched-generator-outcome-notes-01'
GUIDE = 'notes/00-models.portfolio.json'
INDEX = 'notes/portfolio-index.json'
TAIL = '## Original retained fit facts before this review\n\n'
LABELS = dict(outcome='Observed outcome',lesson='Lesson',reproduction='Reproduction')
NEXT = dict(outcome='\n\n**Lesson.** ',lesson='\n\n**Reproduction.** ',
    reproduction='\n\n## Exact source bindings for this review\n')


def replace_paragraph(text,key,old,new):
    prefix = '**'+LABELS[key]+'.** '
    start = text.index(prefix)+len(prefix)
    end = text.index(NEXT[key],start)
    require(text[start:end].strip() == old,'Current card/guide paragraph differs')
    return text[:start]+new+text[end:]


def run():
    require(not OUTPUT.exists(),'Preserve original matched outcome proposal')
    full_path = DOC/'reports/complete-generator-GPU-benchmark-01.json'
    f64_path = DOC/'reports/same-case-F64-GPU-benchmark-02.json'
    comparison_path = DOC/'reports/matched-generator-F64-comparison-01.json'
    proof_path = DOC/'reports/readable-matched-generator-GPU-measurements-qualification-01.json'
    full,f64,comparison,proof = [read(p) for p in (full_path,f64_path,comparison_path,proof_path)]
    require(full['models_count'] == 35 and len(f64['models']) == 4
            and proof['all24960_scientific_observations_exact'] and proof['native_filter_queries'] == 600,
            'Complete matched benchmark and readable table qualification required')
    raw_header,snapshot = header()
    notes = {}
    for entry in snapshot['inventory']:
        if entry['family'] != 'project_file':
            continue
        path = PROJECT/'.tuldok'/entry['path']
        require(binding(path)['sha256'] == entry['sha256'],'Canonical note body changed')
        document = read(path)
        if document['project_id'] == snapshot['project_id']:
            require(document['path'] not in notes,'Repeated native note path')
            notes[document['path']] = (entry,document)
    guide,index = read(PROJECT/GUIDE),read(PROJECT/INDEX)
    models = {m['id']:m for m in guide['models']}
    fits = {f['item_id']:f for f in index['fits']}
    selected = {m['fit_item_id']:m for m in full['models']}
    require(len(models) == len(fits) == 205 and set(selected) == {fid for fid,f in fits.items() if f['task'] == 'full_state_generator'},
            'Exact205model/35generator population required')
    captured = {m['fit_item_id']:m for m in read(RUN/'complete-generator-GPU-protocol-01.json')['models']}
    baselines = {m['K']:m for m in f64['models']}
    ratios = {m['fit_item_id']:m['median_time_ratio_fit_over_F64'] for m in comparison['rows']}
    prep = read(RUN/'matched-generator-GPU-measurements-02/preparation.json')
    packet_by_K = {p['K']:p for p in prep['packets']}
    aliases = {}
    for part in prep['packets']:
        packet = read(RUN/'matched-generator-GPU-measurements-02'/part['path'])
        for subject in packet['subjects']:
            if subject['dimensions']['origin_kind'] == 'original_fit':
                aliases[subject['dimensions']['fit_or_execution_id']] = subject['id']
    staged,changed = {},[]

    def put(path,raw,**extra):
        entry,doc = notes[path]
        old = doc['current']
        require((PROJECT/path).read_bytes() == old['text'].encode() and digest(old['text'].encode()) == old['sha256']
                and 0 < len(raw) <= 1048576,'Original/proposed note identity or byte bound differs')
        staged[path] = raw
        changed.append(dict(kind='note',path=path,expected_revision=old['revision'],revision=old['revision']+1,
            previous_sha256=old['sha256'],sha256=digest(raw),bytes=len(raw),file_id=entry['id'],
            prepared_path=str((OUTPUT/path).relative_to(ROOT)),**extra))

    for fid,model in selected.items():
        fit,node = fits[fid],models[fid]
        path = fit['card_path']
        original = (PROJECT/path).read_text()
        old_summary = copy.deepcopy(node['summary'])
        k,s = model['K'],model['summary']
        worst = s['worst_metrics']
        outcome = (f'Matched GPU benchmark on 64 retained cases: worst pressure, combined velocity and tracer relative L2 are '
            f'{worst["pressure_relative_l2"]:.6g}, {worst["velocity_relative_l2"]:.6g} and {worst["tracer_relative_l2"]:.6g}. '
            f'Median time is {s["median_case_median_block_ms"]:.5f} ms per K{k} block '
            f'({s["median_case_median_per_output_ms"]:.5f} ms per physical output); the worst case median is '
            f'{s["worst_case_median_per_output_ms"]:.5f} ms/output. The retained F64 N1 composition has better worst pressure error '
            f'and {ratios[fid]:.2f} times lower matched median GPU time. These are consumed fitting scenes, not an independent holdout or long rollout.')
        architecture = captured[fid].get('architecture','direct')
        if 'fixed-viscosity' in model['identifier']:
            lesson = ('The controlled K1 diagnostic locates the fixed-viscosity series\' shared 32-cubed pressure limit mainly in its CNN fallback. '
                'Other pressure-map and request errors remain. Freezing accurate local coefficients does not ensure a comparably accurate full generator; '
                'compare the complete pressure, velocities and tracer with the F64 composition before promoting a design.')
        elif architecture in ('endpoint','reuse'):
            lesson = ('Sharing an encoder, endpoint or retained features does not by itself establish a useful accuracy/speed tradeoff. '
                'This fit remains worse than the matched F64 composition in both worst pressure error and median GPU time. '
                'Account for every intermediate output and reconstruction cost; K outputs advance K full 1/60-second intervals.')
        else:
            lesson = ('This retained design has higher worst pressure error and median GPU time than the matched F64 composition. '
                'Its method and fitting purpose remain recorded above; these measurements compare distinct designs and do not show that one checkpoint deteriorated. '
                'Judge complete physical outputs rather than a pressure-only component or a training-loss value.')
        reproduction = ('Exact training receipts, source/configuration bindings, parameter inventories and fitting-data identities remain below. '
            'The CUDA evaluation now passes both process phases using complete_generator_gpu_benchmark.py and complete-generator-GPU-protocol-01.json. '
            + ('Inherited derived pressure maps are explicitly GPU-built precision variants. ' if 'readiness' in captured[fid] else '')
            + 'Clean training reproduction and GPU-only publication probes remain pending; current evaluation does not establish those gates.')
        new_summary = dict(old_summary,outcome=outcome,lesson=lesson,reproduction=reproduction)
        proposed = original
        for key in LABELS:
            proposed = replace_paragraph(proposed,key,old_summary[key],new_summary[key])
        details = ['## Matched GPU benchmark','',
            'Complete native CPU input to every independently owned CPU output, with prediction on an NVIDIA GeForce RTX 5090 Laptop GPU. '
            'Two fresh process phases supply 20 primary wall samples per case; forward-validation hooks, setup and warmups are excluded from primary timing.',
            '',f'K = {k}; each output advances 1/60 second. Full block duration is {k}/60 seconds. '
            'The 64 cases use 16 consumed fitting scenes at starts 0, 24, 64 and 120. Grid density and cell counts remain separate.','',
            '| Cells/m | Grid cells | Worst pressure relative L2 | GPU ms/output | F64 GPU ms/output |',
            '| --- | --- | --- | --- | --- |']
        for key,group in sorted(model['by_density_and_grid'].items(),key=lambda p:(int(p[0].split('cpm:')[0]),p[0])):
            density,grid = key.split('cpm:')
            base = baselines[k]['by_density_and_grid'][key]
            details.append(f'| {density} | {grid} | {group["worst_metrics"]["pressure_relative_l2"]:.6g} | '
                f'{group["median_case_median_per_output_ms"]:.5f} | {base["median_case_median_per_output_ms"]:.5f} |')
        details += ['', 'Times are medians of per-case medians. The native maximum plot instead shows the largest per-case statistic among the selected cases. '
            'The F64 baseline still executes one sequential N1 advance per output; its K-pass speed includes amortized transfer/graph overhead.', '',
            'Retained measurement source: '+packet_by_K[k]['sha256']+'; subject: '+aliases[fid]+'.', '']
        first = proposed.index('**Method.** ')
        proposed = proposed[:first]+'\n'.join(details)+'\n'+proposed[first:]
        archive = ('## Previous source-review statements\n\nThese statements describe the earlier source-review stage. '
            'The matched results above update evaluation status while retaining the original interpretation.\n\n```json\n'
            + encode({key:old_summary[key] for key in LABELS}).decode()+'```\n\n')
        require(original.count(TAIL) == proposed.count(TAIL) == 1,'Original retained-tail marker differs')
        proposed = proposed.replace(TAIL,archive+TAIL,1)
        tail = original.split(TAIL,1)[1].encode()
        require(proposed.encode().endswith(tail),'Original receipt/configuration tail changed')
        receipt = (ROOT/captured[fid]['receipt']['path']).read_bytes()
        require(original.encode().count(receipt) == proposed.encode().count(receipt) == 1,'Original literal receipt changed')
        put(path,proposed.encode(),fit_item_id=fid,original_tail_sha256=digest(tail),original_receipt=binding(ROOT/captured[fid]['receipt']['path']))
        note = changed[-1]
        node['summary'] = new_summary
        node['card'].update(revision=note['revision'],sha256=note['sha256'])
        alias = 'GPU benchmark: '+aliases[fid]
        if alias not in node['aliases']:
            node['aliases'].append(alias)
        fit.update(card_sha256=note['sha256'],matched_GPU_benchmark=dict(packet_sha256=packet_by_K[k]['sha256'],subject_id=aliases[fid],cases=64,K=k))
    index['sources']['matched_generator_GPU_benchmark'] = binding(full_path)
    guide['description'] = ('205 recorded fits/components, all source-reviewed. The 35 full-state generator summaries now include matched 64-case GPU outcomes. '
        'The retained F64 composition is evaluated separately at K1–4; these are execution variants, not new fits. '
        'The 524 parameter byte states and 379 dashboard aliases remain separate counts. Original Trial ancestry is unchanged.')
    put(INDEX,encode(index))
    put(GUIDE,encode(guide))
    overview = ['# Rheon model portfolio','',
        'This project retains 205 recorded fits/components and 524 parameter byte states. All 205 have source reviews; '
        '35 full-state generator fits now have matched GPU benchmarks. The four F64 baseline executions are not four new fits. '
        'The 127 named pressure fits have a separate 128-request GPU comparison.','',
        '## Matched full-state results','',
        'On the same 64 retained native cases, the original F64 cell composition has better worst pressure error and '
        '7.645–31.085 times lower matched median GPU time than every one of the 35 other generator fits. '
        'This compares distinct retained designs; the best checkpoint has not disappeared or deteriorated.','',
        '| Physical outputs per pass | Worst pressure relative L2 | GPU ms/output, median case | GPU ms/output, 32 cubed |',
        '| --- | --- | --- | --- |']
    for k,base in sorted(baselines.items()):
        overview.append(f'| {k} | {base["summary"]["worst_metrics"]["pressure_relative_l2"]:.6g} | '
            f'{base["summary"]["median_case_median_per_output_ms"]:.5f} | {base["by_grid_dimensions"]["32x32x32"]["median_case_median_per_output_ms"]:.5f} |')
    overview += ['', 'Every output includes pressure, all three staggered velocities and tracer, and advances a full 1/60 second. '
        'The F64 implementation executes sequential N1 advances; packing K outputs amortizes transfers and graph-call overhead, '
        'rather than learning a shortcut across intermediate states. The N8-only coordinate change is unused in this regime.', '',
        'The best pressure tails among the 35 other fits are about 5.24e-4. A controlled K1 diagnostic locates the fixed-viscosity '
        'series\' shared 32-cubed worst pressure case mainly in its CNN pressure fallback. Other pressure-map and request errors remain. '
        'The diagnostic does not replace a causal generator with future reference input.', '',
        '## Reading the measurements', '',
        'Choose the retained source titled “Matched GPU generators and F64 baseline”, then select K, grid dimensions and cells per metre. '
        'Plot IDs use model names; b1-v1 and b1-v2 are distinct fits. Original fit identities and exact process captures remain inspectable. '
        'Times cover native CPU input, packing, known controls, GPU prediction and all owned CPU outputs with synchronization.', '',
        'The table above reports medians of per-case medians. A native maximum plot reports the maximum of the selected per-case statistic. '
        'Thus “Median GPU time per physical output” on that plot means the largest case median, not a median across cases. '
        'Each fit card includes density/domain tables and the exact retained source identity.', '',
        'The population is 16 consumed fitting scenes at four starts, densities 8 and 16 cells/m, and grids 15×6×8, 9×11×11, 16 cubed and 32 cubed. '
        'It is not independent holdout evidence or a 600-step own-history rollout. Resource observations found no competing CUDA compute PID; '
        'external host work was present and continuous hardware exclusivity is not established.', '',
        '## Historical timing correction', '',
        'The four earlier “Complete GPU generators” tables retain ten fits measured with a forward-validation hook inside the timing interval. '
        'Their scalar values and original packets stay unchanged. The new matched tables exclude that hook and preserve all prior output hashes. '
        'Paired calibration measures median hook overhead of 0.0593–0.4997 ms per block across fits. '
        'That instrumentation mismatch does not explain the numerical error gap.', '',
        'Historical instrumented packet identities:', '```text']
    overview += [p['sha256'] for p in read(RUN/'common-generator-GPU-measurements-01/preparation.json')['packets']]
    overview += ['```','', 'The quoted near-exact cell and temporal anchors also reproduce on their original populations. '
        'Those use different rectangles, densities or cadences. The original temporal entry requires fine-clock force schedules absent '
        'from these legacy scenes; its cadence results stay separate. No fine histories were invented for this comparison.', '',
        '## Model history and remaining work', '',
        'Pressure-only error/time and complete simulation-output error/time are separate tasks. Guide summaries explain each method, '
        'fitting exposure, purpose and outcome. Authored comparisons organize evidence; all 94 genuine Trials retain their original unknown ancestry.', '',
        'Clean GPU training/publication reproduction, remaining component/checkpoint benchmarks and long drift still need qualification. '
        'Thirty-five parameter states remain outside native declarations, and 936 registrations still need disposition. '
        'No unique weights are approved for deletion by these results alone.', '',
        'Old-Rheon references remain the benchmark authority. The updated Rheon changes fluid behavior and needs a separate physics epoch '
        'with training/test regeneration scripts. Final portable save/load acceptance and practical fast Open remain unfinished.','']
    put('notes/portfolio-overview.md','\n'.join(overview).encode())
    require(len(changed) == 38 and sum(n['bytes'] for n in changed) <= 8*1048576,'Native38note batch bound exceeded')
    # Measurement imports may add records concurrently, but cannot revise notes.
    _,latest = header()
    latest_notes = {(e['family'],e['id']):e for e in latest['inventory'] if e['family'] == 'project_file'}
    for path,(entry,doc) in notes.items():
        require(latest_notes.get(('project_file',entry['id'])) == entry,'A native note changed during preparation')
    OUTPUT.mkdir()
    for path,raw in staged.items():
        target = OUTPUT/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    packet = dict(status='matched_outcome_notes_prepared_not_adopted',created_unix=time.time(),source_project_id=snapshot['project_id'],
        source_checkpoint_at_read=snapshot['checkpoint'],notes=changed,matched_fits=35,models=205,authored_comparisons=26,
        guide_revision=notes[GUIDE][1]['current']['revision']+1,literal_summary_paragraphs=1230,
        original_receipt_configuration_tails_preserved=35,previous_review_statements_retained=True,
        producer=binding(Path(__file__)),evidence=[binding(p) for p in (full_path,f64_path,comparison_path,proof_path)],
        new_neural_forwards=0,native_mutations=0,Trial_lineage_modified=False,full_portfolio_complete=False)
    (OUTPUT/'preparation.json').write_bytes(encode(packet))
    print('Prepared38note revisions:35matched outcomes,guide,index andoverview;original tails and26relationships preserved.')


if __name__ == '__main__':
    run()
