"""Prepare26revisions:23pressure outcomes plus guide,index and overview."""
import copy
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,PROJECT,header,read,save,binding,digest,encode,require
from prepare_matched_generator_outcome_notes import replace_paragraph,TAIL,LABELS

OUTPUT = RUN/'native-pressure-outcome-notes-01'
GUIDE = 'notes/00-models.portfolio.json'
INDEX = 'notes/portfolio-index.json'


def run():
    require(not OUTPUT.exists(),'Fresh pressure outcome proposal required')
    timing_path = DOC/'reports/original-pressure-GPU-timing-batch-01.json'
    accuracy_path = DOC/'reports/original-train-pressure-CUDA-accuracy-01.json'
    qualification_path = DOC/'reports/all150-pressure-GPU-measurements-qualification-01.json'
    timing,accuracy,proof = [read(path) for path in (timing_path,accuracy_path,qualification_path)]
    require(timing['fits'] == accuracy['fits'] == 23 and timing['fresh_processes'] == 46
            and proof['exact_scientific_observations'] == 96000,'Accepted23timings and150native scalar qualification required')
    chart_path = RUN/'all150-pressure-GPU-measurements-01/preparation.json'
    charts = read(chart_path)
    material_path = RUN/'remaining-pressure-material-01.json'
    material = {row['fit_item_id']:row for row in read(material_path)['models']}
    scored = {row['fit_item_id']:row for row in accuracy['rows']}
    packet_by_fit = {}
    for part in charts['packets']:
        packet = read(chart_path.parent/part['path'])
        for subject in packet['subjects']:
            packet_by_fit[subject['dimensions']['fit_item_id']] = dict(packet_sha256=part['sha256'],subject_id=subject['id'])
    _,snapshot = header()
    current = {}
    for entry in snapshot['inventory']:
        if entry['family'] != 'project_file':
            continue
        path = PROJECT/'.tuldok'/entry['path']
        raw = path.read_bytes()
        require(digest(raw) == entry['sha256'] and len(raw) == entry['bytes'],'Current canonical note changed')
        doc = read(path)
        if doc['project_id'] == snapshot['project_id']:
            require(doc['path'] not in current,'Repeated current-owner note path')
            current[doc['path']] = entry,doc
    guide,index = read(PROJECT/GUIDE),read(PROJECT/INDEX)
    nodes = {node['id']:node for node in guide['models']}
    fits = {fit['item_id']:fit for fit in index['fits']}
    require(len(nodes) == len(fits) == 205 and len(guide['relationships']) == 26,'Exact205model/26relationship organization required')
    staged,notes = {},[]

    def put(path,raw,**extra):
        entry,doc = current[path]
        old = doc['current']
        require((PROJECT/path).read_bytes() == old['text'].encode() and digest(old['text'].encode()) == old['sha256']
                and 0 < len(raw) <= 1048576,'Current/proposed note bytes differ or exceed bound')
        staged[path] = raw
        notes.append(dict(kind='note',path=path,expected_revision=old['revision'],revision=old['revision']+1,
            previous_sha256=old['sha256'],sha256=digest(raw),bytes=len(raw),file_id=entry['id'],
            prepared_path=str((OUTPUT/path).relative_to(ROOT)),**extra))

    for model in timing['rows']:
        fid = model['fit_item_id']
        fit,node,original_entry = fits[fid],nodes[fid],material[fid]
        path = fit['card_path']
        original = (PROJECT/path).read_text()
        old_summary = copy.deepcopy(node['summary'])
        worst = model['worst']
        outcome = (f'On the common 128 consumed old-solver pressure requests, worst mean-free pressure relative L2 is '
            f'{worst["pressure_mean_free_relative_l2"]:.6g} and worst equation residual relative L2 is '
            f'{worst["common_equation_relative_l2"]:.6g}. Median case GPU latency is {model["case_median_median_ms"]:.5f} ms per pressure request; '
            f'the largest case median is {model["worst_case_median_ms"]:.5f} ms/request. Two fresh process phases preserve every input, reference and output hash, '
            'the original final learned tensors and the captured default pressure policy. These are pressure proposals, not complete simulation steps or independent holdout results.')
        controls = [row['identifier'] for row in scored[fid]['control_comparisons'] if row['all128_request_reference_and_output_hashes_exact']]
        extra_lesson = (' The common population exactly reproduces all 128 pressure fields of '+', '.join(controls)+
            '; retain separate fit histories despite shared final parameter bytes. This establishes only this population, not general execution-policy equivalence.' if controls else
            ' The common benchmark now quantifies this retained default policy; its training objective and smoke/publication duration are not pressure accuracy or deployment latency. '
            'Use the recorded same-request errors and complete pressure-call cost when deciding whether to retain or extend this design.')
        new_summary = dict(old_summary,outcome=outcome,lesson=old_summary['lesson']+extra_lesson,
            reproduction=old_summary['reproduction']+' The exact original-training-output CUDA evaluation is now qualified on all 128 cases with original_train_pressure_accuracy_02.py '
            'and two fresh timing phases with original_train_pressure_timing.py. Independent NumPy accuracy and stdlib timing reductions retain exact original sources, tensors and outputs. '
            'This does not qualify clean training, historical CPU restore/publication paths, new-Rheon regeneration or long drift.')
        proposed = original
        for key in LABELS:
            proposed = replace_paragraph(proposed,key,old_summary[key],new_summary[key])
        intro = 'Authored interpretation of captured source and receipts. No new training, benchmark, lineage or scientific promotion is implied.'
        require(proposed.count(intro) == 1,'Original source-review introduction differs')
        proposed = proposed.replace(intro,'Authored interpretation of captured source and receipts, updated with verified GPU pressure benchmarks. No new training, lineage or scientific promotion is implied.',1)
        details = ['## Common GPU pressure benchmark','',
            'NVIDIA GeForce RTX 5090 Laptop GPU; exact retained F32 learned parameters and F64 pressure arithmetic. '
            'The boundary starts with owned native CPU arrays/scalars and includes validation, packing, H2D, captured CUDA pressure prediction, '
            'independent owned CPU output, D2H and synchronization. No pressure-pass or K division is applied.','',
            'Twenty primary wall samples per case come from two fresh processes. Construction, first calls and five warmups per case are separate. '
            'The device observer is removed for primary calls. Four paired on/off calibrations per case per phase measure its overhead separately; '
            f'this fit has median paired overhead {model["paired_observer_overhead_median_ms"]:.6g} ms/request. No overhead is subtracted or negative differences clipped.','',
            '| Cells/m | Grid cells | Input dt (s) | Requests | Worst pressure relative L2 | GPU ms/request |',
            '| --- | --- | --- | --- | --- | --- |']
        for group in model['density_grid_timestep_strata']:
            grid = 'x'.join(map(str,group['counts_xyz']))
            details.append(f'| {group["cells_per_metre"]:g} | {grid} | {group["input_dt_s"]:.8g} | {group["cases"]} | '
                f'{group["worst"]["pressure_mean_free_relative_l2"]:.6g} | {group["median_case_ms"]:.5f} |')
        plot = packet_by_fit[fid]
        details += ['', 'Times above are medians of case medians; a native maximum plot instead selects the largest case statistic. '
            'Grid dimensions and cells per metre are independent filters. These 128 consumed requests do not establish generalization or own-history drift. '
            'Clear compute-PID snapshots do not prove continuous hardware exclusivity.','',
            f'Unified pressure table: {plot["packet_sha256"]}; subject: {plot["subject_id"]}. '
            'The 150-fit view preserves original benchmark bindings and historical 127-fit tables. Raw process bodies, pressure-field archives and model weights are not copied into this chart evidence.','']
        first = proposed.index('**Method.** ')
        proposed = proposed[:first]+'\n'.join(details)+'\n'+proposed[first:]
        archive = ('## Previous source-review statements\n\nThese paragraphs describe the earlier source-review stage; the common GPU measurements above update evaluation status.\n\n```json\n'
            +encode({key:old_summary[key] for key in LABELS}).decode()+'```\n\n')
        require(original.count(TAIL) == proposed.count(TAIL) == 1,'Original retained-fact marker differs')
        proposed = proposed.replace(TAIL,archive+TAIL,1)
        tail = original.split(TAIL,1)[1].encode()
        receipt_spec = original_entry['original_receipt']
        receipt = (ROOT/receipt_spec['path']).read_bytes()
        require(binding(ROOT/receipt_spec['path']) == receipt_spec and proposed.encode().endswith(tail)
                and original.encode().count(receipt) == proposed.encode().count(receipt) == 1,'Original receipt/configuration tail changed')
        put(path,proposed.encode(),fit_item_id=fid,original_tail_sha256=digest(tail),original_receipt=receipt_spec)
        note = notes[-1]
        node['summary'] = new_summary
        node['card'].update(revision=note['revision'],sha256=note['sha256'])
        alias = 'GPU pressure: '+plot['subject_id']
        require(alias not in node['aliases'],'Pressure benchmark alias already exists')
        node['aliases'].append(alias)
        fit.update(card_sha256=note['sha256'],common_GPU_pressure_benchmark=dict(**plot,cases=128,inference_device='cuda',timing_units='ms/pressure_request'))
    index['sources']['additional23_common_GPU_pressure_benchmark'] = binding(timing_path)
    index['sources']['all150_pressure_GPU_comparison'] = binding(chart_path)
    guide['description'] = ('205 recorded fits/components, all source reviewed. 150 pressure fits have comparable 128-request GPU benchmarks; '
        '35 complete generator fits have matched 64-case GPU benchmarks. These are different tasks and populations. '
        'Twenty other components, full 524-state/intended-variant coverage, clean reproduction and long drift remain pending. '
        'Authored comparisons do not change original Trial ancestry.')
    put(INDEX,encode(index))
    put(GUIDE,encode(guide))
    overview_path = 'notes/portfolio-overview.md'
    overview = (PROJECT/overview_path).read_text()
    title = '# Rheon model portfolio\n\n'
    require(overview.startswith(title),'Current overview title differs')
    addition = ('## Pressure benchmarks now cover 150 fits\n\n'
        'The unified “150 pressure fits · complete GPU requests · 128 consumed cases” view joins the 127 named fits with 23 original-training-output fits. '
        'All 185 measured default fit identities now have GPU accuracy and timing: 150 pressure proposals and 35 full-state generators. '
        'Twenty other components still need appropriately scoped benchmarks. This does not complete all 524 parameter states or intended solver variants.\n\n'
        'For the 23 newly measured pressure fits, median case latency ranges from 2.08794 to 8.65121 GPU ms per pressure request and worst mean-free pressure error '
        'ranges from 0.00602154 to 1.58280. These are retained experiments, not promoted application-quality models. A pressure request does not advance a full simulation step; '
        'compare it separately from the complete F64 generator results below.\n\n'
        'Select both unified pressure partitions to compare all 150 subjects; filter cells per metre separately from grid X/Y/Z and input dt. '
        'Primary timing excludes forward-validation hooks; separate paired overhead for the additional 23 fits is retained without subtraction. '
        'The native maximum chart shows the maximum over selected case statistics; cards also report the median across case medians. '
        'Original process captures remain at their exact bound paths, and historical measurement tables stay unchanged.\n\n')
    put(overview_path,(title+addition+overview[len(title):]).encode())
    require(len(notes) == 26 and sum(note['bytes'] for note in notes) <= 8*1048576,'Native26note batch bound exceeded')
    _,latest = header()
    latest_entries = {(entry['family'],entry['id']):entry for entry in latest['inventory']}
    for note in notes:
        entry,_ = current[note['path']]
        require(latest_entries[('project_file',entry['id'])] == entry,'Target note changed during preparation')
    OUTPUT.mkdir()
    for path,raw in staged.items():
        target = OUTPUT/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    save(OUTPUT/'preparation.json',dict(status='pressure26outcome_notes_prepared_not_adopted',created_unix=time.time(),
        source_project_id=snapshot['project_id'],source_checkpoint_at_read=snapshot['checkpoint'],notes=notes,pressure_fits=23,models=205,
        authored_comparisons=26,guide_revision=current[GUIDE][1]['current']['revision']+1,literal_summary_paragraphs=1230,
        original_receipt_configuration_tails_preserved=23,previous_review_statements_retained=True,
        producer=binding(Path(__file__)),evidence=[binding(path) for path in (timing_path,accuracy_path,qualification_path,chart_path,material_path)],
        native_mutations=0,new_model_forwards=0,Trial_lineage_modified=False,full_portfolio_complete=False))
    print('26pressure outcome notes prepared;23exact original receipts/tails and205model identities retained.',flush=True)


if __name__ == '__main__':
    run()
