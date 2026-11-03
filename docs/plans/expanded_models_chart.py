"""Recorded CPU/GPU successor measurements, with explicit population and clock scope.

This importer does not run models, benchmarks, native references or training.
Legacy CPU pressure-cohort data remains in its original comparison catalog.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import math
import os
import statistics

STUDY='full-ai-data-diversity-v1'
FIELDS=('equation','pressure','velocity','tracer')


def enrich_expanded_models(study,base,lab,load):
    base,lab=Path(base),Path(lab)
    docs,root=base/STUDY,lab/'runs'/STUDY
    if not (docs/'gpu-fusion-results-04.json').exists():return
    hashes={}
    def sha(path):
        path=Path(path)
        if path not in hashes:
            with path.open('rb') as stream:hashes[path]=hashlib.file_digest(stream,'sha256').hexdigest()
        return hashes[path]
    def read(path,expected=None):return load(Path(path),expected)
    def witness(value):
        for key in ('source_sha256','source_evidence','evidence_sha256','reference_sources'):
            sources=value.get(key,{})
            if not isinstance(sources,dict):continue
            for source,expected in sources.items():
                if not isinstance(expected,str) or len(expected)!=64:continue
                if sha(lab/source)!=expected:raise ValueError('Expanded measurement evidence changed: '+source)
    def result(path,status=None):
        value=read(path)
        if status and value['status']!=status:raise ValueError('Incomplete expanded measurement: '+str(path))
        if value.get('optimizer_updates',0) or value.get('expanded_reserved52_references_opened',0):
            raise ValueError('Evaluation changed training or reserved population')
        witness(value)
        return value
    def source(path):return os.path.relpath(path,base)
    def group(row):return (row.get('K',row.get('horizons')),tuple(row['counts_xyz']),row['cells_per_metre'])

    data=dict(variants=[],accuracy=[],timing=[],traces=[],component=None,training=None,sources=[],
        caveat='All GPU populations are consumed development/regression data. The 52 expanded reserved scenes remain unopened. K counts published simulation steps; internal advances and transfer scopes remain separate. No automatic latency ranking or joint target pass.')
    component=read(docs/'shared-spectral-component-results-01.json');witness(component)
    if component['status']!='managed_expanded_shared_pressure_component_passed_all24development_scenes':raise ValueError('Incomplete shared pressure component')
    pair=read(root/'shared-spectral-pair-01/receipt.json')
    restore=read(root/'shared-spectral-pair-01/restore/receipt.json',pair['restore_receipt_sha256'])
    train=read(root/'shared-spectral-pair-01/train/receipt.json',pair['training_receipt_sha256'])
    if (pair['status']!='completed' or pair['optimizer_updates']!=160 or pair['parameters']!=4 or pair['model_promotion']
            or train['optimizer_updates']!=160 or restore['optimizer_updates'] or restore['execution']['cuda_initialized']
            or not restore['typed_cpu_weight_restore_verified'] or restore['cpu_model_forward_executed']):
        raise ValueError('Shared pressure training/typed restore custody differs')
    for filename,parity in pair['artifact_parity'].items():
        if not parity or sha(root/'shared-spectral-pair-01/train'/filename)!=sha(root/'shared-spectral-pair-01/restore'/filename):
            raise ValueError('Shared pressure artifact parity differs')
    data['component']=dict(source=source(docs/'shared-spectral-component-results-01.json'),
        parameters=4,optimizer_calls=160,training_scenes=148,groups=component['development']['summaries_by_density_and_grid'],
        cpu_probe_scope=restore['probe_scope'])
    block_lookup={}
    block_hashes={}
    def variant(identifier,label,device='gpu',kind='execution'):
        data['variants'].append(dict(id=identifier,label=label,device=device,kind=kind))
    def aggregate(rows,identifier,population,path,rollout=False,internal=None):
        grouped=defaultdict(list)
        for row in rows:grouped[group(row)].append(row)
        for (k,counts,density),selected in grouped.items():
            errors={f:max(r['metrics']['relative'][f] for r in selected) for f in FIELDS}
            if not all(math.isfinite(e) and e>=0 for e in errors.values()):raise ValueError('Nonfinite expanded accuracy')
            passed=all(r.get('all_accuracy_pass',r['metrics'].get('all_accuracy_criteria_pass',r['metrics'].get('all_below_1e5',False))) for r in selected)
            item=dict(variant=identifier,population=population,K=k,counts_xyz=list(counts),cells_per_metre=density,
                internal_substeps=internal if internal is not None else selected[0].get('internal_substeps'),
                field_worst=errors,accuracy_pass=passed,scored_states=len(selected),
                scenes=len({r['identifier'] for r in selected}),source=source(path),interval='rollout_240' if rollout else 'single_block')
            data['accuracy'].append(item)
        return grouped
    def gpu_blocks(identifier,label,folder,protocol):
        path=root/folder/'results.json';value=result(path,'completed_expanded_cuda_full_state_block_evaluation')
        reg=read(docs/protocol,value['protocol_sha256']);witness(reg)
        if (value['cpu_model_forward'] or not value['every_intermediate_output_scored']
                or not value.get('every_measured_complete_output_hash_verified',value.get('every_actual_complete_output_hash_verified',False)) or len(value['rows'])!=960
                or value['model_identity']['numerical_pressure_solver'] or value['model_identity']['analytical_physical_inverse']):
            raise ValueError('GPU block scope differs: '+folder)
        pressure=value['model_identity']['pressure']
        if pressure['weights_sha256']!=sha(root/'shared-spectral-pair-01/restore/weights.npz'):raise ValueError('GPU pressure checkpoint differs')
        if value['learned_inventory_before_and_after'] is not True and isinstance(value['learned_inventory_before_and_after'],dict):
            inventories=value['learned_inventory_before_and_after']
            if 'before' in inventories and inventories['before']!=inventories['after']:raise ValueError('GPU evaluation changed learned tensors')
        variant(identifier,label)
        aggregate(value['rows'],identifier,'expanded_blocks',path)
        bycase=defaultdict(list)
        for row in value['rows']:bycase[(row['identifier'],row['K'],row['start'])].append(row)
        for key,selected in bycase.items():
            if {r['lead'] for r in selected}!=set(range(1,key[1]+1)) or len(selected)!=key[1]:raise ValueError('Missing intermediate GPU output')
            block_lookup[(identifier,*key)]=dict(field_worst={f:max(r['metrics']['relative'][f] for r in selected) for f in FIELDS},
                accuracy_pass=all(r['metrics']['all_accuracy_criteria_pass'] for r in selected),source=source(path))
            block_hashes[(identifier,*key)]=[r['output_sha256'] for r in sorted(selected,key=lambda r:r['lead'])]
        for timing in value.get('timings',[]):
            actual=block_lookup[(identifier,timing['identifier'],timing['K'],timing['start'])]
            for clock,stats in timing['full_block_ms'].items():
                if not math.isclose(stats['median']/timing['K'],timing['amortized_ms_per_published_step'][clock]['median'],rel_tol=1e-12):raise ValueError('GPU timing divided by internal substeps instead of K')
                scope=clock
                # Earlier resident/event measurements omit known controls.
                # Variant6 and repaired variant7 execute them inside the graph.
                if clock in ('cuda_event','synchronized_resident_wall'):
                    scope+=('_provided_controls' if identifier in ('raw','corrected','fusion1','fusion2','fusion3','fusion4','fusion5') else '_complete_controls')
                data['timing'].append(dict(actual,variant=identifier,device='gpu',population='expanded_blocks',
                    identifier=timing['identifier'],start=timing['start'],K=timing['K'],counts_xyz=timing['counts_xyz'],
                    cells_per_metre=timing['cells_per_metre'],internal_substeps=timing['internal_substeps'],scope=scope,
                    median_block_ms=stats['median'],maximum_block_ms=stats['maximum'],p95_block_ms=stats['p95'],
                    repetitions=stats['samples'],timing_source=source(path),scope_description=timing.get('full_scope',''),
                    device_model=value['device']['gpu_model']))
        data['sources'].append(source(path))
    def gpu_rollout(identifier,folder,protocol,status='completed_expanded_head_cuda_own_history_regression'):
        path=root/folder/'results.json';value=result(path,status)
        reg=read(docs/protocol,value['protocol_sha256']);witness(reg)
        if value['newly_independent_holdout'] or not value['own_history_without_teacher_resets'] or len(value['rows'])!=15360:
            raise ValueError('Legacy GPU rollout population differs')
        grouped=aggregate(value['rows'],identifier,'legacy_consumed',path,True)
        trace_rows(identifier,'legacy_consumed',path,value['rows'])
    def trace_rows(identifier,population,path,rows,context=None):
        grouped=defaultdict(list)
        for row in rows:grouped[(row.get('identifier',context.get('identifier') if context else None),row['K'])].append(row)
        for (scene,k),selected in grouped.items():
            selected.sort(key=lambda r:r['step'])
            if [r['step'] for r in selected]!=list(range(1,241)):raise ValueError('Incomplete 240-step GPU trace')
            first=context or selected[0]
            data['traces'].append(dict(variant=identifier,population=population,identifier=scene,K=k,counts_xyz=first['counts_xyz'],
                cells_per_metre=first['cells_per_metre'],internal_substeps=first.get('internal_substeps'),source=source(path),
                errors=[[r['step'],*[r['metrics']['relative'][f] for f in FIELDS]] for r in selected],
                failed_steps=sum(not r['metrics']['all_accuracy_criteria_pass'] for r in selected)))

    gpu_blocks('raw','Shared head · original arithmetic','expanded-generator-blocks-01','expanded-generator-block-protocol-01.json')
    gpu_rollout('raw','expanded-legacy-rollout-01','expanded-legacy-rollout-protocol-01.json')
    gpu_blocks('corrected','Shared head · corrected deployment','expanded-generator-blocks-02','expanded-generator-block-protocol-02.json')
    gpu_rollout('corrected','expanded-legacy-rollout-02','expanded-legacy-rollout-protocol-02.json')
    for n in range(1,8):
        identifier='repair7' if n==7 else f'fusion{n}'
        label='Complete graph · numerical repair V2' if n==7 else f'Fusion V{n}'
        gpu_blocks(identifier,label,f'fused-generator-blocks-{n:02}',f'cuda-fusion-blocks-protocol-{n:02}.json')
        gpu_rollout(identifier,f'fused-legacy-rollout-{n:02}',f'cuda-fusion-legacy-protocol-{n:02}.json')
    # Balanced complete-entry clocks are separate captures, with actual raw
    # samples. They are not relabelled resident CUDA events or pooled medians.
    for n,modes in [(1,{'standard':'Standard transfer','packed_pinned':'Packed pinned transfer'}),
                    (2,{'separate_control':'Separate controls · paired capture','complete_graph':'Complete graph V1 · paired capture'})]:
        path=root/f'gpu-complete-entry-benchmark-{n:02}'/'results.json'
        value=result(path,'completed_balanced_complete_gpu_entry_benchmark')
        reg=read(docs/f'gpu-complete-entry-benchmark-protocol-{n:02}.json',value['protocol_sha256']);witness(reg)
        if not value['every_actual_output_hash_checked'] or not value['all_actual_timed_and_warmup_output_hashes_equal_qualified_baseline']:
            raise ValueError('Complete-entry timing output identity missing')
        for mode,label in modes.items():
            identifier=f'entry{n}-{mode}';variant(identifier,label)
            data['accuracy'].extend(dict(a,variant=identifier) for a in list(data['accuracy']) if a['variant']=='corrected' and a['population']=='expanded_blocks')
            grouped=defaultdict(list)
            for row in value['rows']:
                if row['mode']==mode:grouped[(row['identifier'],row['K'],row['start'])].append(row)
            for key,selected in grouped.items():
                if len(selected)!=reg['repetitions'] or len({r['repeat'] for r in selected})!=len(selected):raise ValueError('Incomplete paired entry repeats')
                actual=block_lookup[('corrected',*key)];first=selected[0]
                times=[r['full_block_ms'] for r in selected]
                for row in selected:
                    if not math.isclose(row['full_block_ms']/row['K'],row['ms_per_published_step'],rel_tol=1e-12):raise ValueError('Complete-entry per-step scope differs')
                data['timing'].append(dict(actual,variant=identifier,device='gpu',population='expanded_blocks',identifier=key[0],start=key[2],K=key[1],
                    counts_xyz=first['counts_xyz'],cells_per_metre=first['cells_per_metre'],internal_substeps=first['internal_substeps'],
                    scope='complete_entry',median_block_ms=statistics.median(times),maximum_block_ms=max(times),p95_block_ms=None,
                    repetitions=len(times),timing_source=source(path),scope_description=value['scope'],device_model=value['gpu_model']))
    # Original completed large-grid failures remain separate from the repaired
    # short blocks and consumed legacy regressions. A live retest is not a pass.
    summary=read(docs/'streamed-expanded-rollout-results-03.json')
    for item in summary['rows']:
        path=lab/item['evidence'];value=read(path,item['evidence_sha256']);witness(value)
        if value['status']!='completed_streamed_expanded_gpu_240step_scene' or value['latency_claim'] or value['newly_independent_holdout']:
            raise ValueError('Large-grid streamed result is not complete')
        rows=[dict(r,identifier=value['identifier'],counts_xyz=value['counts_xyz'],cells_per_metre=value['cells_per_metre'],internal_substeps=value['internal_substeps']) for r in value['rows']]
        aggregate(rows,'fusion6','streamed_large',path,True)
        trace_rows('fusion6','streamed_large',path,rows)
    repair=read(docs/'learned-numerical-evaluation-repair-01.json');witness(repair)
    data['repair']=dict(source=source(docs/'learned-numerical-evaluation-repair-01.json'),
        full_large_retest_complete=repair['full_failed_scenario_revalidation_complete'],
        endpoint_metrics=repair['actual_versioned_backend_regression'],endpoint_step=46,
        maximum_latency_verified=repair['maximum_latency_target_verified'])
    training=read(docs/'expanded-multistep-data-results-03.json');witness(training)
    data['training']=dict(source=source(docs/'expanded-multistep-data-results-03.json'),training_scenes=148,development_scenes=24,reserved_scenes=52,
        training_states=training['unique_verified_training_states'],development_states=training['unique_verified_development_states'],
        complete_K4_windows=training['complete_training_K4_windows'],reserved_opened=training['reserved52_references_opened'])
    # The earlier CPU prototype now has its own measured complete calls and
    # deferred accuracy. Native controls are explicitly reference measurements.
    cpu_report=read(docs/'divisor-prototype-characterization-01.json');witness(cpu_report)
    for cores in (1,2,4):
        path=root/f'divisor-prototype-benchmark-p{cores}-01'/'results.json';value=result(path,'completed_divisor_generator_benchmark')
        accuracy_path=root/f'divisor-measured-accuracy-p{cores}-01.json';measured=result(accuracy_path,'completed_exact_measured_output_accuracy_verification')
        if measured['measurement_sha256']!=sha(path) or not measured['measured_output_identity_parity']:raise ValueError('CPU measured accuracy binding differs')
        identifier=f'divisor-cpu{cores}';variant(identifier,f'Original divisor · {cores} CPU core(s)','cpu')
        rows=[dict(r,K=r['horizons'],step=r['start']+r['lead'],identifier=r['identifier']) for r in measured['rows']]
        aggregate(rows,identifier,'divisor_measured',accuracy_path)
        bycase=defaultdict(list)
        for row in rows:bycase[(row['identifier'],row['horizons'],row['start'])].append(row)
        timings=defaultdict(list)
        for row in value['rows']:
            if not row['every_output_bitwise_verified']:raise ValueError('CPU actual timed output differs')
            timings[(row['identifier'],row['horizons'],row['start'])].append(row)
        for key,selected in timings.items():
            actual=bycase[key];first=selected[0];times=[r['complete_block_ms'] for r in selected]
            data['timing'].append(dict(variant=identifier,device='cpu',population='divisor_measured',identifier=key[0],start=key[2],K=key[1],
                counts_xyz=first['counts_xyz'],cells_per_metre=first['cells_per_metre'],internal_substeps=1,scope='complete_cpu',core_count=cores,
                median_block_ms=statistics.median(times),maximum_block_ms=max(times),p95_block_ms=None,repetitions=len(times),
                field_worst={f:max(r['metrics']['relative'][f] for r in actual) for f in FIELDS},accuracy_pass=all(r['all_accuracy_pass'] for r in actual),
                source=source(accuracy_path),timing_source=source(path),scope_description=value['timing_scope'],device_model='CPU '+','.join(map(str,value['cores']))))
    path=root/'divisor-native-control-benchmark-p1-01/results.json'
    native=result(path,'completed_native_generator_control_benchmark')
    if native['candidate_inference'] or not native['numerical_solver_is_explicit_reference_control']:
        raise ValueError('Native reference incorrectly classified as AI')
    identifier='native-cpu1';variant(identifier,'Native reference control · 1 CPU core','cpu','reference')
    groups=defaultdict(list)
    for row in native['rows']:
        if not row['warm_call']:groups[(row['identifier'],row['horizons'],row['start'])].append(row)
    for key,selected in groups.items():
        first=selected[0];times=[r['complete_block_with_state_transfer_ms'] for r in selected]
        metrics=[m for r in selected for m in r['metrics']]
        data['timing'].append(dict(variant=identifier,device='cpu',population='native_reference',identifier=key[0],start=key[2],K=key[1],
            counts_xyz=first['counts_xyz'],cells_per_metre=first['cells_per_metre'],internal_substeps=1,scope='complete_cpu',core_count=1,
            median_block_ms=statistics.median(times),maximum_block_ms=max(times),p95_block_ms=None,repetitions=len(times),
            field_worst={f:max(m['relative'][f] for m in metrics) for f in FIELDS},accuracy_pass=all(r['all_accuracy_pass'] for r in selected),
            source=source(path),timing_source=source(path),scope_description=native['timing_scope'],device_model='Native numerical reference · CPU'))
    for identifier,label,folder in [('prototype-eager','Original divisor · GPU eager','cuda-divisor-qualification-02'),
                                    ('prototype-graph','Original divisor · GPU graph','cuda-graph-divisor-qualification-02')]:
        path=root/folder/'results.json';value=result(path,'completed_cuda_divisor_block_qualification')
        if not value['every_timed_resident_output_scored'] or not value['full_transfer_output_hashes_match_resident']:
            raise ValueError('Original GPU prototype timing accuracy missing')
        variant(identifier,label)
        aggregate(value['rows'],identifier,'prototype_gpu_blocks',path)
        bycase=defaultdict(list)
        for row in value['rows']:bycase[(row['identifier'],row['K'],row['start'])].append(row)
        for timing in value['timings']:
            selected=bycase[(timing['identifier'],timing['K'],timing['start'])]
            for clock,stats in timing['full_block_ms'].items():
                scope=clock+('_provided_controls' if clock in ('cuda_event','synchronized_resident_wall') else '')
                if not math.isclose(stats['median']/timing['K'],timing['amortized_ms_per_generated_step'][clock]['median'],rel_tol=1e-12):
                    raise ValueError('Original GPU prototype timing task differs')
                data['timing'].append(dict(variant=identifier,device='gpu',population='prototype_gpu_blocks',identifier=timing['identifier'],start=timing['start'],K=timing['K'],
                    counts_xyz=timing['counts_xyz'],cells_per_metre=timing['cells_per_metre'],internal_substeps=1,scope=scope,
                    median_block_ms=stats['median'],maximum_block_ms=stats['maximum'],p95_block_ms=stats['p95'],repetitions=stats['samples'],
                    field_worst={f:max(r['metrics']['relative'][f] for r in selected) for f in FIELDS},accuracy_pass=all(r['metrics']['all_accuracy_criteria_pass'] for r in selected),
                    source=source(path),timing_source=source(path),scope_description=timing['host_input_scope'],device_model=value['device']['gpu_model']))
    gpu_rollout('prototype-graph','cuda-graph-rollout-01','cuda-graph-rollout-protocol-01.json','completed_cuda_graph_own_history_evaluation')
    if (root/'fused-generator-blocks-08/results.json').exists():
        successors={8:('sampler8','Physical sampler repair'),9:('transform9','Generic fixed-transform repair'),
            10:('allocation10','Allocation fill elimination'),11:('integer11','Integer sampler · not selected'),
            12:('affine12','Affine fusion'),13:('tables13','Sampler table elimination · not selected'),
            14:('layout14','64³ matrix-layout repair · accuracy only')}
        for n,(identifier,label) in successors.items():
            path=root/f'fused-generator-blocks-{n:02}/results.json'
            if not path.exists():continue
            gpu_blocks(identifier,label,path.parent.name,f'cuda-fusion-blocks-protocol-{n:02}.json')
            legacy=root/f'fused-legacy-rollout-{n:02}/results.json'
            if legacy.exists():
                # The terminal layout-repair regression was re-registered after
                # its 960-output gate; the run folder retains version 14.
                gpu_rollout(identifier,legacy.parent.name,f'cuda-fusion-legacy-protocol-{15 if n==14 else n:02}.json')

        # These complete-entry studies bind every actual timed output to the
        # qualified generic-transform blocks, rather than borrowing old timings.
        entries=[('entry3','GPU-execution-entry-benchmark-01/results.json','GPU-execution-entry-benchmark-protocol-01.json',
                  {'baseline':'Generic transform · allocation control','no_scratch_fill':'Allocation fill elimination · paired capture'}),
                 ('entry4','GPU-execution-entry-benchmark-03/results.json','GPU-execution-entry-benchmark-protocol-03.json',
                  {'no_scratch_fill':'Allocation fill elimination · affine control','affine_fused':'Affine fusion · paired capture'}),
                 ('entry5','gpu-workspace-entry-benchmark-01/finalized-results-01.json','gpu-workspace-entry-benchmark-protocol-02.json',
                  {'packed':'Packed transfer · workspace control','workspace':'Reusable workspace · finalized capture'})]
        for prefix,folder,protocol,modes in entries:
            path=root/folder
            if not path.exists():continue
            value=result(path,'completed_balanced_workspace_gpu_entry_benchmark_finalized_offline' if prefix=='entry5' else 'completed_balanced_execution_gpu_entry_benchmark')
            reg=read(docs/protocol,value['protocol_sha256']);witness(reg)
            if prefix=='entry5':
                read(docs/'gpu-workspace-benchmark-finalization-protocol-01.json',value['finalization_protocol_sha256'])
                if (sha(lab/value['raw_measurement_file'])!=value['raw_measurement_sha256']
                        or not value['offline_finalization_only'] or value['new_inference_or_timing_samples_executed']
                        or not value['every_actual_timed_output_hash_verified_offline']):
                    raise ValueError('Workspace offline finalization custody differs')
            elif not value['every_actual_timed_and_warmup_output_hash_verified']:
                raise ValueError('Execution entry output qualification missing')
            if reg['baseline_outputs']!='runs/full-ai-data-diversity-v1/fused-generator-blocks-09/results.json':
                raise ValueError('Execution benchmark baseline differs')
            for mode,label in modes.items():
                identifier=f'{prefix}-{mode}';variant(identifier,label)
                data['accuracy'].extend(dict(a,variant=identifier) for a in list(data['accuracy']) if a['variant']=='transform9' and a['population']=='expanded_blocks')
                grouped=defaultdict(list)
                for row in value['rows']:
                    if row['mode']==mode:grouped[(row['identifier'],row['K'],row['start'])].append(row)
                for key,selected in grouped.items():
                    if len(selected)!=reg['repetitions'] or len({r['repeat'] for r in selected})!=len(selected):
                        raise ValueError('Incomplete execution entry repeats')
                    for row in selected:
                        if (row['actual_complete_output_sha256']!=block_hashes[('transform9',*key)]
                                or not math.isclose(row['full_block_ms']/row['K'],row['ms_per_generated_step'],rel_tol=1e-12)):
                            raise ValueError('Actual execution entry output or per-step timing differs')
                    actual=block_lookup[('transform9',*key)];first=selected[0];times=[r['full_block_ms'] for r in selected]
                    data['timing'].append(dict(actual,variant=identifier,device='gpu',population='expanded_blocks',identifier=key[0],start=key[2],K=key[1],
                        counts_xyz=first['counts_xyz'],cells_per_metre=first['cells_per_metre'],internal_substeps=first['internal_substeps'],
                        scope='complete_entry',median_block_ms=statistics.median(times),maximum_block_ms=max(times),p95_block_ms=None,
                        repetitions=len(times),timing_source=source(path),scope_description=value['scope'],device_model=value['gpu_model']))

        # Only terminal scene files enter the charts. Each scene is a complete
        # 240-step own-history trajectory for every K, including failed cases.
        streamed={2:'repair7',3:'sampler8',4:'transform9',5:'affine12',6:'layout14',7:'layout14'}
        native_cases={}
        for n,identifier in streamed.items():
            protocol=docs/f'streamed-expanded-rollout-protocol-{n:02}.json'
            if not protocol.exists():continue
            for path in sorted((root/f'streamed-expanded-rollout-{n:02}').glob('*/results.json')):
                value=read(path)
                if value['status']!='completed_streamed_expanded_gpu_240step_scene':continue
                witness(value);reg=read(protocol,value['protocol_sha256']);witness(reg)
                proc=value['native_process']
                if (value['latency_claim'] or value['newly_independent_holdout'] or value['optimizer_updates']
                        or value['expanded_reserved52_references_opened'] or value['cpu_model_forward']
                        or not value['candidate_own_history_without_teacher_resets']
                        or not value['every_intermediate_physical_output_scored'] or len(value['rows'])!=960
                        or proc['returncode']!=0 or not proc['reaped'] or not proc['graceful_protocol_complete']):
                    raise ValueError('Expanded native history scope differs')
                population='streamed_large' if n<=4 else 'expanded_development'
                rows=[dict(r,identifier=value['identifier'],counts_xyz=value['counts_xyz'],cells_per_metre=value['cells_per_metre'],internal_substeps=value['internal_substeps']) for r in value['rows']]
                aggregate(rows,identifier,population,path,True);trace_rows(identifier,population,path,rows)
                if n==4:native_cases[value['identifier']]=(path,rows)

        # Reuse native errors only after checking exact output hashes for all
        # 3,840 states of an execution's separately captured own-history run.
        for n,identifier in [(1,'allocation10'),(2,'integer11'),(3,'affine12'),(4,'tables13'),(5,'layout14')]:
            path=root/f'GPU-full-history-equivalence-{n:02}/results.json'
            if not path.exists():continue
            value=read(path)
            if value['status']!='passed_all_four_GPU_own_history_byte_equivalence':continue
            witness(value);reg=read(docs/f'GPU-full-history-equivalence-protocol-{6 if n==5 else n:02}.json',value['protocol_sha256']);witness(reg)
            if (not value['every_output_exact_to_native_qualified_baseline'] or not value['own_history_without_teacher_resets']
                    or value['reference_teacher_executed'] or value['optimizer_updates'] or value['expanded_reserved52_references_opened']
                    or len(value['rows'])!=3840):raise ValueError('Own-history equivalence scope differs')
            indexed={(r['identifier'],r['K'],r['step']):r for r in value['rows']}
            if len(indexed)!=3840:raise ValueError('Duplicate history equivalence states')
            for case in value['completed_cases']:
                baseline,rows=native_cases[case['identifier']]
                if lab/case['baseline_results']!=baseline or sha(baseline)!=case['baseline_sha256']:
                    raise ValueError('History equivalence native baseline differs')
                for row in rows:
                    actual=indexed[(row['identifier'],row['K'],row['step'])]
                    if not actual['native_qualified_baseline_exact'] or actual['output_sha256']!=row['output_sha256']:
                        raise ValueError('Actual own-history output differs from native-qualified baseline')
                start=len(data['traces'])
                aggregate(rows,identifier,'streamed_large',baseline,True)
                trace_rows(identifier,'streamed_large',baseline,rows)
                for trace in data['traces'][start:]:trace['equivalence_source']=source(path)

        qualification=read(docs/'expanded-generator-qualification-11.json')
        complete={t['identifier'] for t in data['traces'] if t['variant']=='layout14'}
        development={s for s in complete if s.startswith('diversity-development-')}
        data['progress']=dict(source=source(docs/'expanded-generator-qualification-11.json'),
            fixed_native_cases=len(native_cases),required_fixed_cases=4,
            layout_development_cases=len(development),required_development_cases=24,
            layout_has_timing=any(t['variant']=='layout14' for t in data['timing']),
            new_failure_source=source(docs/'cube64-density100-rollout-failure-01.json'),
            full_generator_targets_verified=qualification['full_generator_target_pass'],reserved_opened=qualification['expanded_reserved52_references_opened'])
    from expanded_followup_chart import enrich_expanded_followups
    enrich_expanded_followups(data,base,lab,load)
    study['expanded_models']=data
