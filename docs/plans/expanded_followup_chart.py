"""Import terminal expanded-model follow-ups without executing any models.

Block and history errors are inherited only after checking actual output hashes.
Train/development populations and measured work per published step stay separate.
"""
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import statistics

FIELDS=('equation','pressure','velocity','tracer')


def enrich_expanded_followups(data,base,lab,load):
    base,lab=Path(base),Path(lab)
    docs=base/'full-ai-data-diversity-v1';root=lab/'runs/full-ai-data-diversity-v1'
    if not (docs/'development-full240-qualification-01.json').exists():return
    hashes={}
    def sha(path):
        path=Path(path)
        if path not in hashes:
            with path.open('rb') as stream:hashes[path]=hashlib.file_digest(stream,'sha256').hexdigest()
        return hashes[path]
    def source(path):return os.path.relpath(path,base)
    def witness(value):
        for key in ('source_sha256','source_evidence','evidence_sha256','reference_sources'):
            for name,expected in (value.get(key) or {}).items():
                if isinstance(expected,str) and len(expected)==64 and sha(lab/name)!=expected:
                    raise ValueError('Follow-up evidence changed: '+name)
    protocols={sha(p):p for p in docs.glob('*protocol*.json')}
    def read(path,expected=None,registered=True):
        value=load(Path(path),expected)
        witness(value)
        if registered and value.get('protocol_sha256'):
            protocol=protocols.get(value['protocol_sha256'])
            if protocol is None:raise ValueError('Missing exact registered follow-up protocol: '+str(path))
            reg=load(protocol,value['protocol_sha256']);witness(reg)
        if (value.get('optimizer_updates',0) or value.get('cpu_model_forward',False)
                or value.get('expanded_reserved52_references_opened',0)):
            raise ValueError('Follow-up evaluation changed learning, device or reserved scope')
        return value
    def add_variant(identifier,label,kind='execution'):
        data['variants'].append(dict(id=identifier,label=label,device='gpu',kind=kind))
    def passed(row):
        return row.get('all_accuracy_pass',row['metrics'].get('all_accuracy_criteria_pass',row['metrics'].get('all_below_1e5',False)))
    def internal(row):
        work=row.get('actual_internal_advances_per_block',row.get('candidate_internal_advances_per_block'))
        if work is not None:
            if work%row['K']:raise ValueError('Nonintegral work per published step')
            return work//row['K']
        return row.get('internal_substeps',row.get('student_stages'))
    def aggregate(rows,identifier,path,interval='single_block',population=None,equivalence=None):
        grouped=defaultdict(list)
        for row in rows:
            grouped[(row['K'],tuple(row['counts_xyz']),row['cells_per_metre'],row.get('split','development'),internal(row))].append(row)
        for (k,counts,density,split,substeps),selected in grouped.items():
            errors={f:max(r['metrics']['relative'][f] for r in selected) for f in FIELDS}
            if not all(math.isfinite(e) and e>=0 for e in errors.values()):raise ValueError('Invalid follow-up error')
            item=dict(variant=identifier,population=population or ('training_blocks' if split=='train' else 'expanded_blocks'),split=split,
                K=k,counts_xyz=list(counts),cells_per_metre=density,internal_substeps=substeps,
                teacher_substeps=selected[0].get('teacher_substeps',8 if any('teacher_internal_advances_per_block' in r for r in selected) else None),field_worst=errors,
                accuracy_pass=all(passed(r) for r in selected),scored_states=len(selected),scenes=len({r['identifier'] for r in selected}),
                source=source(path),interval=interval)
            if equivalence:item['equivalence_source']=source(equivalence)
            data['accuracy'].append(item)
    def histories(rows,identifier,path,population,equivalence=None):
        grouped=defaultdict(list)
        for row in rows:grouped[(row['identifier'],row['K'])].append(row)
        for (scene,k),selected in grouped.items():
            selected.sort(key=lambda r:r['step'])
            if [r['step'] for r in selected]!=list(range(1,241)):raise ValueError('Incomplete follow-up history')
            first=selected[0]
            item=dict(variant=identifier,population=population,identifier=scene,K=k,counts_xyz=first['counts_xyz'],
                cells_per_metre=first['cells_per_metre'],internal_substeps=internal(first),source=source(path),
                errors=[[r['step'],*[r['metrics']['relative'][f] for f in FIELDS]] for r in selected],failed_steps=sum(not passed(r) for r in selected))
            if equivalence:item['equivalence_source']=source(equivalence)
            data['traces'].append(item)
        aggregate(rows,identifier,path,'rollout_240',population,equivalence)

    native_cache={}
    def native(path,expected=None):
        path=Path(path)
        if expected and sha(path)!=expected:raise ValueError('Native scored source hash differs')
        if path not in native_cache:
            value=read(path)
            proc=value['native_process']
            if (value['status']!='completed_streamed_expanded_gpu_240step_scene' or len(value['rows'])!=960
                    or value['latency_claim'] or value['newly_independent_holdout']
                    or not value['candidate_own_history_without_teacher_resets'] or not value['every_intermediate_physical_output_scored']
                    or proc['returncode']!=0 or not proc['reaped'] or not proc['graceful_protocol_complete']):
                raise ValueError('Native scored history scope differs')
            native_cache[path]=[dict(r,identifier=value['identifier'],counts_xyz=value['counts_xyz'],
                cells_per_metre=value['cells_per_metre'],internal_substeps=value['internal_substeps'],split=value['source_split']) for r in value['rows']]
        return native_cache[path]

    development=read(docs/'development-full240-qualification-01.json')
    if development['scene_count']!=24 or not development['all_accuracy_pass'] or development['independent_confirmation_verified']:
        raise ValueError('Expanded development coverage differs')
    existing={(t['variant'],t['identifier'],t['K']) for t in data['traces']}
    for item in development['rows']:
        path=lab/item['path'];rows=native(path,item['sha256'])
        if not all(passed(r) for r in rows):raise ValueError('Development qualification contradicts scored states')
        if ('layout14',item['identifier'],1) not in existing:
            histories(rows,'layout14',path,'expanded_development')

    # Each registry entry represents an actual complete prediction execution or
    # a fitted student. Primitive-only and capacity estimates are kept out of it.
    trials=[
        ('reciprocal16','Scalar reciprocal · accuracy only','fused-generator-blocks-16','fused-legacy-rollout-16','GPU-full-history-equivalence-06'),
        ('gather17','Gather fusion','fused-generator-blocks-17','fused-legacy-rollout-17','GPU-full-history-equivalence-07'),
        ('descriptor18','Grid descriptor · not selected','fused-generator-blocks-18','fused-legacy-rollout-18','GPU-full-history-equivalence-08'),
        ('geometry19','Geometry specialization V1 · not selected','fused-generator-blocks-19','fused-legacy-rollout-19','GPU-full-history-equivalence-09'),
        ('geometry20','Geometry specialization V2 · not selected','fused-generator-blocks-20','fused-legacy-rollout-20','GPU-full-history-equivalence-10'),
        ('native-entry1','Complete native entry V1','complete-native-entry-blocks-01','complete-native-entry-legacy-01','complete-native-entry-histories-01'),
        ('native-entry2','Geometry native entry · retained baseline','complete-native-entry-blocks-02','complete-native-entry-legacy-02','complete-native-entry-histories-02'),
        ('guarded-terminal','Guarded terminal · profile rejected','guarded-terminal-blocks-01','guarded-terminal-legacy-01','guarded-terminal-histories-01'),
        ('guarded-middle','Guarded middle · profile rejected','guarded-middle-blocks-01','guarded-middle-legacy-01','guarded-middle-histories-01'),
        ('integer-quotient','Integer quotient · accuracy only','integer-quotient-blocks-01','integer-quotient-legacy-01','integer-quotient-histories-01'),
        ('absolute-eager','Absolute learned pressure · eager','absolute-pressure-blocks-01',None,None),
        ('absolute-graph2','Absolute learned pressure · graph V2','absolute-pressure-graph-blocks-02',None,None),
        ('absolute-regime','Absolute N8 pressure · original N1','absolute-pressure-graph-blocks-03',None,'absolute-pressure-history-inheritance-01'),
        ('nested-F64','Nested F64 interpolation','nested-F64-blocks-01','nested-F64-legacy-01','nested-F64-histories-01'),
        ('bound-sampler','Bound parameter sampler','bound-sampler-blocks-01','bound-sampler-legacy-01','bound-sampler-histories-01'),
        ('cell-coordinate','N8 cell-coordinate sampler','cell-coordinate-blocks-01','cell-coordinate-legacy-01','cell-coordinate-histories-02'),
        ('FFT-pressure','F64 FFT pressure · slower trial','FFT-pressure-blocks-01','FFT-pressure-legacy-01','FFT-pressure-histories-01'),
        ('FP32-interpolator','FP32 interpolation · accuracy rejected','FP32-interpolator-blocks-01',None,None),
        ('student3','Temporal student · 3 stages','macrostep-student-s3-blocks-01',None,None),
        ('student4','Temporal student · 4 stages','macrostep-student-s4-blocks-01',None,None),
        ('alpha-argument','Exact alpha argument reuse','alpha-argument-blocks-01','alpha-argument-legacy-01','alpha-argument-histories-01'),
        ('paired-interpolator','Compensated interpolation · accuracy rejected','paired-interpolator-blocks-01',None,None),
        ('residual-s1','Internal residual S1 · trained, accuracy failed','internal-residual-s1-native-blocks-01',None,None),
        ('residual-s2','Internal residual S2 · initial checkpoint selected','internal-residual-s2-native-blocks-01',None,None),
        ('residual-graph','Internal residual S1 · plain GPU graph','internal-residual-s1-native-graph-01',None,None),
        ('residual-fused','Internal residual S1 · fused GPU graph','internal-residual-s1-native-fused-graph-01',None,None),
        ('residual-s3','Internal residual S3 · physical loss','internal-residual-s3-native-blocks-01',None,None),
        ('residual-s4','Internal residual S4 · delta MSE','internal-residual-s4-native-blocks-01',None,None),
        ('residual-s3-fused','Internal residual S3 · fused GPU graph','internal-residual-s3-native-fused-graph-01',None,None),
        ('residual-s4-fused','Internal residual S4 · fused GPU graph','internal-residual-s4-native-fused-graph-01',None,None),
    ]
    blocks={}
    for identifier,label,block,legacy,history in trials:
        path=root/block/'results.json'
        if not path.exists():continue
        value=read(path)
        if not value['status'].startswith('completed_'):continue
        if len(value['rows']) not in (960,1018,2120) or not value.get('every_intermediate_output_scored',value.get('every_K_intermediate_scored',False)):
            raise ValueError('Follow-up block coverage differs: '+block)
        add_variant(identifier,label,'fit' if identifier.startswith('student') or identifier in ('residual-s1','residual-s2','residual-s3','residual-s4') else 'execution')
        aggregate(value['rows'],identifier,path)
        # Keep only small block hashes and metrics for matching timed calls.
        cases=defaultdict(list)
        for row in value['rows']:cases[(row['identifier'],row['K'],row['start'])].append(row)
        blocks[identifier]=(path,{k:sorted(v,key=lambda r:r['lead']) for k,v in cases.items()})
        if legacy:
            legacy_path=root/legacy/'results.json';v=read(legacy_path)
            if v['status']!='completed_expanded_head_cuda_own_history_regression' or len(v['rows'])!=15360 or v['newly_independent_holdout']:
                raise ValueError('Follow-up legacy scope differs')
            histories(v['rows'],identifier,legacy_path,'legacy_consumed')
        if history:
            history_path=root/history/'results.json';v=read(history_path)
            if not v['status'].startswith('passed_all') or len(v['rows']) not in (23040,24960):raise ValueError('Follow-up history is not complete')
            indexed={(r['identifier'],r['K'],r['step']):r for r in v['rows']}
            cases=v.get('completed_cases',v.get('cases'))
            if len(indexed)!=len(v['rows']):raise ValueError('Duplicate history states')
            for case in cases:
                baseline=lab/case.get('baseline_results',case.get('native_scored_source'))
                rows=native(baseline,case.get('baseline_sha256',case.get('native_scored_source_sha256')))
                for row in rows:
                    actual=indexed[(row['identifier'],row['K'],row['step'])]
                    if not actual.get('all_field_bits_exact',actual.get('native_qualified_baseline_exact',False)) or actual['output_sha256']!=row['output_sha256']:
                        raise ValueError('Actual complete history output does not match scored native source')
                # Retain the fixed four and all-24 population as independently
                # selectable views, with each native source explicitly linked.
                population='expanded_development' if rows[0]['split']=='development' else 'streamed_large'
                histories(rows,identifier,baseline,population,history_path)

    # Untrained controls predict full states too; they do not become fitted models.
    for folder,prefix in [('macrostep-controls-01','control'),('grouped-macrostep-01','grouped')]:
        path=root/folder/'results.json'
        if not path.exists():continue
        value=read(path)
        if not value['status'].startswith('completed_'):raise ValueError('Incomplete macrostep control')
        for mode in sorted({r['mode'] for r in value['rows']}):
            identifier=f'{prefix}-{mode}';add_variant(identifier,mode.replace('_',' ')+' · untrained control','control')
            selected=[r for r in value['rows'] if r['mode']==mode]
            aggregate(selected,identifier,path)

    students=load(docs/'macrostep-student-pilot-results-01.json');witness(students)
    for student in students['students']:
        pair_path=lab/student['pair_receipt'];pair=load(pair_path)
        restore=load(pair_path.parent/'restore/receipt.json',pair['restore_receipt_sha256'])
        train=load(pair_path.parent/'train/receipt.json',pair['training_receipt_sha256'])
        if (pair['status']!='completed' or pair['optimizer_updates']!=160 or pair['parameters']!=student['parameters']
                or restore['optimizer_updates'] or train['optimizer_updates']!=160 or pair['model_promotion']):
            raise ValueError('Temporal student training/restore custody differs')
        for filename,parity in pair['artifact_parity'].items():
            if not parity or sha(pair_path.parent/'train'/filename)!=sha(pair_path.parent/'restore'/filename):
                raise ValueError('Temporal student artifact parity differs')
    data['students']=dict(source=source(docs/'macrostep-student-pilot-results-01.json'),students=students['students'],
        additional_optimizer_calls=students['new_optimizer_calls'],additional_fits=students['new_managed_fits'])

    # A separate execution ID for every mode/capture prevents repeated controls
    # from silently replacing earlier timings. All costs retain every K output.
    benchmarks=[
        ('entry6','GPU-execution-entry-benchmark-04',{'current_qualified':'reciprocal16','gather_fused':'gather17'}),
        ('entry7','GPU-execution-entry-benchmark-05',{'gather_fused':'gather17','grid_descriptor':'descriptor18'}),
        ('entry8','complete-native-entry-benchmark-01',{'gather_fused':'gather17','complete_native_entry':'native-entry1'}),
        ('entry9','complete-native-entry-benchmark-02',{'gather_fused':'gather17','complete_native_entry':'native-entry1','geometry_native_entry':'native-entry2'}),
        ('entry10','absolute-pressure-benchmark-01',{'geometry_native_entry':'native-entry2','absolute_regime_entry':'absolute-regime'}),
        ('entry11','nested-F64-benchmark-01',{'absolute_regime_entry':'absolute-regime','nested_F64_entry':'nested-F64'}),
        ('entry12','bound-sampler-benchmark-01',{'nested_F64_entry':'nested-F64','bound_sampler_entry':'bound-sampler'}),
        ('entry13','cell-coordinate-benchmark-01',{'bound_sampler_entry':'bound-sampler','cell_coordinate_entry':'cell-coordinate'}),
        ('entry14','FFT-pressure-benchmark-01',{'cell_coordinate_entry':'cell-coordinate','FFT_pressure_entry':'FFT-pressure'}),
        ('entry15','alpha-argument-benchmark-01',{'cell_coordinate_entry':'cell-coordinate','alpha_argument_entry':'alpha-argument'}),
        ('entry16','paired-interpolator-diagnostic-benchmark-01',{'cell_coordinate_entry':'cell-coordinate','paired_interpolator_entry':'paired-interpolator'}),
    ]
    labels={v['id']:v['label'] for v in data['variants']}
    for prefix,folder,modes in benchmarks:
        path=root/folder/'results.json'
        if not path.exists():continue
        value=read(path)
        expected_status='completed_accuracy_failed_candidate_diagnostic_GPU_benchmark' if prefix=='entry16' else 'completed_balanced_execution_gpu_entry_benchmark'
        if value['status']!=expected_status or not value['every_actual_timed_and_warmup_output_hash_verified']:
            raise ValueError('Follow-up complete entry benchmark incomplete')
        reg=load(protocols[value['protocol_sha256']],value['protocol_sha256'])
        if set(modes)!=set(reg['modes']):raise ValueError('Complete entry modes differ')
        for mode,parent in modes.items():
            identifier=f'{prefix}-{mode}';add_variant(identifier,labels[parent]+' · '+folder)
            accuracy_path,cases=blocks[parent]
            data['accuracy'].extend(dict(a,variant=identifier) for a in list(data['accuracy']) if a['variant']==parent and a['interval']=='single_block' and a.get('split','development')=='development')
            grouped=defaultdict(list)
            for row in value['rows']:
                if row['mode']==mode:grouped[(row['identifier'],row['K'],row['start'])].append(row)
            for key,selected in grouped.items():
                scored=cases[key];first=selected[0]
                if len(scored)!=key[1] or [r['lead'] for r in scored]!=list(range(1,key[1]+1)):
                    raise ValueError('Missing scored intermediate block state')
                if len(selected)!=reg['repetitions'] or len({r['repeat'] for r in selected})!=len(selected):raise ValueError('Incomplete entry repeats')
                for row in selected:
                    if (row['actual_complete_output_sha256']!=[r['output_sha256'] for r in scored]
                            or not math.isclose(row['full_block_ms']/key[1],row['ms_per_generated_step'],rel_tol=1e-12)):
                        raise ValueError('Timed follow-up state or per-step clock differs')
                times=[r['full_block_ms'] for r in selected]
                data['timing'].append(dict(variant=identifier,device='gpu',population='expanded_blocks',split='development',
                    identifier=key[0],start=key[2],K=key[1],counts_xyz=first['counts_xyz'],cells_per_metre=first['cells_per_metre'],
                    internal_substeps=first['internal_substeps'],scope='diagnostic_complete_entry' if prefix=='entry16' else 'complete_entry',median_block_ms=statistics.median(times),maximum_block_ms=max(times),
                    p95_block_ms=None,repetitions=len(times),field_worst={f:max(r['metrics']['relative'][f] for r in scored) for f in FIELDS},
                    accuracy_pass=all(passed(r) for r in scored),source=source(accuracy_path),timing_source=source(path),scope_description=value['scope'],device_model=value['gpu_model']))

    if (docs/'internal-residual-s1-results-01.json').exists():
        data['residual_fits']=[];data['diagnostic_timing']=[]
        for name in ('s1','s2','s3','s4'):
            if not (docs/f'internal-residual-{name}-results-01.json').exists():continue
            report=load(docs/f'internal-residual-{name}-results-01.json');witness(report)
            pair_path=lab/report.get('managed_pair',report.get('pair'));pair=load(pair_path)
            train=load(pair_path.parent/'train/receipt.json',pair['training_receipt_sha256'])
            restore=load(pair_path.parent/'restore/receipt.json',pair['restore_receipt_sha256'])
            history=load(pair_path.parent/'train/history.json')
            if (pair['status']!='completed' or pair['parameters']!=10876 or pair['optimizer_updates']!=report['optimizer_updates']
                    or train['optimizer_updates']!=pair['optimizer_updates'] or restore['optimizer_updates'] or pair['model_promotion']):
                raise ValueError('Residual fit training/restore custody differs')
            for filename,parity in pair['artifact_parity'].items():
                if not parity or sha(pair_path.parent/'train'/filename)!=sha(pair_path.parent/'restore'/filename):
                    raise ValueError('Residual fit artifact parity differs')
            if name=='s2' and (history['selected_epoch']!=0 or sha(pair_path.parent/'restore/weights.npz')!=sha(pair_path.parent/'restore/initial-weights.npz')):
                raise ValueError('S2 selected initial control differs')
            item=dict(id=f'residual-{name}',source=source(docs/f'internal-residual-{name}-results-01.json'),
                parameters=pair['parameters'],optimizer_updates=pair['optimizer_updates'],selected_epoch=history.get('selected_epoch'),
                training_scenes=148,active_scenes=29,control_scenes=119,native_outputs=report.get('native_outputs',report.get('native_scored_outputs')),native_failed=report.get('native_failing_outputs',report.get('native_failed_outputs')))
            if name in ('s3','s4'):
                item.update(selected_evaluation=history['selected_evaluation'],closure_evaluations=history['closure_evaluations'],objective=report['training_objective_kind'])
            data['residual_fits'].append(item)

        for variant,parent_id in [('residual-graph','residual-s1'),('residual-fused','residual-s1'),('residual-s3-fused','residual-s3'),('residual-s4-fused','residual-s4')]:
            if variant not in blocks:continue
            _,parent=blocks[parent_id];_,candidate=blocks[variant]
            if parent.keys()!=candidate.keys():raise ValueError('Residual graph case coverage differs')
            for key in parent:
                if [r['output_sha256'] for r in parent[key]]!=[r['output_sha256'] for r in candidate[key]]:
                    raise ValueError('Residual graph changed complete native outputs')

        # These timing records lack per-timed-call output hashes. Keep them in
        # a separate diagnostic cost plot without joining an accuracy coordinate.
        diagnostics=[('diagS1','internal-residual-s1-GPU-diagnostic-timing-01',{'eager':'residual-s1'}),
                     ('diagS2','internal-residual-s2-GPU-diagnostic-timing-01',{'eager':'residual-s2'}),
                     ('diagGraph','internal-residual-s1-GPUgraph-01',{'eager':'residual-s1','graph':'residual-graph'}),
                     ('diagFused','internal-residual-s1-fused-GPUgraph-01',{'eager':'residual-s1','fused_graph':'residual-fused'}),
                     ('diagPaired','internal-residual-s1-paired-graphs-01',{'graph':'residual-graph','fused_graph':'residual-fused'}),
                     ('diagS3','internal-residual-s3-GPU-timing-01',{'fused_graph':'residual-s3-fused'}),
                     ('diagS4','internal-residual-s4-GPU-timing-01',{'fused_graph':'residual-s4-fused'})]
        for prefix,folder,modes in diagnostics:
            path=root/folder/'results.json'
            if not path.exists():continue
            value=read(path);reg=load(protocols[value['protocol_sha256']],value['protocol_sha256'])
            if not value['status'].startswith('completed_') or value['acceptance_latency_qualification'] or value['full_generator_target_pass']:
                raise ValueError('Residual diagnostic scope differs')
            if 'checked' in value and not value['every_checked_output_byte_exact']:raise ValueError('Residual graph subset parity differs')
            for mode,parent in modes.items():
                identifier=f'{prefix}-{mode}';add_variant(identifier,labels[parent]+' · '+folder)
                data['accuracy'].extend(dict(a,variant=identifier) for a in list(data['accuracy']) if a['variant']==parent and a['interval']=='single_block')
                groups=defaultdict(list)
                for row in value['rows']:
                    if row.get('mode',next(iter(modes)) if len(modes)==1 else 'eager')==mode:groups[(row.get('identifier',value.get('identifier')),row['K'])].append(row)
                for (scene,k),selected in groups.items():
                    start=reg.get('timing_start',reg['start']);_,cases=blocks[parent];scored=cases[(scene,k,start)]
                    if len({r['repeat'] for r in selected})!=len(selected):raise ValueError('Duplicate residual diagnostic repeats')
                    times=[r.get('wall_block_ms',r.get('block_ms')) for r in selected]
                    for row,time in zip(selected,times):
                        if not math.isclose(time/k,row.get('wall_ms_per_generated_step',row.get('ms_per_generated_step')),rel_tol=1e-12):
                            raise ValueError('Residual diagnostic timing scope differs')
                    first=scored[0]
                    data['diagnostic_timing'].append(dict(variant=identifier,parent_variant=parent,device='gpu',split=first['split'],
                        identifier=scene,start=start,K=k,counts_xyz=first['counts_xyz'],cells_per_metre=first['cells_per_metre'],
                        internal_substeps=internal(first),median_block_ms=statistics.median(times),maximum_block_ms=max(times),
                        repetitions=len(times),source=source(path),scope_description=value['scope'],device_model=value.get('gpu_model',value.get('GPU_model')),
                        accuracy_paired_to_actual_timed_outputs=False))

    latest_path=docs/('expanded-generator-qualification-37.json' if (docs/'expanded-generator-qualification-37.json').exists() else 'expanded-generator-qualification-35.json' if (docs/'expanded-generator-qualification-35.json').exists() else 'expanded-generator-qualification-25.json')
    latest=load(latest_path);witness(latest)
    data['current_status']=dict(source=source(latest_path),
        development_source=source(docs/'development-full240-qualification-01.json'),development_cases=24,development_outputs=23040,
        retained_baseline='native-entry2',full_generator_targets_verified=latest['full_generator_targets_verified'],reserved_opened=latest['expanded_reserved52_references_opened'],
        note='All 24 consumed development scenes passed. Geometry native entry remains the retained baseline; no independent 52-scene, full speed/scaling or 600-step qualification. FP32 interpolation failed native block accuracy before timing; FFT pressure passed accuracy but was slower than its parent. Both temporal students failed the strict N8 block target and have no measured timing or full-history qualification.')
    if data.get('residual_fits'):
        data['current_status']['note']+=' Internal-residual S1 and loss-conditioned S2 failed native block accuracy; S2 selected the initial epoch-zero control. Graph and fused-graph executions preserve S1 accuracy failures. Their subset timing diagnostics do not establish an accepted speedup.'
    if any(r['id']=='residual-s3' for r in data.get('residual_fits',[])):
        data['current_status']['note']+=' S3 physical-loss and S4 delta-MSE fits each retain 1,151 failed native outputs; their warm fused-graph timing is a separate diagnostic capture.'
    from temporal_pilot_chart import enrich_temporal_pilots
    enrich_temporal_pilots(data,base,lab,load)
