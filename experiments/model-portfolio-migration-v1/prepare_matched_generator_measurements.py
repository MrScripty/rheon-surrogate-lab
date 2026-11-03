"""Package35original fits and4matched F64executions in native K-specific tables.

Keep all original per-phase samples and every scored physical lead in each
portable table. Historical instrumented packets remain separate and exact.
This reads metadata only and never starts inference or mutates the project.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from prepare_common_generator_measurements import block_values
from prepare_portfolio_measurements import source

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = RUN/'matched-generator-GPU-measurements-01'
NATIVE = RUN/'sources/Tuldok-composition-source-artifacts'


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(value,message):
    if not value:
        raise ValueError(message)


def prepare():
    require(not OUTPUT.exists(),'Preserve original matched native-table preparation')
    fit_path = DOC/'reports/complete-generator-GPU-benchmark-01.json'
    f64_path = DOC/'reports/same-case-F64-GPU-benchmark-02.json'
    comparison_path = DOC/'reports/matched-generator-F64-comparison-01.json'
    fits,f64,comparison = read(fit_path),read(f64_path),read(comparison_path)
    require(fits['status'] == 'all35_original_generators_common_GPU_primary_and_hook_calibration_reduced'
            and f64['status'] == 'all_K1to4_F64_cell_same64_native_cases_reduced'
            and comparison['status'] == 'all35_original_fit_summaries_compared_with_same_K_F64_native_baseline'
            and fits['models_count'] == 35 and not fits['competing_GPU_observations'] and not f64['competing_GPU_observations'],
            'Complete matched reductions with recorded GPUresource scope required')
    for report in (fits,f64):
        for spec in report['evidence']:
            require(binding(ROOT/spec['path']) == spec,'Accepted benchmark evidence changed')
    captured = read(RUN/'complete-generator-GPU-protocol-01.json')
    models = {m['fit_item_id']:m for m in captured['models']}
    cases = read(ROOT/captured['cases']['path'])['cases']
    fit_owner = read(RUN/'complete-generator-GPU-benchmark-01/status.json')
    f64_owner = read(RUN/'same-case-F64-GPU-benchmark-02/status.json')
    require(fit_owner['status'] == 'all70_original_workers_complete_reduction_pending'
            and f64_owner['status'] == 'all8_original_matched_F64_workers_complete_reduction_pending',
            'Both original serial owners must be terminal')
    fit_raw = {(b['fit_item_id'],b['phase']):ROOT/b['result'] for b in fit_owner['completed']}
    f64_raw = {(b['K'],b['phase']):ROOT/b['result'] for b in f64_owner['completed']}
    template_path = RUN/'common-generator-GPU-measurements-01/K1-measurements.json'
    template = read(template_path)
    template_context = json.loads(base64.b64decode(next(s for s in template['sources'] if s['id'] == 'context')['data_base64']))
    metrics = copy.deepcopy(template_context['metrics'])
    for metric in metrics:
        if 'timing_scope' in metric:
            metric['timing_scope'] = 'Native CPU5fields through layout checks,packing,known forcing,H2D,CUDAprediction,reconstruction,allKowned CPUoutputs/D2H and synchronization. No forward-validation hook in primary calls.'
            metric['exclusions'] += ' Forward-device validation is confined to setup/first/warmups; separate paired calibration is excluded.'
    dimensions = copy.deepcopy(template['dimensions']) + [
        dict(id='origin_kind',label='Fit or assembly execution',type='string',units=None,scope='subject')]
    for dimension in dimensions:
        if dimension['id'] == 'fit_item_id':
            dimension.update(id='fit_or_execution_id',label='Fit or assembly execution identity')
    sys.path.insert(0,str(NATIVE))
    from retained_measurements import MeasurementTable,encode,digest
    from projects import manifest
    staged = []
    accepted = [binding(p) for p in (fit_path,f64_path,comparison_path)]
    f64_models = {m['K']:m for m in f64['models']}
    for k in (1,2,3,4):
        scope = dict(id=f'old-common64-production-GPU-K{k}-v2',
            label=f'Matched GPU generators and F64 baseline · K{k} · same64cases',population='consumed',
            task='Complete native pressure,staggered velocities and tracer at every physical output.',
            independence='All16original fitting scenes,fourstarts;not independent holdout or own-history drift.',
            reference_binary_sha256=captured['reference_binary_sha256'],timestep_s=1/60,physical_outputs=k,block_duration_s=k/60,
            comparison='SameK,grid density,dimensions and current/native-reference identity. Historical instrumented scopes are incompatible.',
            timing_instrumentation='Primary calls have no forward-validation hooks; setup,first,warmups and paired hook calibration separate.',
            baseline='Original cell-coordinate native entry uses original sequential N1history branch. N8coordinate change unused; no learned joint Kgeneration.',
            original_quoted_anchor_cases_included=False,F64_cell_baseline_measured_on_this_population=True,
            temporal_fine_clock_baseline='Unsupported on these legacy scenes;retain its original cadence results separately.',
            resource_scope='Before/after GPU/load observations with no competing CUDAcompute PID recorded;continuous hardware exclusivity unproved.')
        contexts = []
        for case in cases:
            dims = dict(zip(('grid_x','grid_y','grid_z'),case['scene']['counts']))
            dims.update(cells_per_metre=case['scene']['cells_per_metre'],trajectory_id=case['identifier'],
                start_index=case['start'],K=k,output_dt_s=1/60,block_duration_s=k/60,split=case['split'])
            contexts.append(dict(id='case-'+digest([case['identifier'],case['start'],k])[:20],dimensions=dims,original_case=case))
        selected = [m for m in fits['models'] if m['K'] == k]
        entries = [(m['fit_item_id'],m['identifier'],m,models[m['fit_item_id']],
                    [fit_raw[(m['fit_item_id'],phase)] for phase in (0,1)],'original_fit') for m in selected]
        baseline_id = f'F64-original-N1-K{k}'
        entries.append((baseline_id,'F64 original N1',f64_models[k],
            dict(assembly_execution_id=f64_models[k]['assembly_execution_id'],
                original_parameter_inventory=f64['original_parameter_inventory'],original_fit_identity_invented=False),
            [f64_raw[(k,phase)] for phase in (0,1)],'assembly_execution'))
        model_evidence,rows,sources,subjects = {},[],[],[]
        for identity,label,model,original,raw_paths,kind in entries:
            model_evidence[identity] = dict(origin_kind=kind,original_model=original,summary=model['summary'],
                raw_process_blocks=[binding(p) for p in raw_paths],accepted_reductions=accepted)
            for phase,path in enumerate(raw_paths):
                sources.append(source(identity+f'-raw-{phase}',label+f' original CUDA phase{phase}',path.read_bytes()))
            population = {(c['identifier'],c['start']):c for c in model['pooled_rows']}
            require(len(population) == 64,'Every subject must retain all64cases')
            for index,context in enumerate(contexts):
                case = context['original_case']
                row = population[(case['identifier'],case['start'])]
                require(row['K'] == k and row['counts_xyz'] == case['scene']['counts']
                        and row['cells_per_metre'] == case['scene']['cells_per_metre'], 'Exact physical case required')
                rows.append(dict(subject_id=identity,case_id=context['id'],original_pooled_case=row,scalars=block_values(row)))
            dims = dict(inference_device='cuda',gpu_model=model['GPU'],origin_kind=kind,
                architecture=original.get('architecture','sequential_original_N1') if kind == 'original_fit' else 'sequential_original_N1',
                pressure_dtype='float64',carrier_dtype='float32',fit_or_execution_id=identity)
            display = label + (' · '+identity[4:12] if kind == 'original_fit' else '') + f' · GPU · full K{k} block'
            subjects.append(dict(id=identity,label=display,dimensions=dims,reference=dict(source_id='reduction',pointer='/models/'+identity)))
        context = dict(scope=scope,metrics=metrics,cases=contexts)
        evidence = dict(schema='rheon.matched-production-generator-GPU-partition.v1',K=k,
            models=model_evidence,rows=rows,accepted_reductions=accepted,producer=binding(Path(__file__)),
            historical_instrumented_packets_unchanged=True,new_inference=False,new_reference_generation=False)
        sources = [source('context','Exact common cases,metrics and hooks-outside-timing scope',encode(context)),
            source('reduction','All original lead metrics,samples and reduced scalars',encode(evidence)),*sources]
        observations = [dict(id='o-'+str(i*len(metrics)+j).zfill(5),subject_id=row['subject_id'],case_id=row['case_id'],
            metric_id=metric['id'],state='measured',value=row['scalars'][metric['id']],reason=None,
            dimensions=dict(samples=20,process_blocks=2),reference=dict(source_id='reduction',pointer=f'/rows/{i}/scalars/'+metric['id']))
            for i,row in enumerate(rows) for j,metric in enumerate(metrics)]
        packet = dict(format='retained_measurements_v1',schema_version=1,title=scope['label'],
            scope={name:scope[name] for name in ('id','label','population')}|dict(definition=dict(source_id='context',pointer='/scope',sha256=digest(scope))),
            dimensions=dimensions,subjects=subjects,
            cases=[dict(id=c['id'],dimensions=c['dimensions'],reference=dict(source_id='context',pointer='/cases/'+str(i))) for i,c in enumerate(contexts)],
            metrics=[{name:m[name] for name in ('id','label','units')}|dict(definition=dict(source_id='context',pointer='/metrics/'+str(i),sha256=digest(m))) for i,m in enumerate(metrics)],
            observations=observations,sources=sources)
        raw = encode(packet)
        table = MeasurementTable(raw)
        staged.append((dict(K=k,path=f'K{k}-measurements.json',sha256=table.sha256,bytes=len(raw),scope=scope['id'],
            subjects=[s['id'] for s in subjects],cases=64,measured=len(observations),raw_blocks=2*len(subjects),
            retained_source_bytes=sum(s['bytes'] for s in sources)),raw))
    parts = [p for p,_ in staged]
    declaration = dict(schema_version=1,id=digest(parts)[:32],name='35generator fits and4matched original F64 executions · GPU',
        data=[dict(id=f'K{p["K"]}-complete-GPU',label=p['scope'],path=str((OUTPUT/p['path']).relative_to(ROOT)),sha256=p['sha256']) for p in parts],
        recipes=[],evidence=[],requirements=[])
    manifest(declaration)
    require(sum(p['measured'] for p in parts) == 24960 and sum(len(p['subjects']) for p in parts) == 39
            and 'torch' not in sys.modules and 'numpy' not in sys.modules, 'All39configurations/24960scalars and metadata-only preparation required')
    OUTPUT.mkdir()
    for part,raw in staged:
        (OUTPUT/part['path']).write_bytes(raw)
    (OUTPUT/'source-project.json').write_bytes(encode(declaration))
    result = dict(status='four_matched_generator_GPU_packets_prepared_not_adopted',created_unix=time.time(),packets=parts,
        exact_fit_count=35,F64_assembly_executions=4,exact_subject_count=39,measured_scalars=24960,original_raw_blocks=78,
        accepted_reductions=accepted,producer=binding(Path(__file__)),native_table_owner=binding(NATIVE/'retained_measurements.py'),
        source_manifest=binding(OUTPUT/'source-project.json'),historical_instrumented_packets_preserved=True,
        native_project_mutations=0,new_inference=False,new_reference_generation=False,payload_copies=0,full_portfolio_complete=False)
    (OUTPUT/'preparation.json').write_bytes(encode(result))
    print('Prepared39configurations/24960native scalars in4Kscopes;',sum(p['bytes'] for p in parts),'packet bytes;no native adoption.')


if __name__ == '__main__':
    prepare()
