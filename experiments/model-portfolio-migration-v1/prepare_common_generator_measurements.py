"""Package ten full-state GPU generators in four explicit physical-K scopes.

Retain original raw process blocks once and every individual lead metric. This
packages accepted measurements only; it imports no neural or numerical runtime.
Native adoption belongs to a later serialized owner, not this preparer.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

from prepare_portfolio_measurements import source

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
NATIVE = RUN/'sources/Tuldok-composition-source-artifacts'
OUTPUT = RUN/'common-generator-GPU-measurements-01'
REPORTS = [DOC/'reports'/name for name in
    ('joint-generator-GPU-benchmark-01.json','sharing-generator-GPU-benchmark-01.json')]
ACCURACY = ('pressure_relative_l2','velocity_relative_l2','tracer_relative_l2',
            'equation_relative_l2','pressure_absolute_rms','tracer_absolute_rms')
TIMING = ('block_median_ms','per_output_median_ms','per_output_p95_ms','per_output_maximum_ms')


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition,message):
    if not condition:
        raise ValueError(message)


def block_values(case):
    """Conservative block maxima require every requested physical lead."""
    k = case['K']
    require(len(case['metrics']) == k and [r['lead'] for r in case['metrics']] == list(range(1,k+1)),
            'Every ordered physical lead required')
    result = {}
    for name in ACCURACY:
        values = [r['values'][name] for r in case['metrics']]
        require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in values),
                'Undefined or invalid lead metric cannot become a measured block maximum')
        result[name] = max(values)
    blocks = case['block_ms']
    require(len(blocks) == 20 and all(math.isfinite(v) and v > 0 for v in blocks),
            'Exactly twenty positive original block samples required')
    outputs = case['per_output_ms']
    require(outputs == [v/k for v in blocks], 'Only physical K may divide block time')
    ordered = sorted(outputs)
    rank = .95*(len(ordered)-1)
    lower,upper = math.floor(rank),math.ceil(rank)
    result.update(block_median_ms=statistics.median(blocks),per_output_median_ms=statistics.median(outputs),
        per_output_p95_ms=ordered[lower]+(ordered[upper]-ordered[lower])*(rank-lower),
        per_output_maximum_ms=max(outputs))
    return result


def prepare():
    require(not OUTPUT.exists(),'Preserve original common-generator measurement preparation')
    reports = [read(p) for p in REPORTS]
    require([r['models_measured'] for r in reports] == [4,6]
            and all(r['cases'] == 64 and r['timing_qualified_at_observed_resource_scope']
                and r['all_phase_output_hashes_equal'] and r['CPU_neural_forwards'] == 0 for r in reports),
            'Both complete independently reduced GPU campaigns required')
    cases_paths = [DOC/'reports/joint-generator-case-preflight-01.json',RUN/'sharing-generator-GPU-cases-01.json']
    prepared = [read(p) for p in cases_paths]
    require(prepared[0]['cases'] == prepared[1]['cases'] and len(prepared[0]['cases']) == 64
            and reports[0]['reference_method'] == reports[1]['reference_method'] == prepared[0]['reference_method']
            and reports[0]['reference_native_probe_binary_sha256'] == reports[1]['reference_native_probe_binary_sha256'],
            'Both campaigns must use the same actual physical cases and old reference')
    population = prepared[0]['cases']
    cases_by_key = {(c['identifier'],c['start']):c for c in population}
    require(len(cases_by_key) == 64,'Exactly 64 unique scene/start cases required')
    combined = {}
    captured_models = {}
    raw_blocks = {}
    all_bindings = []
    for report,path,captured in zip(reports,REPORTS,prepared,strict=True):
        all_bindings.append(binding(path))
        for spec in report['evidence']:
            require(binding(ROOT/spec['path']) == spec,'Accepted benchmark evidence changed')
        original = {m['identifier']:m for m in captured['models']}
        for name,model in report['models'].items():
            require(name not in combined and model['fit_item_id'] == original[name]['fit_item_id'],
                    'One original identity per measured model required')
            combined[name] = model
            captured_models[name] = original[name]
            raw_blocks[name] = []
            for phase in (0,1):
                matches = [ROOT/s['path'] for s in report['evidence'] if Path(s['path']).name == name+f'-phase{phase}.json']
                require(len(matches) == 1,'Each model requires both original raw process blocks')
                raw_blocks[name].append(matches[0])
    require(len(combined) == 10,'All ten measured generators required')
    sys.path.insert(0,str(NATIVE))
    from retained_measurements import MeasurementTable,encode,digest
    from projects import manifest
    metrics = []
    labels = dict(pressure_relative_l2='Pressure mean-free relative L2',velocity_relative_l2='Combined velocity relative L2',
        tracer_relative_l2='Tracer relative L2',equation_relative_l2='Pressure equation relative L2',
        pressure_absolute_rms='Pressure mean-free absolute RMS',tracer_absolute_rms='Tracer absolute RMS')
    for name in ACCURACY:
        metrics.append(dict(id=name,label='Worst in K-output block · '+labels[name],
            units=('Pa' if name == 'pressure_absolute_rms' else 'tracer units' if name == 'tracer_absolute_rms' else 'relative L2'),
            formula='Maximum of the original independently scored scalar over every physical lead 1..K.',
            field=name,zero_denominator='Nonzero difference with zero truth norm is undefined and cannot be measured as zero.',
            normalization='Pressure removes spatial means; velocity concatenates all three staggered components; tracer uses its full native field.',
            scoring_source=binding(ROOT/'experiments/model-portfolio-migration-v1/joint_generator_gpu_benchmark.py')))
    for name,label,units in [('block_median_ms','Median full GPU block','ms/block'),
            ('per_output_median_ms','Median GPU time per physical output','ms/output'),
            ('per_output_p95_ms','P95 GPU time per physical output','ms/output'),
            ('per_output_maximum_ms','Maximum GPU time per physical output','ms/output')]:
        metrics.append(dict(id=name,label=label,units=units,
            formula='Statistic over twenty original complete-call wall samples, ten per process; per-output samples are block/K. P95 uses linear interpolation at .95*(N-1).',
            timing_scope='Native CPU pressure/velocities/tracer through validation, packing, known forcing, H2D, CUDA forward, reconstruction, allocation, all K owned CPU outputs/D2H and sync.',
            exclusions='Model loading, constructor, first call, warmups, hashes, scoring and ownership checks are separate.',
            finite_observation_not_hard_realtime=True,continuous_hardware_exclusivity_proven=False))
    columns = [
        ('grid_x','Grid X','integer','cells','case'),('grid_y','Grid Y','integer','cells','case'),
        ('grid_z','Grid Z','integer','cells','case'),('cells_per_metre','Grid density','number','cells/m','case'),
        ('trajectory_id','Original trajectory','string',None,'case'),('start_index','Start step','integer','step','case'),
        ('K','Physical outputs per block','integer','outputs','case'),('output_dt_s','Duration per output','number','s','case'),
        ('block_duration_s','Duration per block','number','s','case'),('split','Original split','string',None,'case'),
        ('inference_device','Inference device','string',None,'subject'),('gpu_model','GPU','string',None,'subject'),
        ('architecture','Generator architecture','string',None,'subject'),('fit_item_id','Original fit identity','string',None,'subject'),
        ('pressure_dtype','Pressure output dtype','string',None,'subject'),('carrier_dtype','Velocity/tracer dtype','string',None,'subject'),
        ('samples','Wall samples per case','integer','samples','observation'),('process_blocks','Independent process blocks','integer','blocks','observation')]
    dimensions = [dict(id=i,label=l,type=t,units=u,scope=s) for i,l,t,u,s in columns]
    OUTPUT.mkdir()
    packets = []
    for k in (1,2,3,4):
        names = sorted(n for n,m in combined.items() if m['K'] == k)
        require(len(names) == (1 if k == 1 else 3),'Exact matching-K architecture population required')
        scope = dict(id=f'old-common64-complete-generator-GPU-K{k}-v1',
            label=f'Complete GPU generators · K{k} · same 64 old-reference cases',population='consumed',
            task='All native pressure, staggered velocities and tracer at every requested physical output.',
            independence='16 original fitting scenes, four starts each; not independent holdout or own-history drift.',
            reference_method=prepared[0]['reference_method'],
            reference_binary_sha256=prepared[0]['reference_native_probe_binary_sha256'],
            timestep_s=1/60,physical_outputs=k,block_duration_s=k/60,
            comparison='Compare architectures only within the same K, grid density, domain dimensions and current-state/reference identity. Other K scopes are incompatible.',
            future_reference_fields_are_labels_only=True,
            old_near_exact_F64_anchors_not_measured_on_this_population=True,
            resource_scope='Before/after worker GPU compute observations; no continuous exclusivity guarantee.')
        contexts = []
        for case in population:
            dims = dict(zip(('grid_x','grid_y','grid_z'),case['scene']['counts']))
            dims.update(cells_per_metre=case['scene']['cells_per_metre'],trajectory_id=case['identifier'],
                start_index=case['start'],K=k,output_dt_s=1/60,block_duration_s=k/60,split=case['split'])
            contexts.append(dict(id='case-'+digest([case['identifier'],case['start'],k])[:20],
                dimensions=dims,original_case=case))
        model_evidence = {}
        rows = []
        sources = []
        for name in names:
            selected = captured_models[name]
            model_evidence[name] = dict(fit_item_id=selected['fit_item_id'],K=k,
                architecture=combined[name].get('architecture','joint'),
                original_model=selected,original_reductions=all_bindings)
            for phase,path in enumerate(raw_blocks[name]):
                raw = path.read_bytes()
                sources.append(source(name+f'-raw-{phase}',name+f' original CUDA phase {phase}',raw))
            model_cases = {(c['identifier'],c['start']):c for c in combined[name]['cases']}
            require(model_cases.keys() == cases_by_key.keys(),'Every original scene/start required')
            for index,context in enumerate(contexts):
                c = context['original_case']
                row = model_cases[(c['identifier'],c['start'])]
                require(row['K'] == k and row['counts_xyz'] == c['scene']['counts']
                        and row['cells_per_metre'] == c['scene']['cells_per_metre']
                        and row['output_dt_s'] == 1/60 and row['block_advance_s'] == k/60,
                        'Exact physical dimensions, density and horizon required')
                rows.append(dict(subject_id=name,case_id=context['id'],case_index=index,
                    original_pooled_case=row,scalars=block_values(row)))
        evidence = dict(schema='rheon.common-generator-GPU-measurement-partition.v1',K=k,
            models=model_evidence,rows=rows,raw_blocks={n:[binding(p) for p in raw_blocks[n]] for n in names},
            accepted_reductions=all_bindings,producer=binding(Path(__file__)),
            new_inference=False,new_reference_generation=False)
        context = dict(scope=scope,metrics=metrics,cases=contexts)
        sources = [source('context','Physical cases, metric definitions and comparison scope',encode(context)),
            source('reduction','Exact accepted pooled cases and every original lead scalar',encode(evidence)),*sources]
        subjects = [dict(id=n,label=n+' · GPU · full K'+str(k)+' block',
            dimensions=dict(inference_device='cuda',gpu_model='NVIDIA GeForce RTX 5090 Laptop GPU',
                architecture=model_evidence[n]['architecture'],fit_item_id=model_evidence[n]['fit_item_id'],
                pressure_dtype='float64',carrier_dtype='float32'),
            reference=dict(source_id='reduction',pointer='/models/'+n)) for n in names]
        cases = [dict(id=c['id'],dimensions=c['dimensions'],reference=dict(source_id='context',pointer='/cases/'+str(i)))
                 for i,c in enumerate(contexts)]
        observations = []
        for i,row in enumerate(rows):
            for metric in metrics:
                name = metric['id']
                observations.append(dict(id='o-'+str(len(observations)).zfill(5),subject_id=row['subject_id'],
                    case_id=row['case_id'],metric_id=name,state='measured',value=row['scalars'][name],reason=None,
                    dimensions=dict(samples=20,process_blocks=2),
                    reference=dict(source_id='reduction',pointer=f'/rows/{i}/scalars/{name}')))
        packet = dict(format='retained_measurements_v1',schema_version=1,title=scope['label'],
            scope={key:scope[key] for key in ('id','label','population')}|dict(definition=dict(
                source_id='context',pointer='/scope',sha256=digest(scope))),dimensions=dimensions,subjects=subjects,
            cases=cases,metrics=[{key:m[key] for key in ('id','label','units')}|dict(definition=dict(
                source_id='context',pointer='/metrics/'+str(i),sha256=digest(m))) for i,m in enumerate(metrics)],
            observations=observations,sources=sources)
        raw = encode(packet)
        table = MeasurementTable(raw)
        path = OUTPUT/f'K{k}-measurements.json'
        path.write_bytes(raw)
        packets.append(dict(K=k,path=path.name,sha256=table.sha256,bytes=len(raw),scope=scope['id'],
            subjects=names,cases=64,measured=len(observations),raw_blocks=2*len(names),
            retained_source_bytes=sum(s['bytes'] for s in sources)))
    declaration = dict(schema_version=1,id=digest(packets)[:32],name='Ten complete GPU generators · old common64 cases',
        data=[dict(id=f'K{p["K"]}-complete-GPU',label=p['scope'],path=str((OUTPUT/p['path']).relative_to(ROOT)),
            sha256=p['sha256']) for p in packets],recipes=[],evidence=[],requirements=[])
    manifest(declaration)
    (OUTPUT/'source-project.json').write_bytes(encode(declaration))
    result = dict(status='four_common_generator_GPU_packets_prepared_not_adopted',created_unix=time.time(),packets=packets,
        exact_model_count=10,measured_scalars=sum(p['measured'] for p in packets),original_raw_blocks=20,
        producer=binding(Path(__file__)),native_table_owner=binding(NATIVE/'retained_measurements.py'),
        source_manifest=binding(OUTPUT/'source-project.json'),accepted_reductions=all_bindings,
        original_case_selections=[binding(p) for p in cases_paths],new_inference=False,new_reference_generation=False,
        native_project_mutations=0,payload_copies=0,full_portfolio_complete=False)
    require(result['measured_scalars'] == 6400 and 'torch' not in sys.modules and 'numpy' not in sys.modules,
            'Complete 6400 scalars and metadata-only runtime required')
    (OUTPUT/'preparation.json').write_bytes(encode(result))
    print('Prepared four K-specific native GPU tables: ten models,6400 scalars,20 original raw blocks; no native adoption.')


if __name__ == '__main__':
    prepare()
