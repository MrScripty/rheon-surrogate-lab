"""Prepare a native comparison of five fits / six native-RHS CUDA policies.

The accepted scalar reports are embedded once as exact JSON sources. This is a
pressure-request scope, distinct from CNN pressure requests and full generators.
No native mutation, neural runtime, model payload or prediction field is used.
"""
from pathlib import Path
import struct
import sys
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require
from prepare_portfolio_measurements import source

OUTPUT = RUN/'native-RHS-component-measurements-01'
NATIVE = RUN/'sources/Tuldok-composition-source-artifacts'
METRICS = [
    ('pressure_mean_free_relative_l2', 'Pressure mean-free relative L2', 'relative L2'),
    ('common_equation_relative_l2', 'Pressure equation relative L2', 'relative L2'),
    ('pressure_absolute_rms', 'Pressure mean-free absolute RMS', 'Pa'),
    ('pressure_absolute_max', 'Pressure mean-free absolute maximum', 'Pa'),
    ('pressure_call_median_ms', 'Median complete GPU pressure request', 'GPU ms/request'),
]


def run():
    require(not OUTPUT.exists(), 'Preserve original native-RHS comparison preparation')
    paths = [DOC/'reports/modal-component-GPU-benchmark-01.json', DOC/'reports/shared-component-GPU-benchmark-02.json']
    modal, shared = [read(p) for p in paths]
    require(modal['fits'] == 4 and shared['fits'] == 1 and modal['cases_per_fit'] == shared['cases_per_fit'] == 144
            and modal['CPU_neural_forwards'] == shared['CPU_neural_forwards'] == 0
            and modal['primary_observer_removed'] and shared['primary_observer_removed']
            and shared['all_mode_and_process_outputs_exact'], 'Qualified original native-RHS campaigns required')
    entries = [(m,'modal','/models/'+str(i),p) for i,m in enumerate(modal['models']) for p in paths[:1]]
    entries += [(m,'shared','/modes/'+str(i),paths[1]) for i,m in enumerate(shared['modes'])]
    require(len(entries) == 6 and len({m['fit_item_id'] for m,_,_,_ in entries}) == 5, 'Five fit identities and six explicit policies required')
    index_path = ROOT/'projects/rheon-model-portfolio-unified-04/notes/portfolio-index.json'
    fits = {f['item_id']:f for f in read(index_path)['fits']}
    sys.path.insert(0,str(NATIVE))
    from retained_measurements import MeasurementTable, encode, digest
    from projects import manifest
    scope = dict(id='old-native-RHS144-CUDA-pressure-components-v1',
        label='Native RHS pressure · five fits/six policies · GPU · 144 cases', population='consumed',
        task='Native RHS and geometry to one pressure field; no complete physical simulation output.',
        comparison='Same input RHS, native reference pressure, geometry and source-state identities; different from the common128 CNN request boundary.',
        exposure='Original native fitting states for modal fits; posthoc legacy diagnostic for the expanded shared fit. Neither is an independent holdout.',
        timing='Complete original native CPU RHS request through CUDA pressure prediction and owned CPU pressure output, transfers and synchronization. Twenty primary samples percase across two processes.',
        preparation='Setup/first/warmups and hook calibration excluded. Shared prepared and unprepared policies are separate subjects; no overhead subtraction.',
        provenance='Original exact reduced reports embedded; their lab source/weight/raw-process closures remain separately bound and not claimed portable here.',
        device='NVIDIA GeForce RTX 5090 Laptop GPU', physical_simulation_outputs=0,
        resource_scope='Original before/after resource snapshots; no continuous exclusivity or hard realtime guarantee.')
    metric_definitions = [dict(id=n,label=label,units=units,
        formula=('Original pooled_median_ms over twenty complete request samples; not a simulation-step time.' if n == 'pressure_call_median_ms'
                 else 'Original independently reduced per-request '+n+'; original gauge, target norm and denominator-floor policies remain bound in the embedded source.'),
        original_metric_field='pooled_median_ms' if n == 'pressure_call_median_ms' else n) for n,label,units in METRICS]
    columns = [('grid_x','Grid X','integer','cells','case'),('grid_y','Grid Y','integer','cells','case'),
        ('grid_z','Grid Z','integer','cells','case'),('cells_per_metre','Grid density','number','cells/m','case'),
        ('trajectory_id','Original trajectory','string',None,'case'),('state_index','Original state','integer','step','case'),
        ('fit_item_id','Recorded fit identity','string',None,'subject'),('execution_policy','Captured execution policy','string',None,'subject'),
        ('inference_device','Inference device','string',None,'subject'),('samples','Timing samples','integer','samples','observation')]
    dimensions = [dict(zip(('id','label','type','units','scope'),row)) for row in columns]
    baseline = modal['models'][0]['rows']; contexts = []
    for i,row in enumerate(baseline):
        require(row['case_index'] == i, 'Ordered original case indices required')
        identity = row['identity']; z,y,x = identity['counts_zyx']
        contexts.append(dict(id='native-'+str(i).zfill(3),
            dimensions=dict(grid_x=x,grid_y=y,grid_z=z,cells_per_metre=identity['cells_per_metre'],trajectory_id=identity['identifier'],state_index=identity['state']),
            identity=identity,input_rhs_sha256=row['input_rhs_sha256'],reference_pressure_sha256=row['reference_pressure_sha256']))
    subjects, observations = [], []
    for model,sid,pointer,path in entries:
        fid = model['fit_item_id']; policy = model.get('mode','modal_original_gain_cache'); subject_id = fid+'-'+policy
        subjects.append(dict(id=subject_id,label=fits[fid]['name']+' · '+fid[4:12]+' · '+policy+' · GPU pressure',
            dimensions=dict(fit_item_id=fid,execution_policy=policy,inference_device='cuda'),reference=dict(source_id=sid,pointer=pointer)))
        require(len(model['rows']) == 144, 'Complete144case matrix required')
        for i,row in enumerate(model['rows']):
            case = baseline[i]
            require(row['case_index'] == i and all(row[n] == case[n] for n in ('identity','input_rhs_sha256','reference_pressure_sha256')),
                    'Different actual RHS/reference/context cannot enter the matched table')
            require(len(row['raw_wall_nanoseconds']) == 20, 'Exactly twenty original timing samples required')
            for name,_,_ in METRICS:
                tail = '/pooled_median_ms' if name == 'pressure_call_median_ms' else '/metrics/'+name
                value = row['pooled_median_ms'] if name == 'pressure_call_median_ms' else row['metrics'][name]
                observations.append(dict(id='o-'+str(len(observations)).zfill(5), subject_id=subject_id,case_id=contexts[i]['id'],metric_id=name,
                    state='measured',value=value,reason=None,dimensions=dict(samples=20),
                    reference=dict(source_id=sid,pointer=pointer+'/rows/'+str(i)+tail)))
    context = dict(scope=scope, metrics=metric_definitions, cases=contexts,accepted_sources=[binding(p) for p in paths],producer=binding(Path(__file__)))
    sources = [source('context','Native-RHS comparison contract and exact case identities',encode(context)),
               source('modal','Original four-modal-fit CUDA scalar reduction',paths[0].read_bytes()),
               source('shared','Original shared-pressure CUDA scalar reduction, both cache policies',paths[1].read_bytes())]
    packet = dict(format='retained_measurements_v1',schema_version=1,title=scope['label'],
        scope={n:scope[n] for n in ('id','label','population')}|dict(definition=dict(source_id='context',pointer='/scope',sha256=digest(scope))),
        dimensions=dimensions,subjects=subjects,
        cases=[dict(id=c['id'],dimensions=c['dimensions'],reference=dict(source_id='context',pointer='/cases/'+str(i))) for i,c in enumerate(contexts)],
        metrics=[{n:m[n] for n in ('id','label','units')}|dict(definition=dict(source_id='context',pointer='/metrics/'+str(i),sha256=digest(m))) for i,m in enumerate(metric_definitions)],
        observations=observations,sources=sources)
    raw = encode(packet); table = MeasurementTable(raw)
    # Independent pointer traversal also checks binary64 values and full membership.
    docs = dict(context=context,modal=modal,shared=shared)
    for obs in observations:
        target = docs[obs['reference']['source_id']]
        for part in obs['reference']['pointer'].split('/')[1:]:
            target = target[int(part)] if isinstance(target,list) else target[part]
        require(struct.pack('!d',target) == struct.pack('!d',obs['value']), 'Original source scalar bits differ')
    require(len(observations) == 4320 and 'torch' not in sys.modules and 'numpy' not in sys.modules, 'Metadata-only complete six-policy comparison required')
    part = dict(path='native-RHS-measurements.json',sha256=table.sha256,bytes=len(raw),scope=scope['id'],subjects=[s['id'] for s in subjects],cases=144,measured=4320,
                retained_source_bytes=sum(s['bytes'] for s in sources))
    declaration = dict(schema_version=1,id=digest(part)[:32],name=scope['label'],data=[dict(id='native-RHS-CUDA',label=scope['label'],
        path=str((OUTPUT/part['path']).relative_to(ROOT)),sha256=part['sha256'])],recipes=[],evidence=[],requirements=[])
    manifest(declaration); OUTPUT.mkdir(); (OUTPUT/part['path']).write_bytes(raw); (OUTPUT/'source-project.json').write_bytes(encode(declaration))
    save(OUTPUT/'preparation.json',dict(status='native_RHS_five_fits_six_policies_4320scalars_prepared_not_imported',created_unix=time.time(),packets=[part],
        exact_fit_count=5,execution_policies=6,measured_scalars=4320,source_pointer_binary64_checks=4320,
        accepted_reductions=[binding(p) for p in paths],source_manifest=binding(OUTPUT/'source-project.json'),producer=binding(Path(__file__)),
        native_table_owner=binding(NATIVE/'retained_measurements.py'),native_project_mutations=0,physical_simulation_outputs=0,
        new_inference=False,model_or_dataset_copies=0,full_portfolio_complete=False))
    print('Prepared five fits/six explicit policies/4320native scalars;',len(raw),'packet bytes;no native import.',flush=True)


if __name__ == '__main__':
    run()
