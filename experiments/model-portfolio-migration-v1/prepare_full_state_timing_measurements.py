"""Package fresh full-state block timings as two incompatible native scopes.

Each packet keeps its own original four raw timing blocks and only its partition
of the independent reduction. No full report is duplicated between packets. A
case represents a whole K-output block; numerical error is the worst of every
intermediate output, and timing is the complete block divided only by K.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

from summarize_full_state_gpu_timing import ROOT, read, sha
from prepare_portfolio_measurements import source

BASE=ROOT/'runs/model-portfolio-migration-v1'
DOC=ROOT/'docs/plans/model-portfolio-migration-v1'
SOURCE=BASE/'sources/Tuldok-composition-source-artifacts'
REPORT=DOC/'reports/frozen-full-state-GPU-timing-01.json'
PARTS=dict(cell=('cell_bound','cell_coordinate'),temporal=('temporal_eager','temporal_graph'))


def prepare(output):
    if output.exists() or not output.is_relative_to(BASE):
        raise ValueError('Fresh task-owned packet output required')
    report=read(REPORT)
    if report['status']!='all_original_full_state_GPU_samples_independently_bound_and_reduced':
        raise ValueError('Complete independently bound raw sample reduction required')
    sys.path.insert(0,str(SOURCE))
    from retained_measurements import MeasurementTable,encode,digest
    from projects import manifest as validate_manifest
    from prepare_full_state_measurements import definitions
    normalization=DOC.parent/'full-ai-tight-targets-v1/sampler-accuracy-normalization-01.json'
    scorer=ROOT/'experiments/full-ai-tight-targets-v1/diagnose_causal_composition.py'
    metric_definitions=[]
    for original in definitions(read(normalization),sha(normalization),sha(scorer))[:4]:
        item=copy.deepcopy(original);item['id']+='-block-worst'
        item['label']='Worst in K-output block · '+original['label']
        item.update(aggregation='Maximum of this original scalar over every physical output lead1..K.',
            accuracy_evidence='Exact old frozen outputs recaptured in every timed/warm call; independent literal metric/case binding.',
            new_independent_physics_scoring=False)
        metric_definitions.append(item)
    for identity,label,units,field in (
        ('complete_block_median_ms','Median complete K-output GPU block','ms/block','block_ms/median'),
        ('per_output_median_ms','Median complete GPU time per physical output','ms/output','ms_per_physical_output/median'),
        ('per_output_p95_ms','P95 complete GPU time per physical output','ms/output','ms_per_physical_output/p95'),
        ('per_output_maximum_ms','Maximum observed GPU time per physical output','ms/output','ms_per_physical_output/maximum')):
        metric_definitions.append(dict(id=identity,label=label,units=units,kind='fresh_complete_GPU_timing',
            formula='Statistic of20complete block wall samples in two balanced process blocks; divide byKonly for per-output metrics.',
            field=field,timing_scope=report['timing_scope'],finite_observation_not_hard_real_time=True))
    columns=[('grid_x','Grid X','integer','cells','case'),('grid_y','Grid Y','integer','cells','case'),
        ('grid_z','Grid Z','integer','cells','case'),('cells_per_metre','Grid density','number','cells/m','case'),
        ('trajectory_id','Original physical trajectory','string',None,'case'),('split','Original split','string',None,'case'),
        ('K','Physical outputs per block','integer','outputs','case'),('output_steps_per_s','Simulation output cadence','number','outputs/s','case'),
        ('output_dt_s','Physical duration per output','number','s','case'),('block_duration_s','Physical duration per block','number','s','case'),
        ('start_index','Original start index','integer','index','case'),('start_index_rate_per_s','Start-index clock','number','indices/s','case'),
        ('reference_updates_output','Frozen reference updates per output','integer','updates/output','case'),
        ('inference_device','Inference device','string',None,'subject'),('gpu_model','GPU','string',None,'subject'),
        ('execution_variant','Captured execution','string',None,'subject'),
        ('integration_schedule','Captured integration policy','string',None,'subject'),
        ('host_logical_cpu','Host logical CPU','integer',None,'subject'),
        ('host_threads','Host threads','integer',None,'subject'),('pressure_dtype','Pressure dtype','string',None,'subject'),
        ('carrier_dtype','Carrier dtype','string',None,'subject'),('samples','Samples per block case','integer','samples','observation'),
        ('balanced_blocks','Balanced process blocks','integer','blocks','observation'),
        ('advances_block','Actual internal advances in block','integer','advances','observation'),
        ('advances_output','Actual internal advances per output','integer','advances/output','observation'),
        ('diagnostic_pass','Original all-output accuracy diagnostic','boolean',None,'observation')]
    dimensions=[dict(id=i,label=l,type=t,units=u,scope=s) for i,l,t,u,s in columns]
    output.mkdir();packets=[]
    for family,executions in PARTS.items():
        rows=[row for row in report['rows'] if row['execution'] in executions]
        raw_blocks=[entry for entry in report['worker_results'] if entry['execution'] in executions]
        models={}
        sources=[]
        for number,entry in enumerate(raw_blocks):
            path=ROOT/entry['path']
            if sha(path)!=entry['sha256']:
                raise ValueError('Original timing block changed')
            value=read(path);identity=value['execution']
            if identity not in models:
                models[identity]={key:value[key] for key in ('model_identity','parameter_inventory','source_module',
                    'GPU_model','torch_version','CUDA_version','host_affinity','torch_threads','interop_threads')}
                models[identity]['is_new_parameter_fit']=False
            sources.append(source('raw-'+str(number+1),'Original '+identity+' phase'+str(entry['phase']),path.read_bytes()))
        evidence=dict(schema='rheon.full-state-timing-family-partition.v1',family=family,
            original_independent_report=dict(path=str(REPORT.relative_to(ROOT)),sha256=sha(REPORT)),
            original_protocol=dict(path=report['protocol_path'],sha256=report['protocol_sha256']),
            independent_reducer_sha256=report['producer_sha256'],models=models,rows=rows,
            raw_blocks=raw_blocks,all_original_output_hashes_exact=True,
            independent_physics_rescore=False,own_history_drift_qualified=False)
        contexts=[];case_index={}
        subject_metadata={}
        def physical_binding(row):
            binding={key:row[key] for key in ('identifier','counts_xyz','cells_per_metre','start_index',
                'start_index_rate_per_s','output_steps_per_s','K',
                'current_state_path','current_state_sha256','input_field_sha256')}
            binding['reference_updates_output']=(row['actual_internal_advances']//row['K'] if family=='cell'
                                                else 240//row['output_steps_per_s'])
            return binding
        def subject_id(row):
            return row['execution'] if family=='cell' else row['execution']+'.'+row['schedule']
        for row in rows:
            binding=physical_binding(row)
            identity='b-'+digest(binding)[:20]
            subject_metadata.setdefault(subject_id(row),dict(execution=row['execution'],
                integration_schedule=row['schedule'],model_identity=models[row['execution']]['model_identity'],
                is_new_parameter_fit=False))
            if identity not in case_index:
                dims=dict(zip(('grid_x','grid_y','grid_z'),row['counts_xyz']))
                dims.update(cells_per_metre=row['cells_per_metre'],trajectory_id=row['identifier'],split=row['split'],
                    K=row['K'],output_steps_per_s=row['output_steps_per_s'],output_dt_s=1/row['output_steps_per_s'],
                    block_duration_s=row['K']/row['output_steps_per_s'],start_index=row['start_index'],
                    start_index_rate_per_s=row['start_index_rate_per_s'],reference_updates_output=binding['reference_updates_output'])
                case_index[identity]=len(contexts);contexts.append(dict(id=identity,dimensions=dims,binding=binding))
        if len(contexts)!=(384 if family=='cell' else 72):
            raise ValueError('Complete family case population required')
        evidence['subjects']=subject_metadata
        label=('Cell sampler' if family=='cell' else 'Temporal sampler')+' · complete GPU blocks · frozen old physics'
        scope=dict(id='old-'+family+'-complete-NumPy-GPU-blocks-v1',label=label,population='consumed',
            task='Complete five-field generator, CPU NumPy input to allKownedCPUstates. Numerical metrics take the worst over allKleads.',
            comparison='Only matched grid/density/cadence/K/start/current-state/reference cases within this family. Integration policies are separate subjects and executed work belongs to observations.',
            timing_scope=report['timing_scope'],timing_samples=20,reference_solver_epoch='frozen_old_Rheon',
            K_does_not_mean_reduced_work=True,source_reference_cadence='Each case preserves its original reference/integration policy.',
            independence='Repeated windows/cadences/K do not create independent scenes; cell24trajectory variants share4scenario groups; both temporal masters are labeled train in the original captures and benchmark rows. The cpm50 trajectory is not a held-out development scene.',
            policy_equivalence=('At120outputs/s, half_passes and single_pass both execute one width2group per output. Named policies remain identifiable but are not distinct learned fits.' if family=='temporal'
                                else 'Both backends retain the same fitted coefficients and exact outputs on this consumed population; coordinate changes apply only to the N8 path.'))
        context=dict(scope=scope,metrics=metric_definitions,cases=contexts)
        sources=[source('context','Exact case/metric/scope definitions',encode(context)),
                 source('reduction','Independently verified '+family+' partition',encode(evidence)),*sources]
        metrics=[dict(id=m['id'],label=m['label'],units=m['units'],definition=dict(
            source_id='context',pointer='/metrics/'+str(i),sha256=digest(m))) for i,m in enumerate(metric_definitions)]
        subjects=[]
        for identity,metadata in sorted(subject_metadata.items()):
            execution=metadata['execution']
            subjects.append(dict(id=identity,label=identity.replace('_',' ')+' · GPU complete call',
                reference=dict(source_id='reduction',pointer='/subjects/'+identity),
                dimensions=dict(inference_device='cuda',gpu_model=models[execution]['GPU_model'],execution_variant=execution,
                    integration_schedule=metadata['integration_schedule'],host_logical_cpu=4,host_threads=1,
                    pressure_dtype='float64',carrier_dtype='float32')))
        cases=[dict(id=c['id'],dimensions=c['dimensions'],reference=dict(source_id='context',pointer='/cases/'+str(i)))
               for i,c in enumerate(contexts)]
        observations=[]
        for index,row in enumerate(rows):
            binding=physical_binding(row)
            case_id='b-'+digest(binding)[:20]
            for number,metric in enumerate(metric_definitions):
                if number<4:
                    field=('pressure','velocity','tracer','equation')[number]
                    value=row['original_worst_relative_error_over_K'][field]
                    suffix='original_worst_relative_error_over_K/'+field
                else:
                    suffix=metric['field'];value=row
                    for part in suffix.split('/'):
                        value=value[part]
                observations.append(dict(id='o-'+str(len(observations)).zfill(6),subject_id=subject_id(row),
                    case_id=case_id,metric_id=metric['id'],state='measured',value=value,reason=None,
                    dimensions=dict(samples=20,balanced_blocks=2,diagnostic_pass=row['original_accuracy_pass'],
                        advances_block=row['actual_internal_advances'],advances_output=row['actual_internal_advances']//row['K']),
                    reference=dict(source_id='reduction',pointer='/rows/'+str(index)+'/'+suffix)))
        packet=dict(format='retained_measurements_v1',schema_version=1,title=label,
            scope={key:scope[key] for key in ('id','label','population')}|dict(definition=dict(
                source_id='context',pointer='/scope',sha256=digest(scope))),dimensions=dimensions,
            subjects=subjects,cases=cases,metrics=metrics,observations=observations,sources=sources)
        raw=encode(packet);table=MeasurementTable(raw)
        if table.metadata()['observation_count']!=len(rows)*8:
            raise ValueError('Native complete scalar matrix differs')
        path=output/(family+'-measurements.json');path.write_bytes(raw)
        packets.append(dict(family=family,path=path.name,sha256=table.sha256,bytes=len(raw),
            scope=scope['id'],subjects=[s['id'] for s in subjects],cases=len(cases),measured=len(observations),
            retained_source_bytes=sum(s['bytes'] for s in sources)))
    declaration=dict(schema_version=1,id=digest(packets)[:32],name='Frozen full-state GPU timing · common complete-call boundary',
        data=[dict(id=p['family']+'-GPU-timing',label=p['scope'],path=str((output/p['path']).relative_to(ROOT)),
                   sha256=p['sha256']) for p in packets],recipes=[],evidence=[],requirements=[])
    validate_manifest(declaration);(output/'source-project.json').write_bytes(encode(declaration))
    prepared=dict(status='two_full_state_GPU_timing_packets_prepared_not_imported',created_unix=time.time(),
        packets=packets,source_manifest_sha256=sha(output/'source-project.json'),
        independent_report_sha256=sha(REPORT),producer_sha256=sha(Path(__file__)),
        measured_scalars=sum(p['measured'] for p in packets),subject_executions=8,
        captured_backend_executions=4,
        raw_timed_calls=24000,timed_physical_outputs=60000,new_parameter_fits=0,
        new_model_forwards=0,new_reference_generation=False,native_project_mutated=False,
        native_ingestion_and_UI_qualified=False,full_portfolio_complete=False)
    (output/'preparation.json').write_bytes(encode(prepared))
    if 'torch' in sys.modules or 'numpy' in sys.modules:
        raise ValueError('Packaging must not import a numerical/model runtime')
    print(json.dumps(dict(status=prepared['status'],packets=packets,measured_scalars=prepared['measured_scalars'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output.resolve())
