"""Verify every native block scalar against independent reduction and raw calls.

No model, numerical runtime, live project or SQL is opened. All packet sources
are checked, accuracy maxima are recomputed over every K lead, and medians and
maxima are recomputed from both original process blocks. P95 is copied exactly
from the independently tested original raw-sample reduction.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/model-portfolio-migration-v1/sources/Tuldok-composition-source-artifacts'
REPORT=ROOT/'docs/plans/model-portfolio-migration-v1/reports/frozen-full-state-GPU-timing-01.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def same(a,b):
    return struct.pack('!d',a)==struct.pack('!d',b)


def identity(row):
    return (row['identifier'],row['start_index'],row['output_steps_per_s'],row['K'],row['schedule'])


def qualify(output,report_path):
    if report_path.exists():
        raise ValueError('Preserve the original independent packet qualification')
    prepared=read(output/'preparation.json');independent=read(REPORT)
    if (prepared['independent_report_sha256']!=sha(REPORT)
            or prepared['source_manifest_sha256']!=sha(output/'source-project.json')
            or independent['status']!='all_original_full_state_GPU_samples_independently_bound_and_reduced'):
        raise ValueError('Original independent reduction or source manifest changed')
    sys.path.insert(0,str(SOURCE))
    from retained_measurements import MeasurementTable,MeasurementComparison
    tables=[];queries=[];total=0;raw_source_ids=set()
    definitions={
        'pressure_relative_rms-block-worst':('accuracy','pressure'),
        'velocity_relative_rms-block-worst':('accuracy','velocity'),
        'tracer_relative_rms-block-worst':('accuracy','tracer'),
        'pressure_equation_relative_l2-block-worst':('accuracy','equation'),
        'complete_block_median_ms':('timing','block_median'),
        'per_output_median_ms':('timing','output_median'),
        'per_output_p95_ms':('timing','output_p95'),
        'per_output_maximum_ms':('timing','output_maximum')}
    for part in prepared['packets']:
        path=output/part['path'];raw=path.read_bytes()
        if sha(path)!=part['sha256']:
            raise ValueError('Prepared packet changed')
        doc=json.loads(raw);table=MeasurementTable(raw)
        sources={s['id']:s for s in doc['sources']}
        decoded={i:json.loads(base64.b64decode(s['data_base64'])) for i,s in sources.items()}
        context=decoded['context'];evidence=decoded['reduction'];rows=evidence['rows']
        subject_map={s['id']:s for s in doc['subjects']}
        executions={s['dimensions']['execution_variant'] for s in doc['subjects']}
        wanted=[row for row in independent['rows'] if row['execution'] in executions]
        if (rows!=wanted or evidence['original_independent_report']['sha256']!=sha(REPORT)
                or context['scope']['id']!=part['scope'] or doc['scope']['id']!=part['scope']
                or len(doc['cases'])!=part['cases'] or len(doc['subjects'])!=(2 if part['family']=='cell' else 6)
                or {m['id'] for m in doc['metrics']}!=set(definitions)):
            raise ValueError('Packet family, exact partition or metric scope differs')
        original_blocks=[]
        for number,original in enumerate(evidence['raw_blocks'],1):
            source=sources['raw-'+str(number)];actual=ROOT/original['path'];payload=base64.b64decode(source['data_base64'])
            if (sha(actual)!=source['sha256'] or source['sha256']!=original['sha256']
                    or actual.read_bytes()!=payload or original['sha256'] in raw_source_ids):
                raise ValueError('Original process block lost, duplicated across families or altered')
            raw_source_ids.add(original['sha256']);original_blocks.append(decoded['raw-'+str(number)])
        raw_rows={}
        for block in original_blocks:
            for row in block['rows']:
                raw_rows.setdefault((block['execution'],identity(row)),[]).extend(row['block_wall_ms'])
        case_map={c['id']:c['dimensions'] for c in doc['cases']}
        values={};row_map={(r['execution'],identity(r)):r for r in rows}
        for observation in doc['observations']:
            case=case_map[observation['case_id']]
            subject=subject_map[observation['subject_id']]
            schedule=subject['dimensions']['integration_schedule'];execution=subject['dimensions']['execution_variant']
            selected=(case['trajectory_id'],case['start_index'],case['output_steps_per_s'],case['K'],schedule)
            row=row_map[(execution,selected)];n=rows.index(row)
            if ([case[k] for k in ('grid_x','grid_y','grid_z')]!=row['counts_xyz']
                    or case['cells_per_metre']!=row['cells_per_metre']
                    or case['reference_updates_output']!=(row['actual_internal_advances']//row['K'] if part['family']=='cell'
                                                           else 240//row['output_steps_per_s'])
                    or case['start_index_rate_per_s']!=row['start_index_rate_per_s']
                    or not same(case['output_dt_s'],1/row['output_steps_per_s'])
                    or not same(case['block_duration_s'],row['K']/row['output_steps_per_s'])
                    or observation['dimensions']!=dict(samples=20,balanced_blocks=2,diagnostic_pass=row['original_accuracy_pass'],
                        advances_block=row['actual_internal_advances'],advances_output=row['actual_internal_advances']//row['K'])
                    or observation['state']!='measured' or observation['reason'] is not None):
                raise ValueError('Physical case/work/sample or measured-state meaning differs')
            kind,name=definitions[observation['metric_id']]
            samples=raw_rows[(execution,identity(row))]
            if len(samples)!=20:
                raise ValueError('Two original ten-call blocks required')
            if kind=='accuracy':
                expected=max(m['relative'][name] for m in row['original_metrics'])
                suffix='original_worst_relative_error_over_K/'+name
            elif name=='block_median':
                expected=statistics.median(samples);suffix='block_ms/median'
            elif name=='output_median':
                expected=statistics.median(samples)/row['K'];suffix='ms_per_physical_output/median'
            elif name=='output_maximum':
                expected=max(samples)/row['K'];suffix='ms_per_physical_output/maximum'
            else:
                expected=row['ms_per_physical_output']['p95'];suffix='ms_per_physical_output/p95'
            if (not same(observation['value'],expected) or observation['reference']!=dict(
                    source_id='reduction',pointer='/rows/'+str(n)+'/'+suffix)):
                raise ValueError('Packaged scalar or exact provenance pointer differs')
            filter_dimensions=dict(case,**subject['dimensions'],**observation['dimensions'])
            values.setdefault((observation['subject_id'],observation['metric_id']),[]).append((filter_dimensions,expected))
            total+=1
        selections=[{},dict(K=4),dict(cells_per_metre=100 if part['family']=='cell' else 50)]
        selections += ([dict(grid_x=64),dict(reference_updates_output=8)] if part['family']=='cell'
                       else [dict(output_steps_per_s=r) for r in (30,60,120)]+[dict(integration_schedule='fine_reference')])
        for filters in selections:
            for metric in definitions:
                result=table.query(metric,filters=filters,limit=1)
                for summary in result['summary']:
                    wanted=[v for case,v in values[(summary['subject_id'],metric)]
                            if all(case.get(k)==v for k,v in filters.items())]
                    maximum=summary['maximum']
                    if (summary['states']['measured']!=len(wanted)
                            or (wanted and (maximum is None or not same(maximum['value'],max(wanted))))
                            or (not wanted and maximum is not None)):
                        raise ValueError('Native case filter, full denominator or maximum differs')
                queries.append(dict(scope=part['scope'],metric=metric,filters=filters,summary=result['summary']))
        tables.append((dict(origin_catalog_id='a'*32,packet_sha256=table.sha256),table))
    if total!=9600 or len(raw_source_ids)!=8:
        raise ValueError('All9600scalars and eight distinct original process blocks required')
    try:
        MeasurementComparison(tables)
    except Exception as error:
        if getattr(error,'code',None)!='incompatible':
            raise
    else:
        raise ValueError('Scientifically different full-state populations were incorrectly pooled')
    if 'torch' in sys.modules or 'numpy' in sys.modules:
        raise ValueError('Packet qualification imported a model or numerical runtime')
    result=dict(status='both_full_state_GPU_timing_packets_every_scalar_and_native_filter_qualified',
        created_unix=time.time(),preparation_sha256=sha(output/'preparation.json'),
        producer_sha256=sha(Path(__file__)),native_table_source_sha256=sha(SOURCE/'retained_measurements.py'),
        exact_measured_scalars=total,original_raw_blocks_retained_once=8,
        all_K_lead_accuracy_maxima_recomputed=True,pooled_medians_and_maxima_recomputed=True,
        p95_bit_exact_to_independent_raw_reduction=True,all_case_density_grid_cadence_K_work_bindings_exact=True,
        native_filter_queries=len(queries),queries=queries,incompatible_population_mix_rejected=True,
        new_neural_forwards=0,native_project_mutated=False,native_ingestion_and_UI_qualified=False,
        full_portfolio_complete=False)
    report_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','exact_measured_scalars','native_filter_queries')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    qualify(args.output.resolve(),args.report.resolve())
