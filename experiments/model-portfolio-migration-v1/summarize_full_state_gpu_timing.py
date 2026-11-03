"""Independently bind/reduce original full-state timing samples, without a model.

Reopen each distinct frozen current-state archive and recompute its field hashes.
Join every output hash and metric to its literal original accuracy row. This is
independent timing reduction and case binding, not new independent physics or
own-history scoring. NumPy is used only for archive/finiteness verification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
FIELDS=('pressure','vx','vy','vz','tracer')
METRICS=('pressure','velocity','tracer','equation')
EXPECTED=dict(cell_bound=384,cell_coordinate=384,temporal_eager=216,temporal_graph=216)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()


def samples_summary(values):
    if (not values or any(type(n) not in (float,int) or not math.isfinite(n) or n<=0 for n in values)):
        raise ValueError('Finite positive complete-call wall samples required')
    ordered=sorted(values);rank=.95*(len(ordered)-1);lower=math.floor(rank);upper=math.ceil(rank)
    return dict(samples=len(values),minimum=ordered[0],median=statistics.median(ordered),
                p95=ordered[lower]+(ordered[upper]-ordered[lower])*(rank-lower),maximum=ordered[-1])


def key(row):
    return (row['identifier'],row['start_index'],row['output_steps_per_s'],row['K'],row['schedule'])


def reduce(protocol_path,report_path):
    if report_path.exists() or not report_path.is_relative_to(ROOT/'docs/plans/model-portfolio-migration-v1/reports'):
        raise ValueError('Fresh task-owned reduction report required')
    protocol=read(protocol_path);base=ROOT/protocol['output_root'];state=read(base/'status.json')
    if (state['status']!='complete_original_frozen_full_state_GPU_timing'
            or state['protocol_sha256']!=sha(protocol_path) or len(state['completed'])!=8
            or Path('/proc',str(state['pid'])).exists()):
        raise ValueError('Original campaign must be acknowledged complete and its owner terminal')
    for name,expected in protocol['source_sha256'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('Registered source changed: '+name)
    originals={family:read(ROOT/path) for family,path in protocol['original_protocols'].items()}
    accuracy={};arrays={};receipt_cache={};windows={}
    for spec in originals['cell']['window_sources']:
        for scene in read(ROOT/spec['controller'])['completed']:
            for window in scene['windows']:
                identity=(scene['identifier'],window['start'])
                if identity in windows:
                    raise ValueError('Duplicate original current-state window')
                windows[identity]=window
    bank={r['identifier']:r for r in read(ROOT/originals['cell']['bank'])['trajectories']}
    raw_by_execution={name:{} for name in EXPECTED}
    expected_sets={}
    cell_population={(identifier,start,60,k,'captured_native') for identifier,start in windows
                     for k in (1,2,3,4)}
    if len(cell_population)!=384 or len(bank)!=24:
        raise ValueError('Original cell population is incomplete')
    expected_sets['cell_bound']=expected_sets['cell_coordinate']=cell_population
    temporal_accuracy=read(ROOT/originals['temporal']['qualification'])['rows']
    temporal_population={(r['identifier'],r['start_fine_step'],r['output_steps_per_s'],r['K'],r['mode'])
                         for r in temporal_accuracy if r['start_fine_step'] in originals['temporal']['start_fine_steps']}
    if len(temporal_population)!=216:
        raise ValueError('Original temporal population is incomplete')
    expected_sets['temporal_eager']=expected_sets['temporal_graph']=temporal_population
    timed_calls=timed_outputs=verified_calls=verified_outputs=0
    records=[];worker_bindings=[]
    import numpy as np
    if 'torch' in sys.modules:
        raise ValueError('Independent reduction must not import a neural runtime')

    def verify_current(row):
        name=row['current_state_path'];path=ROOT/name;expected=row['current_state_sha256']
        if name not in arrays:
            if sha(path)!=expected:
                raise ValueError('Original current-state archive changed')
            with np.load(path,allow_pickle=False) as archive:
                hashes={};sizes={};types={}
                for field in FIELDS:
                    value=archive[field]
                    if value.dtype!=np.dtype('<f8' if field=='pressure' else '<f4') or not np.isfinite(value).all():
                        raise ValueError('Original current-state field type/finiteness differs')
                    hashes[field]=hashlib.sha256(value.tobytes(order='C')).hexdigest()
                    sizes[field]=value.size;types[field]=value.dtype.str
            arrays[name]=dict(sha256=expected,field_sha256=hashes,elements=sizes,dtypes=types)
        original=arrays[name]
        x,y,z=row['counts_xyz']
        needed=dict(pressure=x*y*z,vx=(x+1)*y*z,vy=x*(y+1)*z,vz=x*y*(z+1),tracer=x*y*z)
        if (original['sha256']!=expected or original['field_sha256']!=row['input_field_sha256']
                or original['elements']!=needed):
            raise ValueError('Measured complete input fields differ from frozen bytes or physical extents')

    for number,(completed,selected) in enumerate(zip(state['completed'],protocol['block_order'],strict=True),1):
        execution=selected['execution'];phase=selected['phase'];path=ROOT/completed['result']
        if (completed['block']!=number or completed['execution']!=execution or completed['phase']!=phase
                or sha(path)!=completed['sha256']):
            raise ValueError('Original process block order/result binding differs')
        result=read(path)
        if (result['status']!='complete_original_full_state_GPU_timing' or not result['cleanup_complete']
                or result['execution']!=execution or result['phase']!=phase
                or result['protocol_sha256']!=sha(protocol_path)
                or result['device_qualification']['forbidden_module_calls']
                or result['CPU_model_forwards'] or result['optimizer_updates']
                or not result['parameter_inventory_unchanged'] or not result['source_bindings_unchanged']
                or any(r['other_compute_pids'] for r in result['resource_observations'])
                or result['host_affinity']!=[protocol['host_cpu']]
                or result['torch_threads']!=1 or result['interop_threads']!=1
                or len(result['rows'])!=EXPECTED[execution]):
            raise ValueError('Worker completion, precision/source/device/resource or population contract differs')
        worker_bindings.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),execution=execution,
            phase=phase,GPU_model=result['GPU_model'],torch_version=result['torch_version'],
            CUDA_version=result['CUDA_version'],device_qualification=result['device_qualification'],
            peak_RSS_bytes=result['peak_RSS_bytes'],peak_Torch_GPU_allocated_bytes=result['peak_Torch_GPU_allocated_bytes']))
        seen=set()
        for row in result['rows']:
            identity=key(row)
            if identity in seen or identity not in expected_sets[execution]:
                raise ValueError('Duplicate or unregistered physical timing case')
            seen.add(identity)
            verify_current(row)
            cell=execution.startswith('cell_')
            source=(originals['cell']['baseline_outputs'] if execution=='cell_bound'
                    else originals['cell']['candidate_outputs']) if cell else originals['temporal']['qualification']
            if row['accuracy_source']!=source:
                raise ValueError('Timing row selects another numerical evidence source')
            if source not in accuracy:
                accuracy[source]=read(ROOT/source)['rows']
            selected_accuracy=[accuracy[source][index] for index in row['accuracy_row_indices']]
            if len(selected_accuracy)!=row['K']:
                raise ValueError('Every intermediate physical output must have original accuracy evidence')
            for lead,original in enumerate(selected_accuracy,1):
                if (original['identifier']!=row['identifier'] or original['K']!=row['K'] or original['lead']!=lead
                        or original['counts_xyz']!=row['counts_xyz'] or original['cells_per_metre']!=row['cells_per_metre']):
                    raise ValueError('Original accuracy row differs from the measured case')
                if cell:
                    window=windows[(row['identifier'],row['start_index'])]
                    member=bank[row['identifier']]
                    if (original['start']!=row['start_index'] or row['output_steps_per_s']!=60
                            or row['start_index_rate_per_s']!=60 or row['schedule']!='captured_native'
                            or row['actual_internal_advances']!=row['K']*member['reference_substeps']
                            or row['current_state_path']!=window['current_state_path']
                            or row['current_state_sha256']!=window['current_state_sha256']):
                        raise ValueError('Captured cell integration or window differs')
                elif (original['start_fine_step']!=row['start_index']
                        or original['output_steps_per_s']!=row['output_steps_per_s']
                        or original['mode']!=row['schedule'] or row['start_index_rate_per_s']!=240
                        or original['groups_per_output']!=row['groups_per_output']
                        or original['actual_internal_advances']!=row['actual_internal_advances']):
                    raise ValueError('Captured temporal cadence/integration differs')
            if (row['output_field_sha256']!=[r['output_sha256'] for r in selected_accuracy]
                    or canonical(row['original_metrics'])!=canonical([r['metrics'] for r in selected_accuracy])
                    or row['original_accuracy_pass']!=all(r['all_accuracy_pass'] for r in selected_accuracy)
                    or not row['all_timed_and_warmup_output_bytes_exact']
                    or not row['first_call_retained_after_all_later_calls'] or not row['owned_output_memory_checked']
                    or row['verified_calls']!=1+protocol['warmups_per_case']+protocol['samples_per_case_per_block']
                    or len(row['block_wall_ms'])!=protocol['samples_per_case_per_block']):
                raise ValueError('Literal output/metric/ownership or complete sample evidence differs')
            samples_summary(row['block_wall_ms'])
            timed_calls+=len(row['block_wall_ms']);timed_outputs+=row['K']*len(row['block_wall_ms'])
            verified_calls+=row['verified_calls'];verified_outputs+=row['K']*row['verified_calls']
            phases=raw_by_execution[execution].setdefault(identity,{})
            if phase in phases:
                raise ValueError('Duplicate process phase for one case')
            phases[phase]=row
        if seen!=expected_sets[execution]:
            raise ValueError('The original worker omitted registered cases')
    if (timed_calls!=protocol['expected_timed_calls'] or timed_outputs!=protocol['expected_timed_physical_outputs']
            or verified_outputs-timed_outputs!=protocol['expected_first_and_warmup_physical_outputs']):
        raise ValueError('Full original sample/output accounting differs')
    for execution,cases in raw_by_execution.items():
        for identity,phases in sorted(cases.items()):
            if set(phases)!={0,1}:
                raise ValueError('Two balanced process blocks required per exact case')
            a,b=phases[0],phases[1]
            scientific=('identifier','split','counts_xyz','cells_per_metre','start_index','start_index_rate_per_s',
                'output_steps_per_s','K','schedule','groups_per_output','actual_internal_advances',
                'current_state_path','current_state_sha256','input_field_sha256','output_field_sha256',
                'accuracy_source','accuracy_row_indices','original_metrics','original_accuracy_pass')
            if canonical({k:a[k] for k in scientific})!=canonical({k:b[k] for k in scientific}):
                raise ValueError('Balanced process blocks do not measure the identical frozen problem')
            block=samples_summary(a['block_wall_ms']+b['block_wall_ms'])
            relative=[metric['relative'] for metric in a['original_metrics']]
            if any(set(m)!=set(METRICS) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0
                                              for v in m.values()) for m in relative):
                raise ValueError('Complete finite nonnegative original accuracy metrics required')
            worst={name:max(m[name] for m in relative) for name in METRICS}
            records.append(dict(execution=execution,**{name:a[name] for name in scientific},
                block_ms=block,ms_per_physical_output={name:(v/a['K'] if name!='samples' else v)
                                                    for name,v in block.items()},
                original_worst_relative_error_over_K=worst,
                median_below_0p2_ms_per_output=block['median']/a['K']<.2,
                all_observed_samples_below_0p2_ms_per_output=block['maximum']/a['K']<.2,
                phase_median_block_ms=[statistics.median(phases[p]['block_wall_ms']) for p in (0,1)]))
    summaries=[]
    for execution in EXPECTED:
        groups={}
        for row in records:
            if row['execution']!=execution:
                continue
            grouping=(tuple(row['counts_xyz']),row['cells_per_metre'],row['output_steps_per_s'],row['K'],
                      row['schedule'],row['actual_internal_advances'])
            groups.setdefault(grouping,[]).append(row)
        for (counts,density,rate,k,schedule,advances),rows in sorted(groups.items()):
            summaries.append(dict(execution=execution,counts_xyz=list(counts),cells_per_metre=density,
                output_steps_per_s=rate,K=k,schedule=schedule,actual_internal_advances_per_block=advances,
                cases=len(rows),samples=sum(r['block_ms']['samples'] for r in rows),
                median_of_case_medians_ms_per_output=statistics.median(r['ms_per_physical_output']['median'] for r in rows),
                worst_case_median_ms_per_output=max(r['ms_per_physical_output']['median'] for r in rows),
                maximum_observed_sample_ms_per_output=max(r['ms_per_physical_output']['maximum'] for r in rows),
                original_worst_relative_error={name:max(r['original_worst_relative_error_over_K'][name] for r in rows)
                                               for name in METRICS},
                all_original_accuracy_pass=all(r['original_accuracy_pass'] for r in rows)))
    comparisons=[]
    for a,b in (('cell_bound','cell_coordinate'),('temporal_eager','temporal_graph')):
        first={key(r):r for r in records if r['execution']==a};second={key(r):r for r in records if r['execution']==b}
        if set(first)!=set(second):
            raise ValueError('Paired execution populations differ')
        ratios=[second[i]['block_ms']['median']/first[i]['block_ms']['median'] for i in sorted(first)]
        comparisons.append(dict(baseline=a,comparison=b,paired_cases=len(first),
            median_paired_block_latency_ratio=statistics.median(ratios),comparison_faster_cases=sum(v<1 for v in ratios),
            exact_complete_output_identity_cases=sum(first[i]['output_field_sha256']==second[i]['output_field_sha256'] for i in first),
            implication='Same physical case only; no cross-family or different-cadence/integration Pareto claim.'))
    anchors=[r for r in records if
        (r['execution'] in ('cell_bound','cell_coordinate') and r['identifier']=='diversity-development-g025-cpm100'
         and r['start_index']==192 and r['K']==4) or
        (r['execution'] in ('temporal_eager','temporal_graph')
         and r['identifier']=='long-v1-continuous_varying-oscillating-r0-cpm50' and r['start_index']==0
         and r['output_steps_per_s']==120 and r['K']==4 and r['schedule']=='fine_reference')]
    if len(anchors)!=4:
        raise ValueError('Both originally quoted anchors and paired execution controls required')
    result=dict(schema='rheon.full-state-GPU-timing-independent-reduction.v1',
        status='all_original_full_state_GPU_samples_independently_bound_and_reduced',created_unix=time.time(),
        producer_sha256=sha(Path(__file__)),protocol_path=str(protocol_path.relative_to(ROOT)),
        protocol_sha256=sha(protocol_path),raw_status_sha256=sha(base/'status.json'),
        source_bindings_verified=len(protocol['source_sha256']),worker_results=worker_bindings,
        timed_calls=timed_calls,timed_physical_outputs=timed_outputs,verified_complete_calls=verified_calls,
        verified_physical_outputs=verified_outputs,complete_pooled_cases=len(records),
        distinct_current_state_archives=len(arrays),frozen_input_field_hashes_independently_recomputed=True,
        every_original_accuracy_row_and_metric_literal_checked=True,all_original_output_hashes_exact=True,
        inference_device='cuda',timing_scope=protocol['timing_scope'],CPU_model_forwards=0,
        independent_physics_rescore=False,own_history_drift_qualified=False,full_portfolio_complete=False,
        new_parameter_fits=0,new_reference_generation=False,output_archives_copied=0,
        scope_note='These are frozen complete full-state executions, not C106/C107 pressure-only requests. '
            'All physical outputs are counted; K is not an internal substep divisor. '
            'Original numerical metrics are inherited only through exact field-byte/case binding. '
            'Saved TF32false is the worker configuration established before captured imports/construction; '
            'pressure/carrier output dtypes and every output bit are independently checked. '
            'Sampled resource observations are not continuous GPU exclusivity proof.',
        anchors=anchors,paired_execution_comparisons=comparisons,summaries=summaries,rows=records)
    if 'torch' in sys.modules:
        raise ValueError('Independent reducer imported a neural runtime')
    report_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],timed_calls=timed_calls,timed_physical_outputs=timed_outputs,
                         verified_physical_outputs=verified_outputs,pooled_cases=len(records),input_archives=len(arrays))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();reduce(args.protocol.resolve(),args.report.resolve())
