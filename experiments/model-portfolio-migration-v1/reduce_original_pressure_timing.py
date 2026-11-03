"""Independent stdlib reduction of primary samples and paired observer costs."""
import argparse
from collections import Counter,defaultdict
import statistics
from pathlib import Path
import time

from import_common_generator_measurements import ROOT,RUN,DOC,read,save,binding,require
from original_train_pressure_accuracy_02 import request_binding


def validate_case(row,baseline,case,index,phase,protocol):
    require(row['case_index'] == index and row['case_id'] == case['id']
            and request_binding(row) == request_binding(baseline) == case['original_case']
            and row['output_sha256'] == baseline['output_sha256'] and row['output_independent_of_inputs']
            and row['cells_per_metre'] == case['cells_per_metre'] and row['input_dt_s'] == case['input_dt_s'],
            'Exact case/input/reference/output identities required')
    samples = row['raw_wall_nanoseconds']
    require(len(samples) == protocol['samples_per_case'] and all(type(v) is int and v > 0 for v in samples)
            and row['median_ms'] == statistics.median(samples)/1e6,'Primary raw samples/statistic differ')
    require(type(row['instrumented_first_call_ns']) is int and row['instrumented_first_call_ns'] > 0
            and len(row['instrumented_warmup_ns']) == protocol['warmups_per_case']
            and all(type(v) is int and v > 0 for v in row['instrumented_warmup_ns']),
            'Guarded first call/warmups must be retained separately')
    require(len(row['calibration']) == protocol['calibration_pairs_per_case'],'Complete paired calibration required')
    differences = []
    for number,pair in enumerate(row['calibration']):
        order = ['on','off'] if (phase+index+number)%2 == 0 else ['off','on']
        require(pair['pair_index'] == number and pair['order'] == order
                and all(type(pair[name]) is int and pair[name] > 0 for name in ('on_ns','off_ns'))
                and pair['on_minus_off_ns'] == pair['on_ns']-pair['off_ns'],
                'Calibration order,raw samples or unmodified signed difference differs')
        differences.append(pair['on_minus_off_ns'])
    return samples,differences


def reduce_protocol(protocol_path):
    protocol = read(protocol_path)
    folder = ROOT/protocol['output_root']
    status_path = folder/'status.json'
    status = read(status_path)
    require(status['status'] == 'all_original_pressure_GPU_timing_phases_terminal' and status['all_qualified']
            and not status['pending'] and not status['adapter_gaps'] and status['active_child'] is None
            and not Path(f'/proc/{status["pid"]}').exists(),'Original timing owner must be terminal and complete')
    require(status['protocol'] == binding(protocol_path),'Timing owner protocol binding differs')
    for name,expected in protocol['source_sha256'].items():
        import hashlib
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected,'Frozen timing source changed')
    accepted = read(ROOT/protocol['accuracy_reduction']['path'])
    require(binding(ROOT/protocol['accuracy_reduction']['path']) == protocol['accuracy_reduction']
            and accepted['status'] == 'all23_original_training_pressure_fits_GPU_accuracy_independently_reduced',
            'Independent original accuracy reduction changed')
    accuracy_by_fit = {row['fit_item_id']:row for row in accepted['rows']}
    require(binding(ROOT/protocol['frozen_cases']['path']) == protocol['frozen_cases'],'Frozen case document changed')
    frozen = read(ROOT/protocol['frozen_cases']['path'])
    expected_blocks = {(row['fit_item_id'],row['phase']) for row in protocol['block_order']}
    actual_blocks = [(row['fit_item_id'],row['phase']) for row in status['qualified']]
    require(len(actual_blocks) == len(expected_blocks) == 2*len(protocol['fit_ids'])
            and set(actual_blocks) == expected_blocks,'Every fit requires exactly two fresh phases')
    by_fit = defaultdict(dict)
    result_bindings,pids = [],set()
    for block in status['qualified']:
        spec = block['result']
        require(binding(ROOT/spec['path']) == spec,'Original raw timing result changed')
        result = read(ROOT/spec['path'])
        fid,phase = block['fit_item_id'],block['phase']
        require(result['fit_item_id'] == fid and result['phase'] == phase
                and result['status'] == 'original_train_pressure_GPU_timing_qualified'
                and result['protocol'] == binding(protocol_path) and result['cases'] == len(result['rows']) == 128
                and result['pid'] == block['pid'] and result['pid'] not in pids and not Path(f'/proc/{result["pid"]}').exists(),
                'Two genuinely fresh terminal CUDA processes required')
        pids.add(result['pid'])
        require(result['accuracy'] == protocol['accuracy'][fid] == accuracy_by_fit[fid]['result']
                and binding(ROOT/result['accuracy']['path']) == result['accuracy'],'Original accuracy binding differs')
        accuracy = read(ROOT/result['accuracy']['path'])
        require(result['weights_sha256'] == accuracy['weights_sha256'] and result['original_architecture'] == accuracy['architecture']
                and result['learned_tensors_exact_and_unchanged'] and result['captured_architecture_unchanged']
                and result['all_output_hashes_exact'] and not result['unbound_lab_Python_modules']
                and result['inference_device'] == 'cuda' and result['GPU'] == protocol['expected_GPU']
                and result['timing_units'] == 'ms/pressure_request' and not result['full_simulation_step']
                and result['threads'] == result['interop_threads'] == 1 and result['host_cpu'] == protocol['host_cpu']
                and not result['timed_module_hooks_enabled'] and result['primary_observer_calls'] == 0,
                'Source,parameter,architecture,device or timing contract differs')
        observations = result['resource_observations']
        require(len(observations) == 5 and all(not row['other_compute_pids'] for row in observations)
                and result['peak_CUDA_allocated_bytes'] <= protocol['maximum_Torch_GPU_allocated_bytes']
                and result['peak_RSS_bytes'] <= protocol['maximum_worker_RSS_bytes'],'Resource qualification differs')
        guard = result['device_observation']
        require(guard['closed'] and not guard['observer_attached'] and not guard['forbidden_forward_attempts'],
                'Device observer lifecycle differs')
        counts = guard['calls_by_phase']
        first = counts['first_call']
        require(first and counts['warmup'] == {name:value*protocol['warmups_per_case'] for name,value in first.items()}
                and counts['calibration_on'] == {name:value*protocol['calibration_pairs_per_case'] for name,value in first.items()}
                and not counts.get('primary') and not counts.get('calibration_off'),
                'Observed warmup/calibration counts disagree with same-input guarded calls')
        order = list(range(128)) if phase == 0 else list(reversed(range(128)))
        require(result['case_execution_order'] == order and [row['case_index'] for row in result['rows']] == order,
                'Registered reversed case execution order differs')
        indexed = {}
        for row in result['rows']:
            index = row['case_index']
            validate_case(row,accuracy['rows'][index],frozen['cases'][index],index,phase,protocol)
            indexed[index] = row
        require(result['task_calls_timed'] == 128*protocol['samples_per_case']
                and result['logical_calls'] == 128*(1+protocol['warmups_per_case']+protocol['samples_per_case']+
                                                  2*protocol['calibration_pairs_per_case']),
                'Physical request call totals differ')
        require(result['case_median_median_ms'] == statistics.median(row['median_ms'] for row in result['rows'])
                and result['worst_case_median_ms'] == max(row['median_ms'] for row in result['rows']),
                'Worker summary differs from independently reduced raw timings')
        by_fit[fid][phase] = indexed
        result_bindings.append(spec)
    rows = []
    for fid in protocol['fit_ids']:
        phases = by_fit[fid]
        require(set(phases) == {0,1},'Both original phases required')
        pooled,all_differences,groups = [],[],defaultdict(list)
        accuracy = read(ROOT/protocol['accuracy'][fid]['path'])
        for index in range(128):
            a,b = phases[0][index],phases[1][index]
            require(request_binding(a) == request_binding(b) and a['output_sha256'] == b['output_sha256'],
                    'Cross-phase exact input/output qualification differs')
            samples = a['raw_wall_nanoseconds']+b['raw_wall_nanoseconds']
            differences = [pair['on_minus_off_ns'] for row in (a,b) for pair in row['calibration']]
            item = dict(case_index=index,case_id=a['case_id'],counts_xyz=a['counts_xyz'],cells_per_metre=a['cells_per_metre'],
                input_dt_s=a['input_dt_s'],phase_median_ms=[a['median_ms'],b['median_ms']],pooled_median_ms=statistics.median(samples)/1e6,
                minimum_ms=min(samples)/1e6,maximum_ms=max(samples)/1e6,
                paired_observer_overhead_median_ms=statistics.median(differences)/1e6,accuracy=accuracy['rows'][index]['metrics'])
            pooled.append(item)
            all_differences.extend(differences)
            groups[(a['cells_per_metre'],tuple(a['counts_xyz']),a['input_dt_s'])].append(item)
        strata = []
        for (density,grid,dt),cases in sorted(groups.items()):
            strata.append(dict(cells_per_metre=density,counts_xyz=list(grid),input_dt_s=dt,cases=len(cases),
                median_case_ms=statistics.median(item['pooled_median_ms'] for item in cases),
                maximum_case_median_ms=max(item['pooled_median_ms'] for item in cases),
                worst={name:max(item['accuracy'][name] for item in cases) for name in accuracy['worst']}))
        rows.append(dict(fit_item_id=fid,name=accuracy_by_fit[fid]['name'],accuracy=protocol['accuracy'][fid],cases=128,
            worst=accuracy['worst'],case_median_median_ms=statistics.median(item['pooled_median_ms'] for item in pooled),
            worst_case_median_ms=max(item['pooled_median_ms'] for item in pooled),
            phase_case_median_medians_ms=[statistics.median(phases[p][i]['median_ms'] for i in range(128)) for p in (0,1)],
            paired_observer_overhead_median_ms=statistics.median(all_differences)/1e6,
            paired_observer_negative_differences=sum(value < 0 for value in all_differences),
            primary_calls=256*protocol['samples_per_case'],calibration_pairs=256*protocol['calibration_pairs_per_case'],
            density_grid_timestep_strata=strata,case_summaries=pooled))
    return dict(protocol=binding(protocol_path),owner_status=binding(status_path),rows=rows,raw_results=result_bindings,
        fits=len(rows),fresh_processes=len(pids),primary_calls=len(rows)*256*protocol['samples_per_case'],
        logical_calls=len(rows)*256*(1+protocol['warmups_per_case']+protocol['samples_per_case']+2*protocol['calibration_pairs_per_case']),
        calibration_pairs=len(rows)*256*protocol['calibration_pairs_per_case'])


def run(mode):
    report = DOC/f'reports/original-pressure-GPU-timing-{mode}-01.json'
    require(not report.exists(),'Preserve original timing reduction')
    pilot_path = RUN/'original-pressure-GPU-timing-pilot-protocol-01.json'
    first = reduce_protocol(pilot_path)
    chunks = [first] if mode == 'pilot' else [first,reduce_protocol(RUN/'original-pressure-GPU-timing-batch-protocol-01.json')]
    rows = [row for chunk in chunks for row in chunk['rows']]
    expected = 1 if mode == 'pilot' else 23
    require(len(rows) == len({row['fit_item_id'] for row in rows}) == expected,'Exact default pressure fit coverage required')
    save(report,dict(status='original_pressure_GPU_timing_'+mode+'_independently_reduced',created_unix=time.time(),
        fits=len(rows),rows=rows,fresh_processes=sum(chunk['fresh_processes'] for chunk in chunks),
        primary_calls=sum(chunk['primary_calls'] for chunk in chunks),logical_calls=sum(chunk['logical_calls'] for chunk in chunks),
        calibration_pairs=sum(chunk['calibration_pairs'] for chunk in chunks),protocols=[chunk['protocol'] for chunk in chunks],
        owner_statuses=[chunk['owner_status'] for chunk in chunks],raw_results=[spec for chunk in chunks for spec in chunk['raw_results']],
        accuracy_reduction=binding(DOC/'reports/original-train-pressure-CUDA-accuracy-01.json'),
        inference_device='cuda',timing_units='ms/pressure_request',primary_hooks_enabled=False,
        paired_calibration_reported_separately=True,overhead_subtracted=False,negative_overhead_samples_clipped=False,
        all_output_hashes_exact=True,all_learned_tensors_and_architectures_exact=True,
        device_observation_scope='CUDA construction/setup,first,warmup,on-calibration. Primary/off-calibration have observer removed, exact frozen code/input/weight/output identities; those calls are not claimed individually observed.',
        resource_scope='Before/every32cases/after GPU compute snapshots are clear, not continuous exclusivity. Host load and GPU clocks/temperature retained in raw results.',
        comparison_scope='Same128old-solver consumed native pressure requests and complete CPU-array/CUDA/ownedCPU-array boundary as127named pressure fits. GPU time per pressure request; no pressure-pass or simulation-step division. Campaigns not simultaneous paired optimization trials.',
        full_simulation_step=False,independent_holdout=False,own_history_drift=False,new_parameter_updates=0,
        new_reference_generation=False,full_portfolio_complete=False,producer=binding(Path(__file__))))
    print(expected,'original pressure fits independently reduced; primary and calibration separated',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('pilot','batch'),required=True)
    args = parser.parse_args()
    run(args.mode)
