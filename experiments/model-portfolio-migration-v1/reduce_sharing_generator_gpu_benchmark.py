"""Reduce the complete six-model sharing campaign, preserving every case/lead."""
from collections import defaultdict
import math
from pathlib import Path
import time

import numpy as np

from reduce_joint_generator_gpu_benchmark import summarize
from verify_joint_generator_retained_probes import relative
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROTOCOL = RUN/'sharing-generator-GPU-benchmark-protocol-01.json'
BASE = RUN/'sharing-generator-GPU-benchmark-01'
OUTPUT = DOC/'reports/sharing-generator-GPU-benchmark-01.json'
FIELDS = ('pressure','vx','vy','vz','tracer')


def saved_probe_check(model, prepared, measured):
    """Original saved CPU array versus fresh GPU field metrics; no CPU forward."""
    case = next(c for c in prepared['cases'] if c['identifier'] == 'temporal-g0-cpm8-r0' and c['start'] == 0)
    nx,ny,nz = case['scene']['counts']
    slices = ((slice(nz),slice(ny),slice(nx)),(slice(nz),slice(ny),slice(nx+1)),
              (slice(nz),slice(ny+1),slice(nx)),(slice(nz+1),slice(ny),slice(nx)),(slice(nz),slice(ny),slice(nx)))
    receipt_path = ROOT/model['receipt']['path']
    receipt = read(receipt_path)
    probe_path = receipt_path.parent/'probe.npy'
    require(binding(probe_path)['sha256'] == receipt['outputs']['probe.npy']['sha256'], 'Original saved probe changed')
    probe = np.load(probe_path,allow_pickle=False)
    require(probe.dtype.str == '<f4' and probe.shape == (1,model['K'],5,nz+1,ny+1,nx+1), 'Original raw probe extent differs')
    norm = read(ROOT/model['normalization']['path'])
    actual = next(r for r in measured['rows'] if r['identifier'] == case['identifier'] and r['start'] == 0)
    comparisons = []
    for lead in range(model['K']):
        physical = {}
        for index,name in enumerate(FIELDS):
            scale = np.float32(norm['scales']['velocity' if name in ('vx','vy','vz') else name])
            value = (probe[0,lead,index][slices[index]]*scale).reshape(-1)
            if name == 'pressure':
                value = value.astype(np.float64)
                value -= value[0]
            physical[name] = value
        with np.load(ROOT/case['reference_leads'][lead]['path'],allow_pickle=False) as truth:
            expected = {name+'_relative_l2':relative(physical[name],truth[name],name == 'pressure') for name in FIELDS}
            expected['velocity_relative_l2'] = relative(np.concatenate([physical[n] for n in ('vx','vy','vz')]),
                                                       np.concatenate([truth[n] for n in ('vx','vy','vz')]))
        observed = actual['metrics'][lead]['values']
        require(all((v is None and observed[n] is None) or (v is not None and observed[n] is not None
                    and math.isclose(v,observed[n],rel_tol=5e-4,abs_tol=5e-6)) for n,v in expected.items()),
                'Saved original CPU probe metrics disagree with fresh GPU boundary')
        comparisons.append(dict(lead=lead+1,original_saved_CPU_metrics=expected,
            fresh_GPU_metrics={n:observed[n] for n in expected}))
    return dict(identifier=model['identifier'],comparisons=comparisons,probe=binding(probe_path),
        scope='First16cubed/cpm8/start0case only; metric agreement, not bitwise CPU/GPU equivalence.')


def run():
    require(not OUTPUT.exists(), 'Preserve original sharing reduction')
    protocol = read(PROTOCOL)
    protocol_binding = binding(PROTOCOL)
    owner = read(BASE/'status.json')
    require(owner['status'] == 'all_original_workers_complete_reduction_pending'
            and owner['active'] is None and len(owner['completed']) == len(protocol['process_blocks']) == 12
            and owner['protocol_sha256'] == protocol_binding['sha256'], 'All12original workers required')
    for spec in [protocol['cases'],*protocol['dependencies']]:
        require(binding(ROOT/spec['path']) == spec, 'Preregistered source/reference input changed')
    prepared = read(ROOT/protocol['cases']['path'])
    expected_cases = {(c['identifier'],c['start']):c for c in prepared['cases']}
    models = {m['identifier']:m for m in prepared['models']}
    require(len(expected_cases) == 64 and len(models) == 6, 'Full original population required')
    admitted = {(b['identifier'],b['phase'],b['result']) for b in protocol['process_blocks']}
    require(len(admitted) == 12 and admitted == {(b['identifier'],b['phase'],b['result']) for b in owner['completed']},
            'Completed worker membership differs')
    reports, evidence, competing = defaultdict(dict), [], []
    timed_calls, timed_outputs, network_calls, all_outputs = 0,0,0,0
    for completed in owner['completed']:
        path = ROOT/completed['result']
        require(binding(path) == completed['result_binding'], 'Raw worker result changed')
        d = read(path)
        name,phase = d['identifier'],d['phase']
        model = models[name]
        require(d['status'] == 'all64_native_cases_measured' and d['fit_item_id'] == model['fit_item_id']
                and d['K'] == model['K'] and d['architecture'] == model['architecture']
                and d['constructor_arguments'] == [model['K'],12,model['architecture']]
                and d['protocol_sha256'] == protocol_binding['sha256']
                and d['CPU_neural_forwards'] == d['forbidden_forward_attempts'] == 0
                and d['CUDA_network_calls'] == 64*(1+protocol['warmups']+protocol['samples'])
                and d['peak_CUDA_allocated_bytes'] <= protocol['max_CUDA_allocated_bytes']
                and d['reference_method'] == prepared['reference_method']
                and d['reference_native_probe_binary_sha256'] == prepared['reference_native_probe_binary_sha256']
                and phase not in reports[name], 'Actual device/constructor/model/reference work differs')
        require(binding(ROOT/d['constructor_source']['path']) == d['constructor_source']
                and d['constructor_source'] in model['sources']
                and binding(ROOT/d['constructor_adapter']['path']) == d['constructor_adapter'], 'Constructor evidence changed')
        observed = set()
        for row in d['rows']:
            key = (row['identifier'],row['start'])
            case = expected_cases[key]
            require(key not in observed and row['counts_xyz'] == case['scene']['counts']
                    and row['cells_per_metre'] == case['scene']['cells_per_metre']
                    and row['K'] == model['K'] and row['output_dt_s'] == 1/60
                    and row['block_advance_s'] == model['K']/60 and row['all_repeat_and_ownership_checks_passed']
                    and len(row['block_ms']) == len(row['per_output_ms']) == protocol['samples']
                    and len(row['warmup_block_ms']) == protocol['warmups']
                    and len(row['metrics']) == len(row['output_hashes']) == model['K'], 'Case/lead/timing population differs')
            observed.add(key)
            for block,per_output in zip(row['block_ms'],row['per_output_ms'],strict=True):
                require(math.isfinite(block) and block > 0 and per_output == block/model['K'], 'Timing must divide only by physical outputs')
            for lead,metric in enumerate(row['metrics'],1):
                require(metric['lead'] == lead and metric['time_s'] == case['reference_leads'][lead-1]['time_s']
                        and set(row['output_hashes'][lead-1]) == set(FIELDS)
                        and all(v is None or (math.isfinite(v) and v >= 0) for v in metric['values'].values()),
                        'Physical output/metric/timestamp differs')
            timed_calls += len(row['block_ms'])
            timed_outputs += len(row['block_ms'])*model['K']
            all_outputs += (1+len(row['warmup_block_ms'])+len(row['block_ms']))*model['K']
        require(observed == expected_cases.keys(), 'A case was omitted')
        network_calls += d['CUDA_network_calls']
        for where in ('resources_before','resources_after'):
            if d[where]['other_compute_pids']:
                competing.append(dict(identifier=name,phase=phase,where=where,pids=d[where]['other_compute_pids']))
        reports[name][phase] = d
        evidence.append(binding(path))
    require(timed_calls == protocol['expected_timed_network_calls'] == 7680
            and timed_outputs == protocol['expected_timed_physical_outputs'] == 23040
            and network_calls == protocol['expected_all_network_calls_including_first_and_warmups'] == 10752
            and all_outputs == protocol['expected_all_physical_outputs_including_first_and_warmups'] == 32256,
            'Total admitted work differs')
    summaries, probes = {}, []
    for name,pair in reports.items():
        require(set(pair) == {0,1}, 'Both order/process phases required')
        a,b = [{(r['identifier'],r['start']):r for r in pair[p]['rows']} for p in (0,1)]
        pooled = []
        for key,first in a.items():
            second = b[key]
            require(first['output_hashes'] == second['output_hashes'], 'Phase output bytes differ')
            for left,right in zip(first['metrics'],second['metrics'],strict=True):
                require(left['lead'] == right['lead'] and left['time_s'] == right['time_s']
                        and left['values'].keys() == right['values'].keys(), 'Phase metric identity differs')
                require(all((v is None and right['values'][n] is None) or
                    (v is not None and right['values'][n] is not None and math.isclose(v,right['values'][n],rel_tol=1e-12,abs_tol=1e-14))
                    for n,v in left['values'].items()), 'Phase metric values differ')
            pooled.append(dict(first,block_ms=first['block_ms']+second['block_ms'],
                per_output_ms=first['per_output_ms']+second['per_output_ms']))
        def grouped(key):
            groups = defaultdict(list)
            for row in pooled:
                groups[key(row)].append(row)
            return {label:summarize(rows) for label,rows in groups.items()}
        summaries[name] = dict(fit_item_id=models[name]['fit_item_id'],K=models[name]['K'],architecture=models[name]['architecture'],
            summary=summarize(pooled),cases=pooled,by_start=grouped(lambda r:str(r['start'])),
            by_grid_density=grouped(lambda r:str(r['cells_per_metre'])),
            by_grid_dimensions=grouped(lambda r:'x'.join(map(str,r['counts_xyz']))),
            by_density_and_dimensions=grouped(lambda r:str(r['cells_per_metre'])+'cpm-'+'x'.join(map(str,r['counts_xyz']))))
        probe = saved_probe_check(models[name],prepared,pair[0])
        probes.append(probe)
        evidence.append(probe['probe'])
    result = dict(status='all6_sharing_generators_12blocks_independently_reduced',created_unix=time.time(),models=summaries,
        cases=64,models_measured=6,timed_joint_calls=timed_calls,timed_physical_outputs=timed_outputs,
        verified_network_calls_including_first_and_warmups=network_calls,
        verified_physical_outputs_including_first_and_warmups=all_outputs,raw_samples_per_model_case=20,
        all_phase_output_hashes_equal=True,CPU_neural_forwards=0,original_saved_CPU_probe_checks=probes,
        timing_units='GPU wall ms/physical_output; allKblock ms retained; complete native CPUinput/output boundary',
        timing_qualified_at_observed_resource_scope=not competing,observed_competing_GPU_compute=competing,
        continuous_hardware_exclusivity_proven=False,exposure=prepared['exposure'],
        reference_method=prepared['reference_method'],reference_native_probe_binary_sha256=prepared['reference_native_probe_binary_sha256'],
        native_publication_complete=False,full_portfolio_complete=False,new_reference_generation=False,array_copies_to_disk=0,
        producer=binding(Path(__file__)),helpers=[binding(Path(__file__).with_name(n)) for n in
            ['reduce_joint_generator_gpu_benchmark.py','verify_joint_generator_retained_probes.py']],
        evidence=[protocol_binding,binding(ROOT/protocol['cases']['path']),binding(BASE/'status.json'),*evidence])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    for name,row in summaries.items():
        print(name,'GPUms/output',row['summary']['median_case_median_per_output_ms'],
              'worst pressure relativeL2',row['summary']['worst_metrics']['pressure_relative_l2'])


if __name__ == '__main__':
    run()
