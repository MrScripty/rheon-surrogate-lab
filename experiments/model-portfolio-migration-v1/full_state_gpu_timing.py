"""Frozen generators: CPU NumPy native fields to K owned CPU NumPy states.

Model arithmetic, weights, controls and integration policies remain captured code.
The common boundary adds the same native layout checks and NumPy/Torch views to
both families. Hashing, finiteness/ownership oracles, setup and warmups are outside
the primary wall interval. No output fields or simulation archives are copied to
disk: every returned field must match its original qualified bytes.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')
EXECUTIONS = {
    'cell_bound': ('cell', 'bound_sampler_pressure_graph_v1', 'BoundSamplerPressureNativeEntry'),
    'cell_coordinate': ('cell', 'cell_coordinate_pressure_graph_v1', 'CellCoordinatePressureNativeEntry'),
    'temporal_eager': ('temporal', 'temporal_frozen_backbone_v1', 'TemporalFrozenBackbone'),
    'temporal_graph': ('temporal', 'temporal_frozen_graph_v1', 'TemporalFrozenGraph'),
}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def field_sizes(counts):
    if (type(counts) not in (list, tuple) or len(counts) != 3
            or any(type(n) is not int or n <= 0 for n in counts)):
        raise ValueError('Three positive native cell counts required')
    x, y, z = counts
    return dict(pressure=x*y*z, vx=(x+1)*y*z, vy=x*(y+1)*z,
                vz=x*y*(z+1), tracer=x*y*z)


def cpu_views(np, torch, native, sizes):
    """One common timed input adapter; qualification proves finite input bytes."""
    if type(native) is not dict or set(native) != set(FIELDS):
        raise ValueError('Exactly the five native physical fields required')
    result = {}
    for name in FIELDS:
        value = native[name]
        dtype = np.dtype('<f8' if name == 'pressure' else '<f4')
        if (not isinstance(value, np.ndarray) or value.dtype != dtype
                or value.size != sizes[name] or not value.flags.c_contiguous):
            raise ValueError('Exact contiguous native CPU dtype and extent required: ' + name)
        result[name] = torch.from_numpy(value.reshape(-1))
    return result


def numpy_outputs(torch, outputs, k, sizes):
    """Common timed return projection over the captured owned CPU allocations."""
    if type(outputs) is not list or len(outputs) != k:
        raise ValueError('Every requested physical output is required')
    result = []
    for row in outputs:
        if type(row) is not dict or set(row) != set(FIELDS):
            raise ValueError('Complete output fields required')
        projected = {}
        for name in FIELDS:
            value = row[name]
            dtype = torch.float64 if name == 'pressure' else torch.float32
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != dtype or value.numel() != sizes[name]
                    or not value.is_contiguous()):
                raise ValueError('Exact contiguous owned CPU output required: ' + name)
            projected[name] = value.reshape(-1).numpy()
        result.append(projected)
    return result


def hashes(fields):
    return {name: hashlib.sha256(fields[name].tobytes()).hexdigest() for name in FIELDS}


def check_owned(np, outputs, native, previous=None):
    fields = [row[name] for row in outputs for name in FIELDS]
    other = list(native.values())
    if previous is not None:
        other += [row[name] for row in previous for name in FIELDS]
    for index, value in enumerate(fields):
        if any(np.shares_memory(value, candidate) for candidate in other + fields[:index]):
            raise ValueError('Outputs alias input, retained call or another physical field/output')


def resources():
    def query(arguments):
        return subprocess.check_output(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'],
                                       text=True, timeout=20).strip()
    processes = query(['--query-compute-apps=pid,used_memory'])
    return dict(created_unix=time.time(), compute_processes=processes,
        other_compute_pids=[int(row.split(',')[0]) for row in processes.splitlines()
                            if row.strip() and int(row.split(',')[0]) != os.getpid()],
        GPU=query(['--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,'
                   'clocks.current.graphics,clocks.current.memory,utilization.gpu,memory.used']),
        load_average=list(os.getloadavg()))


def cell_cases(reg, execution):
    bank = read(ROOT / reg['bank'])['trajectories']
    if len(bank) != 24 or any(row['split'] != 'development' for row in bank):
        raise ValueError('Exactly the original24development scenes required')
    windows = {}
    for selected in reg['window_sources']:
        for row in read(ROOT / selected['controller'])['completed']:
            if row['identifier'] in windows:
                raise ValueError('Duplicate frozen window scene')
            windows[row['identifier']] = row
    if {row['identifier'] for row in bank} != set(windows):
        raise ValueError('Frozen scene/window populations differ')
    accuracy_path = reg['baseline_outputs'] if execution == 'cell_bound' else reg['candidate_outputs']
    rows = read(ROOT / accuracy_path)['rows']
    expected = {(r['identifier'], r['start'], r['K'], r['lead']): (i, r)
                for i, r in enumerate(rows) if r['split'] == 'development'}
    if len(expected) != 960:
        raise ValueError('All960original development output identities required')
    cases = []
    for scene in sorted(bank, key=lambda row: row['identifier']):
        path = ROOT / scene['receipt']
        if sha(path) != scene['receipt_sha256']:
            raise ValueError('Frozen scene receipt differs')
        physical = read(path)['scene']
        if len(windows[scene['identifier']]['windows']) != 4:
            raise ValueError('All four original current-state windows required')
        for window in windows[scene['identifier']]['windows']:
            for k in (1, 2, 3, 4):
                selected = [expected[(scene['identifier'], window['start'], k, lead)]
                            for lead in range(1, k+1)]
                cases.append(dict(identifier=scene['identifier'], split=scene['split'],
                    scene=physical, start_index=window['start'], start_index_rate_per_s=60,
                    output_steps_per_s=60, K=k, schedule='captured_native',
                    groups_per_output=[1] * scene['reference_substeps'],
                    actual_internal_advances=k * scene['reference_substeps'],
                    current_state_path=window['current_state_path'],
                    current_state_sha256=window['current_state_sha256'],
                    accuracy_source=accuracy_path, accuracy_row_indices=[i for i, _ in selected],
                    expected=[r['output_sha256'] for _, r in selected],
                    original_metrics=[r['metrics'] for _, r in selected],
                    original_accuracy_pass=all(r['all_accuracy_pass'] for _, r in selected),
                    internal_substeps=scene['reference_substeps']))
    return cases


def temporal_cases(reg):
    from long_temporal_dataset_v1 import build_index, LongTemporalDataset
    index = build_index([ROOT / path for path in reg['audits']])
    dataset = LongTemporalDataset(index)
    rows = read(ROOT / reg['qualification'])['rows']
    expected = {(r['identifier'], r['start_fine_step'], r['output_steps_per_s'],
                 r['K'], r['lead'], r['mode']): (i, r) for i, r in enumerate(rows)}
    cases = []
    for example in sorted(index['examples'], key=lambda e: (e['identifier'],
                          e['start_fine_step'], e['output_steps_per_s'], e['K'])):
        if example['start_fine_step'] not in reg['start_fine_steps']:
            continue
        identifier, start, rate, k = (example[name] for name in
            ('identifier', 'start_fine_step', 'output_steps_per_s', 'K'))
        scene = dataset.receipts[identifier][1]['scene']
        stride = 240 // rate
        for mode in reg['modes']:
            groups = {'fine_reference': [1]*stride, 'half_passes': [2]*(stride//2),
                      'single_pass': [stride]}[mode]
            selected = [expected[(identifier, start, rate, k, lead, mode)] for lead in range(1, k+1)]
            path, _, states = dataset.receipts[identifier]
            cases.append(dict(identifier=identifier, split=example['split'], scene=scene,
                start_index=start, start_index_rate_per_s=240, output_steps_per_s=rate, K=k,
                schedule=mode, groups_per_output=groups, actual_internal_advances=k*len(groups),
                current_state_path=str((path.parent / states[start]['path']).relative_to(ROOT)),
                current_state_sha256=states[start]['sha256'],
                accuracy_source=reg['qualification'], accuracy_row_indices=[i for i, _ in selected],
                expected=[r['output_sha256'] for _, r in selected],
                original_metrics=[r['metrics'] for _, r in selected],
                original_accuracy_pass=all(r['all_accuracy_pass'] for _, r in selected)))
    if len(cases) != 216 or len({c['identifier'] for c in cases}) != 2:
        raise ValueError('Exactly216original temporal cases across two audited masters required')
    return cases


def run(protocol_path, execution, phase, output):
    protocol = read(protocol_path)
    if (execution not in EXECUTIONS or phase not in (0, 1) or output.exists()
            or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda'
            or not output.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1')):
        raise ValueError('Fresh task-owned CUDA worker and registered execution/phase required')
    for path, expected in protocol['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen timing dependency differs: ' + path)
    os.sched_setaffinity(0, {protocol['host_cpu']})
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    counts = dict(cuda_module_calls=0, forbidden_module_calls=0)

    def qualify_device(module, args):
        tensors = [value for value in args if isinstance(value, torch.Tensor)]
        if not tensors or any(value.device.type != 'cuda' for value in tensors):
            counts['forbidden_module_calls'] += 1
            raise ValueError('CPU or unclassified neural module forward forbidden')
        counts['cuda_module_calls'] += 1

    guard = torch.nn.modules.module.register_module_forward_pre_hook(qualify_device)
    output.mkdir(parents=True)
    result = dict(schema='rheon.complete-native-full-state-GPU-timing.v1',status='starting',
        pid=os.getpid(),execution=execution,phase=phase,protocol_sha256=sha(protocol_path),
        inference_device='cuda',timing_units='ms/block',per_step_divisor='K physical outputs only',
        timing_scope=protocol['timing_scope'],rows=[],resource_observations=[],
        device_qualification=counts,timed_module_hooks_enabled=False,CPU_model_forwards=0,
        optimizer_updates=0,new_reference_generation=False,reference_fields_passed_to_model=False,
        setup_and_oracles_outside_primary_clock=True,output_archives_written=0,
        independent_accuracy_confirmation=False,own_history_drift_qualified=False,
        full_portfolio_complete=False,cleanup_complete=False)
    model = None
    begun = time.monotonic()

    def checkpoint():
        result['elapsed_worker_seconds'] = time.monotonic()-begun
        result['peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        save(output/'results.json',result)

    def resource_gate():
        observation = resources();result['resource_observations'].append(observation)
        if observation['other_compute_pids']:
            raise RuntimeError('GPU compute contention; retain this incomplete original campaign')

    def bounded():
        if (time.monotonic()-begun > protocol['worker_deadline_seconds']
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > protocol['maximum_worker_RSS_bytes']
                or torch.cuda.max_memory_allocated() > protocol['maximum_Torch_GPU_allocated_bytes']):
            raise TimeoutError('Registered full-state timing resource bound exceeded')

    checkpoint()
    try:
        resource_gate()
        sys.path.insert(0,str(ROOT/'experiments/full-ai-data-diversity-v1'))
        import support
        from diagnose_causal_composition import inventory
        family, module_name, class_name = EXECUTIONS[execution]
        module = importlib.import_module(module_name)
        original = read(ROOT/protocol['original_protocols'][family])
        if sha(ROOT/protocol['original_protocols'][family]) != protocol['original_protocol_sha256'][family]:
            raise ValueError('Original workload protocol differs')
        cases = cell_cases(original,execution) if family == 'cell' else temporal_cases(original)
        if phase == 1:
            cases.reverse()
        if len(cases) != protocol['expected_cases'][family]:
            raise ValueError('Complete registered population required')
        identities = {(c['identifier'], c['start_index'], c['output_steps_per_s'],
                       c['K'], c['schedule']) for c in cases}
        if len(identities) != len(cases):
            raise ValueError('Duplicate physical timing case')
        tick = time.perf_counter()
        build = original.get('baseline_build',original['build']) if execution == 'cell_bound' else original['build']
        model = getattr(module,class_name)(original['transport_pair'],original['shared_pair'],build,original['control_build'])
        torch.cuda.synchronize()
        original_inventory = inventory(model)
        result.update(constructor_ms=1000*(time.perf_counter()-tick),model_identity=model.asset_identity,
            parameter_inventory=original_inventory,source_module=str(Path(module.__file__).resolve().relative_to(ROOT)),
            GPU_model=torch.cuda.get_device_name(),torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            python_version=sys.version,host_affinity=sorted(os.sched_getaffinity(0)),
            torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
            pressure_dtype='float64',carrier_dtype='float32',tf32=False,autocast=False,
            samples_per_case=protocol['samples_per_case_per_block'],warmups_per_case=protocol['warmups_per_case'],
            expected_cases=len(cases))
        from temporal_native_transfer_v1 import transaction
        current_identifier = None
        with torch.inference_mode():
            for index, case in enumerate(cases):
                bounded()
                if case['identifier'] != current_identifier:
                    if current_identifier is not None and hasattr(model,'release_graphs'):
                        model.release_graphs()
                    resource_gate();current_identifier=case['identifier']
                state_path=ROOT/case['current_state_path']
                if sha(state_path) != case['current_state_sha256']:
                    raise ValueError('Frozen current native state archive differs')
                with np.load(state_path,allow_pickle=False) as archive:
                    native={name:archive[name].copy() for name in FIELDS}
                if not all(np.isfinite(value).all() for value in native.values()):
                    raise ValueError('Original current state contains nonfinite fields')
                initial_input=hashes(native);sizes=field_sizes(case['scene']['counts']);k=case['K']

                def complete():
                    fields=cpu_views(np,torch,native,sizes)
                    if family == 'cell':
                        values=model.predict_native(fields,case['scene'],case['start_index'],k,case['internal_substeps'])
                    else:
                        values=transaction(model,fields,case['scene'],case['start_index'],
                                           case['output_steps_per_s'],k,case['groups_per_output'])
                    return numpy_outputs(torch,values,k,sizes)

                def verify(values):
                    if ([hashes(row) for row in values] != case['expected'] or hashes(native) != initial_input
                            or any(not np.isfinite(value).all() for row in values for value in row.values())):
                        raise ValueError('Complete returned state differs from original accuracy capture or input changed')

                tick=time.perf_counter()
                if family == 'cell':
                    model.prepare_native(cpu_views(np,torch,native,sizes),case['scene'],case['start_index'],k,case['internal_substeps'])
                torch.cuda.synchronize()
                explicit_setup_ms=1000*(time.perf_counter()-tick)
                tick=time.perf_counter();retained=complete();first_ms=1000*(time.perf_counter()-tick)
                verify(retained);check_owned(np,retained,native)
                for _ in range(protocol['warmups_per_case']):
                    verify(complete())
                samples=[];last=None
                guard.remove();guard=None
                try:
                    for _ in range(protocol['samples_per_case_per_block']):
                        torch.cuda.synchronize()
                        tick=time.perf_counter();last=complete();elapsed=1000*(time.perf_counter()-tick)
                        samples.append(elapsed);verify(last)
                finally:
                    guard=torch.nn.modules.module.register_module_forward_pre_hook(qualify_device)
                verify(retained);check_owned(np,last,native,retained)
                row={key:case[key] for key in ('identifier','split','start_index','start_index_rate_per_s',
                    'output_steps_per_s','K','schedule','groups_per_output','actual_internal_advances',
                    'current_state_path','current_state_sha256','accuracy_source','accuracy_row_indices',
                    'original_metrics','original_accuracy_pass')}
                row.update(counts_xyz=case['scene']['counts'],cells_per_metre=case['scene']['cells_per_metre'],
                    output_dt_s=1/case['output_steps_per_s'],block_duration_s=k/case['output_steps_per_s'],
                    input_field_sha256=initial_input,output_field_sha256=case['expected'],
                    explicit_setup_ms=explicit_setup_ms,first_pre_warmup_ms=first_ms,
                    first_pre_warmup_includes_any_remaining_lazy_setup=True,
                    block_wall_ms=samples,verified_calls=1+protocol['warmups_per_case']+len(samples),
                    first_call_retained_after_all_later_calls=True,owned_output_memory_checked=True,
                    all_timed_and_warmup_output_bytes_exact=True)
                result['rows'].append(row)
                if (index+1)%16 == 0 or index+1 == len(cases):
                    checkpoint();print(execution,phase,index+1,'/',len(cases),'complete native cases',flush=True)
        bounded();resource_gate()
        if inventory(model) != original_inventory or counts['forbidden_module_calls']:
            raise ValueError('Frozen parameter inventory or GPU-only qualification differs')
        for path, expected in protocol['source_sha256'].items():
            if sha(ROOT/path) != expected:
                raise ValueError('Frozen timing source changed during worker: '+path)
        result.update(status='numerical_timing_complete_cleanup_pending',
            peak_Torch_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
            all_registered_cases_complete=True,source_bindings_unchanged=True,
            parameter_inventory_unchanged=True)
    except BaseException as error:
        result.update(status='incomplete_original_full_state_GPU_timing',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        try:
            if model is not None:
                model.close()
            if guard is not None:
                guard.remove()
            result['cleanup_complete']=True
            if result['status']=='numerical_timing_complete_cleanup_pending':
                result['status']='complete_original_full_state_GPU_timing'
        except BaseException as error:
            result.update(status='incomplete_full_state_GPU_cleanup',cleanup_error_type=type(error).__name__,cleanup_error=str(error))
            raise
        finally:
            checkpoint()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--execution',choices=tuple(EXECUTIONS),required=True)
    parser.add_argument('--phase',type=int,choices=(0,1),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.protocol.resolve(),args.execution,args.phase,args.output.resolve())
