"""Same-case CUDA accuracy at captured/default and explicit correction budgets."""
import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import sys
import time

from pressure_gpu_timing import ROOT, read, save, sha


def run(protocol_path, identifier, output):
    if output.exists() or os.environ.get('TULDOK_TORCH_DEVICE') != 'cuda':
        raise ValueError('Fresh GPU-only correction-budget output required')
    protocol = read(protocol_path)
    for path, expected in protocol['source_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Preregistered policy/loader source changed: ' + path)
    specification = protocol['models'][identifier]
    os.sched_setaffinity(0, {protocol['host_cpu']})
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device_calls = dict(cuda=0, forbidden=0)

    def device_guard(module, args):
        tensors = [value for value in args if isinstance(value, torch.Tensor)]
        if not tensors or any(value.device.type != 'cuda' for value in tensors):
            device_calls['forbidden'] += 1
            raise ValueError('CPU or unclassified neural execution forbidden')
        device_calls['cuda'] += 1

    guard = torch.nn.modules.module.register_module_forward_pre_hook(device_guard)
    root_hook = None
    result = dict(status='starting', pid=os.getpid(), identifier=identifier, rows=[],
        created_unix=time.time(), protocol_sha256=sha(protocol_path), inference_device='cuda',
        reference_epoch='frozen_old_Rheon', physical_simulation_steps_advanced=0,
        latency_claim=False, new_optimizer_updates=0, new_reference_generation=False,
        CPU_model_forward=False, device_calls=device_calls, independent_confirmation=False,
        full_portfolio_complete=False, numerical_scope='Consumed original128native pressure equations; pressure corrections do not advance physical time.')
    output.mkdir(parents=True)
    arrays = {}
    output_members = {}
    try:
        from pressure_checkpoint_owner import load
        state = load(specification)
        np = state.np
        before = state.data.helpers().state_inventory(state.network)
        architecture = copy.deepcopy(state.model.ARCHITECTURE)
        state.model.configure('cuda')
        state.network.to('cuda').eval()
        if any(value.device.type != 'cuda' or value.dtype != torch.float32 for value in state.network.parameters()):
            raise ValueError('Original float32 parameters must reside on CUDA')
        sys.path.insert(0, str(ROOT / 'experiments/pressure-pass-sweep-v1'))
        import inference
        import pressure_cases
        from score_pressure_portfolio import equation, operator_check
        operator_check()
        frozen_path = ROOT / protocol['frozen_cases']['path']
        if sha(frozen_path) != protocol['frozen_cases']['sha256']:
            raise ValueError('Frozen input population changed')
        frozen = read(frozen_path)
        cases = pressure_cases.load(torch, np)
        reference_path = ROOT / specification['reference']['path']
        if sha(reference_path) != specification['reference']['sha256']:
            raise ValueError('Original proposal/parameter-state reference changed')
        reference = read(reference_path)
        reference_archive = reference_path.parent / reference['output_archive']['path']
        if sha(reference_archive) != reference['output_archive']['sha256']:
            raise ValueError('Retained original prediction archive is missing or changed')
        if sha(ROOT / specification['weights_path']) != reference['weights_sha256']:
            raise ValueError('Actual checkpoint bytes differ from the stated reference/equivalence candidate')
        if len(cases) != len(reference['rows']) or len(cases) != 128:
            raise ValueError('Complete128pressure population required')
        root_calls = [0]

        def count_root(module, args):
            root_calls[0] += 1

        root_hook = state.network.register_forward_pre_hook(count_root)
        result.update(custody=state.custody, gpu_model=torch.cuda.get_device_name(),
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            host_affinity=sorted(os.sched_getaffinity(0)), host_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(), architecture=architecture,
            configured_correction=copy.deepcopy(state.model.CORRECTION),
            weights_sha256=reference['weights_sha256'], reference=specification['reference'],
            reference_role=specification['reference_role'],
            tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32),
            autocast=False, network_dtype='float32', pressure_dtype='float64')
        begun = time.monotonic()
        parity_count = 0
        with torch.inference_mode():
            for index, ((identity, sample), expected, case) in enumerate(zip(cases, reference['rows'], frozen['cases'], strict=True)):
                if time.monotonic() - begun > protocol['worker_deadline_seconds']:
                    raise TimeoutError('Finite correction-work deadline')
                inputs = {key: sample[key] for key in ('base','native_input_residual','cell_size_m','rho','dt')}
                hashes = {key: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() for key,value in inputs.items()}
                if hashes != expected['input_sha256'] or hashes != case['original_case']['input_sha256']:
                    raise ValueError('Scored equations differ from frozen original inputs')
                for field, key in (('reference', 'reference_pressure_sha256'), ('rhs', 'reference_rhs_sha256')):
                    if hashlib.sha256(sample[field].contiguous().numpy().tobytes()).hexdigest() != expected[key]:
                        raise ValueError('Original numerical reference field changed')
                for policy in protocol['policies']:
                    # A policy receives fresh device fields, so one candidate
                    # cannot alter the starting equation for the next policy.
                    device = {key: value.to('cuda') for key,value in inputs.items()}
                    prior_calls = root_calls[0]
                    prediction = (state.model.predict_pressure(state.network, device, state.normalization)
                                  if policy == 'captured_default' else inference.fixed(state, device, int(policy.split('_')[1])))
                    if prediction.device.type != 'cuda' or prediction.dtype != torch.float64:
                        raise ValueError('Captured CUDA float64 pressure required')
                    actual_calls = root_calls[0] - prior_calls
                    requested = state.model.CORRECTION['steps'] if policy == 'captured_default' else int(policy.split('_')[1])
                    if actual_calls != requested:
                        raise ValueError('Actual learned correction work differs from declared policy')
                    pressure = prediction.detach().cpu()
                    torch.cuda.synchronize()
                    digest = hashlib.sha256(pressure.contiguous().numpy().tobytes()).hexdigest()
                    matches = digest == expected['output_sha256']
                    if policy == 'captured_default':
                        parity_count += int(matches)
                        if specification['require_reference_parity'] and not matches:
                            raise ValueError('Loaded captured/default policy differs from original GPU output')
                    residual = equation(pressure, sample['cell_size_m'], sample['rho']) - sample['rhs']
                    difference = pressure - sample['reference']
                    difference = difference - difference.mean()
                    reference_pressure = sample['reference'] - sample['reference'].mean()
                    rhs_l2 = float(torch.linalg.vector_norm(sample['rhs']))
                    pressure_l2 = float(torch.linalg.vector_norm(reference_pressure))
                    metrics = dict(common_equation_relative_l2=float(torch.linalg.vector_norm(residual))/max(rhs_l2,1e-30),
                        pressure_mean_free_relative_l2=float(torch.linalg.vector_norm(difference))/max(pressure_l2,1e-30),
                        pressure_absolute_rms=float(difference.square().mean().sqrt()),
                        pressure_absolute_max=float(difference.abs().max()))
                    if not bool(torch.isfinite(pressure).all()) or not all(math.isfinite(value) for value in metrics.values()):
                        raise ValueError('Nonfinite prediction or metric')
                    location = dict(original_capture=specification['reference'],
                                    original_archive=reference['output_archive'], original_member=f'case{index:03d}_pressure')
                    if not matches:
                        output_key = (tuple(pressure.shape), digest)
                        member = output_members.get(output_key)
                        if member is None:
                            member = policy + f'_case{index:03d}_pressure'
                            arrays[member] = pressure.numpy().copy()
                            output_members[output_key] = member
                        location = dict(new_archive='pressure-policy-outputs.npz', member=member)
                    result['rows'].append(dict(case_id=case['id'], **identity, policy=policy,
                        actual_network_root_calls=root_calls[0]-prior_calls,
                        requested_corrections=None if policy == 'captured_default' else int(policy.split('_')[1]),
                        output_sha256=digest, matches_original_reference_output=matches,
                        metrics=metrics, input_sha256=hashes,
                        reference_pressure_sha256=expected['reference_pressure_sha256'],
                        reference_rhs_sha256=expected['reference_rhs_sha256'], output_location=location))
                if any(hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()!=hashes[key] for key,value in inputs.items()):
                    raise ValueError('Original input fields changed')
                if index % 32 == 31:
                    save(output / 'results.json', result)
                    print(identifier,index+1,'native equations across correction policies',flush=True)
        if state.data.helpers().state_inventory(state.network) != before or state.model.ARCHITECTURE != architecture:
            raise ValueError('Parameters or original architecture changed')
        if arrays:
            np.savez_compressed(output / 'pressure-policy-outputs.npz', **arrays)
            result['new_output_archive'] = dict(path='pressure-policy-outputs.npz',
                sha256=sha(output/'pressure-policy-outputs.npz'), bytes=(output/'pressure-policy-outputs.npz').stat().st_size,
                fields=len(arrays))
        summary = {}
        for policy in protocol['policies']:
            own = [row for row in result['rows'] if row['policy']==policy]
            summary[policy] = dict(cases=len(own), actual_network_root_calls=sorted({row['actual_network_root_calls'] for row in own}),
                worst={key:max(row['metrics'][key] for row in own) for key in own[0]['metrics']})
        result.update(status='complete_GPU_pressure_correction_work_matrix', finished_unix=time.time(),
            cases=128, emitted_pressure_fields=len(result['rows']), summary=summary,
            original_default_output_parity_cases=parity_count,
            original_source_and_parameters_unchanged=True, new_unique_output_fields=len(arrays))
    except BaseException as error:
        result.update(status='failed_GPU_pressure_correction_work_matrix',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if root_hook is not None:
            root_hook.remove()
        guard.remove()
        save(output/'results.json',result)
    print(identifier,result['status'],result['summary'],flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--identifier',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.protocol.resolve(),args.identifier,args.output.resolve())
