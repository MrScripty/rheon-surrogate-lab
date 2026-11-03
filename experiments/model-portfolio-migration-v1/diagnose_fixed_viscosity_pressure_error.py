"""Separate native-request, pressure-map and output-quantization error.

The actual generator remains unchanged and must reproduce all earlier field
hashes. A separate oracle-fed pressure-component call uses the original future
native RHS solely for diagnosis; it is not a causal generation result. All
neural operations, including construction and oracle calls, remain CUDA-only.
"""
import hashlib
import math
import os
from pathlib import Path
import time

from full_state_gpu_timing import cpu_views,field_sizes,hashes,numpy_outputs
from inherited_generator_cuda import restore
from review_direction_pressure_fits import operator
from verify_joint_generator_retained_probes import relative
from review_water_learning_sequence import ROOT,binding,canonical,read,require

RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = RUN/'fixed-viscosity-pressure-diagnosis-01.json'
FIELDS = ('pressure','vx','vy','vz','tracer')


def decomposition(np,output,internal,oracle,truth):
    def centered(value):
        value = np.asarray(value,dtype=np.float64).reshape(-1)
        return value-value.mean()
    p,i,o,t = map(centered,(output,internal,oracle,truth))
    denominator = float(np.linalg.norm(t))
    parts = dict(output_quantization=p-i,request_propagation=i-o,pressure_map=o-t)
    total = p-t
    residual = total-sum(parts.values())
    require(float(np.linalg.norm(residual)) <= 1e-12+1e-10*max(denominator,float(np.linalg.norm(total))),
            'Pressure error decomposition fails its vector identity')
    def metric(value):
        norm = float(np.linalg.norm(value))
        return dict(relative_l2=norm/denominator if denominator else (0. if norm == 0 else None),
            absolute_rms=norm/math.sqrt(value.size),maximum_absolute=float(np.abs(value).max()))
    return dict(total=metric(total),**{name:metric(value) for name,value in parts.items()},
        decomposition_residual_l2=float(np.linalg.norm(residual)),reference_mean_free_l2=denominator)


def run():
    require(not OUTPUT.exists() and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda',
            'Fresh CUDA diagnostic required; preserve original execution')
    protocol_path = RUN/'inherited-generator-CUDA-qualification-protocol-01.json'
    reduced_path = DOC/'reports/inherited-generator-CUDA-qualification-01.json'
    protocol,reduced = read(protocol_path),read(reduced_path)
    require(reduced['models_qualified'] == 16,'Completed original CUDA family qualification required')
    model = next(m for m in protocol['models'] if m['identifier'] == 'b1-fixed-viscosity')
    prior_spec = reduced['models'][model['identifier']]['raw_result']
    require(binding(ROOT/prior_spec['path']) == prior_spec,'Original64case generator capture changed')
    prior = read(ROOT/prior_spec['path'])
    expected = {(r['identifier'],r['start']):r for r in prior['rows']}
    cases = read(ROOT/protocol['cases']['path'])['cases']
    for spec in protocol['dependencies']:
        require(binding(ROOT/spec['path']) == spec,'Original diagnostic dependency changed')
    os.sched_setaffinity(0,{6})
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(read(ROOT/model['configuration']['path'])['seed'])
    result = dict(status='original_pressure_error_diagnostic_running',pid=os.getpid(),created_unix=time.time(),
        model=model['identifier'],fit_item_id=model['fit_item_id'],rows=[],CPU_neural_forwards=0,
        oracle_fields_are_not_generator_inputs=True,oracle_component_results_are_not_causal_generation=True,
        inference_latency_qualified=False,new_fit=False,new_reference_generation=False,output_arrays_written=False,
        full_portfolio_complete=False,producer=binding(Path(__file__)))
    handles = []
    captured,enabled = {},False
    evidence = None
    try:
        norm = read(ROOT/model['normalization']['path'])
        with restore(model) as (network,evidence):
            def request_hook(owner,args):
                if enabled:
                    require('q' not in captured,'Exactly one generator pressure request expected forK1')
                    captured['q'] = args[0].detach().clone()

            def pressure_hook(owner,args,value):
                if enabled:
                    captured['p'] = value.detach().clone()

            handles = [network.pressure_core.register_forward_pre_hook(request_hook),
                network.pressure_core.register_forward_hook(pressure_hook)]
            for case in cases:
                path = ROOT/case['current']['path']
                require(binding(path)['sha256'] == case['current']['sha256'],'Original current fields changed')
                with np.load(path,allow_pickle=False) as arrays:
                    native = {name:arrays[name].copy() for name in FIELDS}
                initial = hashes(native)
                captured = {}
                enabled = True
                with torch.inference_mode():
                    generated = network.predict(cpu_views(np,torch,native,field_sizes(case['scene']['counts'])),
                        case['scene'],case['start'],norm)
                    owned = [{name:r[name].to('cpu',copy=True).contiguous() for name in FIELDS} for r in generated]
                    returned = numpy_outputs(torch,owned,1,field_sizes(case['scene']['counts']))
                    torch.cuda.synchronize()
                enabled = False
                old = expected[(case['identifier'],case['start'])]
                require(hashes(native) == initial and [hashes(r) for r in returned] == old['complete_output_hashes']
                        and set(captured) == {'q','p'},'Diagnostic hooks changed original complete-generation fields')
                reference = case['reference_leads'][0]
                path = ROOT/reference['path']
                require(binding(path)['sha256'] == reference['sha256'],'Original future reference changed')
                with np.load(path,allow_pickle=False) as arrays:
                    truth = arrays['pressure'].copy()
                    rhs = arrays['rhs'].copy()
                h = 1/case['scene']['cells_per_metre']
                shape = tuple(reversed(case['scene']['counts']))
                q_reference = rhs.astype(np.float64).reshape(-1)*(1000./h)
                q_from_native_pressure = operator(truth.reshape(shape),h,1000.).reshape(-1)*(1000./h)
                units_check = relative(q_from_native_pressure,q_reference,False)
                require(units_check is not None and units_check < 2e-12,'Native RHS to pressure-unit conversion differs from old reference')
                with torch.inference_mode():
                    q_cuda = torch.from_numpy(q_reference.copy()).to('cuda').reshape(1,1,*shape)
                    oracle_pressure = network.pressure_core(q_cuda).detach().cpu().numpy().reshape(-1)
                    torch.cuda.synchronize()
                q_predicted = captured['q'].cpu().numpy().reshape(-1)
                internal_pressure = captured['p'].cpu().numpy().reshape(-1)
                parts = decomposition(np,returned[0]['pressure'],internal_pressure,oracle_pressure,truth)
                require(math.isclose(parts['total']['relative_l2'],old['metrics'][0]['values']['pressure_relative_l2'],rel_tol=1e-10,abs_tol=1e-13),
                        'Diagnostic total differs from original measured generator error')
                result['rows'].append(dict(identifier=case['identifier'],start=case['start'],lead=1,
                    counts_xyz=case['scene']['counts'],cells_per_metre=case['scene']['cells_per_metre'],
                    pressure_route=old['pressure_route'],output_dt_s=1/60,pressure_error=parts,
                    request_relative_l2=relative(q_predicted,q_reference,True),reference_unit_check_relative_l2=units_check,
                    reference=reference,original_complete_output_hashes=old['complete_output_hashes'],
                    actual_request_sha256=hashlib.sha256(q_predicted.tobytes()).hexdigest(),
                    oracle_request_sha256=hashlib.sha256(q_reference.tobytes()).hexdigest()))
            for handle in handles:
                handle.remove()
            handles = []
        require(len(result['rows']) == 64 and evidence['scoped_hooks_removed']
                and evidence['CPU_neural_forwards'] == evidence['forbidden_forward_attempts'] == 0,
                'All64cases and CUDA-only cleanup required')
        worst = max(result['rows'],key=lambda r:r['pressure_error']['total']['relative_l2'])
        result.update(status='all64_original_generator_hashes_and_pressure_error_components_measured',constructor=evidence,
            worst_generator_case={k:worst[k] for k in ('identifier','start','lead','counts_xyz','cells_per_metre','pressure_route')},
            worst_case_pressure_error=worst['pressure_error'],
            maximum_relative_component_errors={name:max(r['pressure_error'][name]['relative_l2'] for r in result['rows'])
                for name in ('total','output_quantization','request_propagation','pressure_map')},
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),GPU=torch.cuda.get_device_name(),
            evidence=[binding(protocol_path),binding(reduced_path),prior_spec,
                binding(Path(__file__).with_name('inherited_generator_cuda.py')),
                binding(Path(__file__).with_name('review_direction_pressure_fits.py'))])
        print('Worst original generator case:',result['worst_generator_case'],flush=True)
        print('Relative error components at that case:',{n:v['relative_l2'] for n,v in worst['pressure_error'].items()
            if isinstance(v,dict) and 'relative_l2' in v},flush=True)
    except BaseException as error:
        result.update(status='original_pressure_diagnostic_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        for handle in handles:
            handle.remove()
        OUTPUT.write_bytes(canonical(result)+b'\n')


if __name__ == '__main__':
    run()
