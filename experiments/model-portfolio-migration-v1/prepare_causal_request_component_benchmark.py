"""Freeze both original one-step q heads and the matched2048transition GPU plan."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication, observed_file


def run():
    publication,_ = current_publication(ROOT)
    path = RUN/'causal-request-component-GPU-protocol-01.json';output = RUN/'causal-request-component-GPU-benchmark-01'
    require(not path.exists() and not output.exists(), 'Fresh original causal q benchmark required')
    contract_path = RUN/'causal-request-component-contract-01.json';contract = read(contract_path)
    sampling_path = RUN/'causal-request-sampling-membership-01.json';sampling = read(sampling_path)
    require(sampling['status'] == 'both_original_main_q_sampling_sequences_hash_verified', 'Verified actual sampling membership required')
    material = read(ROOT/contract['material']['path']);sources = {};helpers = None
    for source in Path(__file__).parent.glob('*.py'):
        spec = binding(source);sources[spec['path']] = spec['sha256']
    for spec in [binding(contract_path),contract['material'],binding(sampling_path)]:
        observed_file(ROOT,spec['path'],spec);sources[spec['path']] = spec['sha256']
    fits = []
    for selected in contract['fits']:
        fid = selected['fit_item_id'];entry = next(row for row in material['models'] if row['fit_item_id'] == fid)
        current = {spec['original_name']:spec['sha256'] for spec in entry['sources']
            if spec['original_name'] in ('pressure_request_data.py','temporal_codec.py','temporal_data.py','temporal_features_v2.py')}
        require(len(current) == 4 and (helpers is None or current == helpers), 'Identical original decoder/forcing helper bodies required')
        helpers = current
        for spec in [entry['original_receipt'],entry['original_configuration'],*entry['sources'],*entry['outputs'].values(),*entry['inputs'].values()]:
            observed_file(ROOT,spec['path'],spec);sources[spec['path']] = spec['sha256']
        fits.append(dict(fit_item_id=fid,nn_conv3d_calls_per_forward=5 if entry['serialized_tensor_elements'] == 3936 else 4))
    prior = read(RUN/'sampler-component-GPU-protocol-01.json')
    for name,sha in prior['source_sha256'].items():
        if Path(name).is_absolute():
            sources[name] = sha
    runtime = ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json'
    save(path,dict(schema='rheon.causal-request-component-GPU-benchmark.v1',created_unix=time.time(),
        contract=binding(contract_path),sampling_membership=binding(sampling_path),material=contract['material'],publication_acceptance=binding(publication),
        runtime_manifest=binding(runtime),source_sha256=sources,fits=fits,
        phase_order=[dict(fit_item_id=fit['fit_item_id'],phase=phase) for fit in fits for phase in (0,1)],
        output_root=str(output.relative_to(ROOT)),worker='experiments/model-portfolio-migration-v1/causal_request_component_benchmark.py',
        cases_per_fit=2048,parents=16,states_per_parent=129,horizons=1,phase_case_order='0..2047then2047..0',
        expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,threads=1,
        warmups_per_case=5,samples_per_case=10,calibration_pairs_per_case=4,worker_deadline_seconds=1500,aggregate_deadline_seconds=6600,
        minimum_free_disk_bytes=1024**3,maximum_CUDA_allocated_bytes=2*1024**3,maximum_RSS_bytes=8*1024**3,
        prediction_dtype='float32',autocast=False,tf32=False,deterministic=True,primary_observer_removed=True,
        timing_units='ms/normalized_future_q_request',physical_simulation_steps=0,
        timing_scope=contract['timing_scope'],population='2048retained training-parent transitions. Actualmain-loss sampled membership ishash-replayed;112windows lieoutsideoriginalcurrentstate0..120sampler support. Allremain same-parent correlated diagnostics,not independentholdout.',precision_limit=contract['precision_limit'],
        original_data_policy='CapturedoriginalCPUdecoder andF32normalization usedexactly;originalmake_features computesknownforces onCPU evenwhenmodel training usesCUDA. Targets stayseparate. No model/objective/training module imported fordata preparation.',
        independent_reduction_policy='Use the authoritative capturedTorchCPU decoder to preserveoriginalF32rounding;compute all metricsindependently withNumPy. Globalforwardguard rejectsCPUmodel execution;no model orobjective module imported byreducer.',
        observer_overhead_subtracted=False,resource_scope='Before/every256cases/after snapshots;not continuous exclusivity',
        CPU_neural_forward_allowed=False,new_parameter_fits=0,new_reference_generation=False,weight_or_dataset_copies=0,
        producer=binding(Path(__file__)),full_portfolio_complete=False))
    print('Twooriginalqheads/2048transitions/fourCUDAphases frozen;no inference.',flush=True)


if __name__ == '__main__':
    run()
