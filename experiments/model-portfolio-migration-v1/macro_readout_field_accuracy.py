"""Original causal CUDA backbone plus raw published readouts on all58windows.

No primary latency is collected. Saved full fields support independent NumPy
scoring; original supervised QR reconstruction is an offline diagnostic only.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import resource
import sys
import time

from import_common_generator_measurements import ROOT, RUN, read, save, binding, require
from component_material import current_publication, observed_file
from cuda_forward_observer import CUDAForwardObserver
from macro_readout_fields import FIELDS, digest, load_fields
from modal_component_benchmark import verify_sources
from pressure_portfolio_gpu_timing import resources


def run(protocol_path,phase):
    current_publication(ROOT)
    protocol = read(protocol_path)
    require(os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda' and phase in (0,1), 'Explicit original CUDA accuracy phase required')
    verify_sources(protocol)
    folder = ROOT/protocol['output_root']/('phase'+str(phase))
    require(folder.is_relative_to(RUN) and not folder.exists(), 'Fresh field phase required; no retry')
    if phase == 1:
        prior = read(folder.parent/'phase0/result.json')
        require(prior['status'] == 'all58windows_per_original_readout_GPU_fields_collected'
                and not Path('/proc',str(prior['pid'])).exists(), 'Original phase0 must complete before fresh phase1')
    else:
        prior = None
    folder.mkdir(parents=True)
    os.sched_setaffinity(0,{protocol['host_cpu']})
    began = time.monotonic()
    result = dict(status='original_macro_field_accuracy_running',created_unix=time.time(),pid=os.getpid(),
        phase=phase,protocol=binding(protocol_path),models=[],resource_observations=[resources()],
        inference_device='cuda',CPU_neural_forwards=0,latency_qualified=False,primary_timing_samples=0,
        new_parameter_fits=0,new_reference_generation=False,model_dataset_or_project_copies=0,
        references_passed_to_causal_model=False,full_portfolio_complete=False)
    save(folder/'result.json',result)
    backbone = observer = None
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        require(torch.cuda.is_available() and torch.cuda.get_device_name() == protocol['expected_GPU']
                and not torch.is_autocast_enabled('cuda'), 'Actual admitted CUDA hardware/precision required')
        observer = CUDAForwardObserver(torch);observer.enable('constructor')
        sys.path.insert(0,str(ROOT/'experiments/full-ai-data-diversity-v1'))
        import support
        from macrostep_backbone_v1 import FrozenMacroBackbone
        from diagnose_causal_composition import inventory, measure
        from macrostep_loss_design_v1 import design as original_design
        with torch.device('cuda'),torch.no_grad():
            backbone = FrozenMacroBackbone(*protocol['backbone_arguments'])
        require(all(value.device.type == 'cuda' for value in [*backbone.model.parameters(),*backbone.model.buffers()]),
                'All constructor model parameters/buffers must be CUDA before any state-dependent query')
        frozen = inventory(backbone)
        heads = {}
        for item in protocol['models']:
            spec = importlib.util.spec_from_file_location('captured_field_readout_s'+str(item['stages']),ROOT/item['source']['path'])
            module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module;spec.loader.exec_module(module)
            head = module.MacroReadout(item['stages'],device='cuda')
            with np.load(ROOT/item['weights']['path'],allow_pickle=False) as archive:
                require(set(archive.files) == set(item['inventory']), 'Original selected readout tensor membership required')
                state = {}
                for name,description in item['inventory'].items():
                    value = archive[name]
                    require(value.dtype.str == description['dtype'] == '<f8' and list(value.shape) == description['shape']
                            and digest(value) == description['sha256'], 'Raw original learned readout coefficients differ')
                    state[name] = torch.from_numpy(value.copy()).to('cuda')
                head.load_state_dict(state,strict=True)
            heads[item['fit_item_id']] = head.eval().requires_grad_(False)
        norm = read(ROOT/protocol['normalization']['path'])
        contract = read(ROOT/protocol['contract']['path'])
        cases = list(enumerate(contract['cases']))
        models = protocol['models']
        if phase:
            cases = list(reversed(cases));models = list(reversed(models))
        receipts = {}
        result.update(GPU=torch.cuda.get_device_name(),Torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            original_frozen_backbone_identity=backbone.asset_identity,original_frozen_tensor_inventory=frozen)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for item in models:
                fid,stages = item['fit_item_id'],item['stages']
                head = heads[fid]
                require(inventory(head) == item['inventory'], 'Raw original readout state must remain exact')
                model_result = dict(fit_item_id=fid,stages=stages,weights=item['weights'],rows=[],
                    physical_outputs=58,actual_internal_advances=58*stages,readout_coefficient_quantization_applied=False)
                result['models'].append(model_result)
                predictions = {}
                with np.load(ROOT/item['original_design']['path'],allow_pickle=False) as archive:
                    original_factors = archive['design'].copy()
                    require(np.array_equal(archive['starts'],np.array([32,96])), 'Original training windows required')
                previous = next(row for row in prior['models'] if row['fit_item_id'] == fid) if prior else None
                prior_rows = {row['case_index']:row for row in previous['rows']} if previous else {}
                for position,(index,case) in enumerate(cases):
                    require(time.monotonic()-began < protocol['worker_deadline_seconds'], 'Finite accuracy-phase deadline exceeded')
                    if position%14 == 0:
                        result['resource_observations'].append(resources())
                    if case['receipt'] not in receipts:
                        receipts[case['receipt']] = read(ROOT/case['receipt'])
                    scene = receipts[case['receipt']]['scene']
                    require(scene['counts'] == case['counts_xyz'] and scene['cells_per_metre'] == case['cells_per_metre']
                            and case['teacher_substeps'] == 8 and case['trainable_readout_active'],
                            'Only original active N8readout windows enter causal field evaluation')
                    native = load_fields(case['current_state'])
                    inputs = {name:torch.from_numpy(native[name]).to('cuda') for name in FIELDS}
                    require(set(inputs) == set(FIELDS) and all(value.device.type == 'cuda' for value in inputs.values()),
                            'Only the complete original current state may enter the CUDA backbone')
                    observer.set_phase('causal_feature_generation')
                    features = backbone.feature_states(inputs,scene,case['start'],stages,1)[0]
                    require(len(features) == stages+1 and all(set(row) == set(FIELDS) for row in features)
                            and all(value.device.type == 'cuda' for row in features for value in row.values()),
                            'All S+1causal stage fields must exist on CUDA')
                    require(all(p.device.type == 'cuda' and p.dtype == torch.float64 for p in head.parameters()),
                            'Direct readout method parameters must remain CUDA F64')
                    observer.set_phase('direct_CUDA_readout')
                    emitted = head.read(features)
                    require(set(emitted) == set(FIELDS) and all(value.device.type == 'cuda' for value in emitted.values())
                            and all(value.dtype == (torch.float64 if name == 'pressure' else torch.float32)
                                    for name,value in emitted.items()), 'Original full-field output ABI required')
                    cpu = {name:value.detach().to('cpu',copy=True) for name,value in emitted.items()}
                    torch.cuda.synchronize()
                    arrays = {name:value.numpy().copy() for name,value in cpu.items()}
                    require(all(value.ndim == 1 and np.isfinite(value).all() for value in arrays.values())
                            and arrays['pressure'][0] == 0., 'Finite original cell-zero-gauge fields required')
                    outputs = {name:digest(value) for name,value in arrays.items()}
                    # Only now open the future archive. It is never an input to
                    # feature generation or the published readout method.
                    truth = load_fields(case['target_state'],(*FIELDS,'rhs'))
                    observer.set_phase('fixed_original_scoring_and_QR_diagnostic')
                    measured = measure(cpu,truth,scene,norm)
                    qr,stats = original_design(features,truth,scene,norm,stages,True,5e-12,1e-28)
                    qr_array = qr.cpu().numpy().copy()
                    training_factor = original_factors[case['scene_index'],[32,96].index(case['start'])]
                    require(qr_array.shape == training_factor.shape, 'Original full-grid QR diagnostic shape required')
                    differences = qr_array-training_factor
                    row = dict(case_index=index,identity={key:case[key] for key in
                        ('identifier','scene_index','counts_xyz','cells_per_metre','start','teacher_substeps')},
                        current_archive=dict(path=case['current_state']['path'],sha256=case['current_state']['sha256']),
                        target_archive=dict(path=case['target_state']['path'],sha256=case['target_state']['sha256']),
                        input_hashes={name:digest(value) for name,value in native.items()},output_hashes=outputs,
                        original_Torch_physical_scores=measured,targets_loaded_after_causal_output=True,
                        direct_CUDA_readout_calls=1,causal_internal_advances=stages,physical_output_dt_s=1/60,
                        original_QR_factor_bitwise_equal=bool(np.array_equal(qr_array,training_factor)),
                        original_QR_factor_maximum_absolute_difference=float(np.max(np.abs(differences))),
                        original_zero_readout_metric_controls=stats)
                    if prior_rows:
                        require(outputs == prior_rows[index]['output_hashes']
                                and row['input_hashes'] == prior_rows[index]['input_hashes'], 'Fresh-process causal fields differ')
                    else:
                        predictions.update({f'case{index:03d}_{name}':value for name,value in arrays.items()})
                    model_result['rows'].append(row)
                    save(folder/'result.json',result)
                    require(torch.cuda.max_memory_allocated() <= protocol['maximum_CUDA_allocated_bytes']
                            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= protocol['maximum_RSS_bytes'],
                            'Registered accuracy resource ceilings exceeded')
                require(len(model_result['rows']) == 58 and inventory(head) == item['inventory'], 'All active windows and unchanged raw head required')
                if phase == 0:
                    archive_path = folder/(fid+'-fields.npz')
                    np.savez_compressed(archive_path,**predictions)
                    model_result['prediction_archive'] = dict(**binding(archive_path),bytes=archive_path.stat().st_size)
                    del predictions
                else:
                    model_result['prediction_archive'] = previous['prediction_archive']
            require(inventory(backbone) == frozen, 'Causal feature generation changed frozen learned tensors')
        from python_source_coverage import collect
        coverage = collect(sys.modules,ROOT)
        require(all(name in protocol['source_sha256'] for name in coverage['lab_python']), 'Unbound actual imported macro source')
        verify_sources(protocol)
        result.update(status='all58windows_per_original_readout_GPU_fields_collected',source_coverage=coverage,
            learned_backbone_and_raw_heads_unchanged=True,all_model_parameters_and_buffers_CUDA=True,
            total_physical_outputs=116,total_causal_internal_advances=406,direct_CUDA_readout_calls=116,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
            maximum_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            independent_scoring_pending=True,all_QR_factor_diagnostics_bitwise_equal=all(
                row['original_QR_factor_bitwise_equal'] for model in result['models'] for row in model['rows']))
    except BaseException as error:
        result.update(status='original_macro_field_accuracy_failed_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        try:
            if backbone is not None:
                backbone.close()
            if observer is not None:
                observer.close();result['module_forward_guard'] = observer.evidence()
            result['cleanup_complete'] = True
        finally:
            result['resource_observations'].append(resources())
            result['finished_unix'] = time.time();save(folder/'result.json',result)
    print('Original readout field phase',phase,':116physical outputs;406causal internal advances;no latency claim.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--phase',type=int,required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(),args.phase)
