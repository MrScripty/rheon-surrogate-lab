"""Reproduce original CUDA training-residual probes, without a field benchmark.

QR designs contain supervised targets. They are inputs to the original training
objective diagnostic, never evidence of causal full-field inference accuracy.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, DOC, RUN, read, save, binding, require
from component_material import observed_file


PROTOCOL = RUN/'macro-readout-original-GPU-probe-replay-protocol-01.json'
OUTPUT = DOC/'reports/macro-readout-original-GPU-probe-replay-01.json'


def prepare():
    require(not PROTOCOL.exists(), 'Preserve original probe replay registration')
    contract_path = RUN/'macro-readout-component-contract-01.json'
    contract = read(contract_path)
    require(contract['active_full_field_readout_windows_per_fit'] == 58 and contract['fixed_branch_windows'] == 238,
            'Original active/bypass populations must already be admitted')
    material_path = RUN/'remaining-component-material-01.json'
    entries = {row['fit_item_id']: row for row in read(material_path)['models']}
    models = []
    for fid in contract['fit_ids']:
        entry = entries[fid]
        receipt = read(ROOT/entry['original_receipt']['path'])
        source = next(row for row in entry['sources'] if row['original_name'] == 'macrostep_readout_v1.py')
        probe_path = (ROOT/entry['outputs']['weights.npz']['path']).parent/'probe.npy'
        probe = observed_file(ROOT,str(probe_path.relative_to(ROOT)),receipt['outputs']['probe.npy'])
        stages = read(ROOT/entry['original_configuration']['path'])['stages']
        specs = dict(source=source,weights=entry['outputs']['weights.npz'],design=entry['inputs']['part_00'],
            receipt=entry['original_receipt'],history=entry['outputs']['history.json'],probe=probe)
        for spec in specs.values():
            observed_file(ROOT,spec['path'],spec)
        models.append(dict(fit_item_id=fid,stages=stages,inventory=entry['selected_inventory'],**specs))
    runtime_path = RUN.parent/'dependency-migration-v1/runtime/current-manifest-v2.json'
    save(PROTOCOL,dict(status='registered_before_original_macro_readout_GPU_probe_replay',created_unix=time.time(),
        producer=binding(Path(__file__)),contract=binding(contract_path),material=binding(material_path),
        runtime=binding(runtime_path),models=models,expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',
        repeat_calls_per_fit=2,primary_timing_samples=0,latency_claim=False,
        physical_field_accuracy_claim=False,CPU_neural_forwards_allowed=False,
        scope=__doc__.strip(),new_field_features_generated=0,default_coverage_increased=False))
    print('Original two readout CUDA probe replay registered; no field or latency claim.',flush=True)


def run():
    require(not OUTPUT.exists(), 'Preserve original replay outcome')
    protocol = read(PROTOCOL)
    require(binding(Path(__file__)) == protocol['producer']
            and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda', 'Exact preregistered CUDA worker required')
    for spec in [protocol['runtime'],protocol['contract'],protocol['material']]:
        observed_file(ROOT,spec['path'],spec)
    import numpy as np
    import torch
    from cuda_forward_observer import CUDAForwardObserver
    runtime = read(ROOT/protocol['runtime']['path'])
    require(torch.__version__ == runtime['packages']['torch'] and np.__version__ == runtime['packages']['numpy']
            and torch.cuda.is_available() and torch.cuda.get_device_name() == protocol['expected_GPU'],
            'Original pinned managed CUDA runtime/device required')
    os.sched_setaffinity(0,{4})
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    observer = CUDAForwardObserver(torch);observer.enable('original_readout_direct_method_replay')
    result = dict(status='original_macro_readout_GPU_probe_replay_running',created_unix=time.time(),
        pid=os.getpid(),protocol=binding(PROTOCOL),producer=binding(Path(__file__)),models=[],
        GPU=torch.cuda.get_device_name(),inference_device='cuda',CPU_neural_forwards=0,latency_claim=False,
        physical_field_accuracy_qualified=False,default_coverage_increased=False,
        field_feature_generation_calls=0,project_dataset_or_weight_copies=0,new_parameter_fits=0,
        scope=__doc__.strip())
    save(OUTPUT,result)
    try:
        with torch.inference_mode():
            for item in protocol['models']:
                for key in ('source','weights','design','receipt','history','probe'):
                    observed_file(ROOT,item[key]['path'],item[key])
                spec = importlib.util.spec_from_file_location('captured_macro_readout_'+str(item['stages']),ROOT/item['source']['path'])
                module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                model = module.MacroReadout(item['stages'],device='cuda')
                require(all(p.device.type == 'cuda' for p in model.parameters()), 'Constructor parameters must be CUDA')
                with np.load(ROOT/item['weights']['path'],allow_pickle=False) as archive:
                    require(set(archive.files) == set(item['inventory']), 'Exact selected coefficient membership required')
                    state = {}
                    for name,inventory in item['inventory'].items():
                        value = archive[name]
                        require(value.dtype.str == inventory['dtype'] and list(value.shape) == inventory['shape']
                                and hashlib.sha256(value.tobytes()).hexdigest() == inventory['sha256'],
                                'Exact original coefficient bytes required')
                        state[name] = torch.from_numpy(value.copy()).to('cuda')
                    model.load_state_dict(state,strict=True)
                model.eval().requires_grad_(False)
                with np.load(ROOT/item['design']['path'],allow_pickle=False) as archive:
                    require(set(archive.files) == {'design','scene_index','teacher_substeps','starts'}, 'Original complete QR input required')
                    design = archive['design'].copy();teacher = archive['teacher_substeps'].copy()
                    require(np.array_equal(archive['scene_index'],np.arange(148))
                            and np.array_equal(archive['starts'],np.array([32,96]))
                            and design.dtype.str == '<f8' and design.shape == (148,2,4,item['stages']+1,item['stages']+1)
                            and np.isfinite(design).all() and np.count_nonzero(teacher == 8) == 29
                            and np.count_nonzero(teacher == 1) == 119, 'Original all148parents/two-window view required')
                fixed = teacher == 1
                require(np.count_nonzero(design[fixed,...,:-1]) == 0, 'Fixed-branch residuals must not depend on learned coefficients')
                resident = torch.from_numpy(design).to('cuda')
                saved = np.load(ROOT/item['probe']['path'],allow_pickle=False)
                require(saved.dtype.str == '<f8' and saved.shape == (148,2,4,item['stages']+1), 'Original typed GPU probe required')
                outputs = []
                direct_calls = []
                for repeat in range(protocol['repeat_calls_per_fit']):
                    require(resident.device.type == 'cuda' and resident.dtype == torch.float64
                            and all(p.device.type == 'cuda' and p.dtype == torch.float64 for p in model.parameters()),
                            'Direct .residuals method needs explicit operand/parameter device guards')
                    predicted = model.residuals(resident)
                    require(predicted.device.type == 'cuda' and predicted.dtype == torch.float64,
                            'Actual original direct method must return CUDA F64')
                    objective = float(predicted.square().sum(dim=-1).mean())
                    values = predicted.cpu().numpy().copy();torch.cuda.synchronize()
                    outputs.append(values)
                    direct_calls.append(dict(method='MacroReadout.residuals',repeat=repeat,
                        input_device='cuda',parameter_device='cuda',output_device='cuda',dtype='float64',objective=objective))
                require(np.array_equal(outputs[0],outputs[1]), 'Repeated original CUDA residual probes differ')
                for name,inventory in item['inventory'].items():
                    actual = model.state_dict()[name].detach().cpu().numpy()
                    require(hashlib.sha256(actual.tobytes()).hexdigest() == inventory['sha256'], 'Original coefficient bytes changed')
                values = outputs[0];difference = values-saved
                relative = float(np.linalg.norm(difference.ravel())/max(float(np.linalg.norm(saved.ravel())),1e-30))
                squared = np.square(values).sum(axis=-1)
                summaries = {}
                for name,mask in [('all',np.ones(148,dtype=bool)),('learned_active_N8',teacher == 8),('fixed_bypass_N1',fixed)]:
                    selected = squared[mask]
                    summaries[name] = dict(scenes=int(np.count_nonzero(mask)),windows=int(np.count_nonzero(mask))*2,
                        mean_original_QR_squared_relative_residual=float(selected.mean()),
                        worst_original_QR_relative_by_metric=dict(zip(('equation','pressure','velocity','tracer'),
                            map(float,np.sqrt(selected.max(axis=(0,1)))))))
                history = read(ROOT/item['history']['path'])
                result['models'].append(dict(fit_item_id=item['fit_item_id'],stages=item['stages'],
                    parameters=3*item['stages'],original_source=item['source'],original_probe=item['probe'],
                    output_array_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                    saved_probe_array_sha256=hashlib.sha256(saved.tobytes()).hexdigest(),
                    saved_GPU_probe_bitwise_equal=bool(np.array_equal(values,saved)),
                    maximum_probe_absolute_difference=float(np.max(np.abs(difference))),probe_relative_l2=relative,
                    repeated_outputs_exact=True,raw_published_coefficients_unchanged=True,
                    original_history_last_objective=history['history'][-1]['objective'],
                    current_checkpoint_GPU_objective=direct_calls[0]['objective'],direct_method_calls=direct_calls,
                    original_QR_residual_summaries=summaries))
                save(OUTPUT,result)
        require(observer.forbidden_attempts == 0, 'CPU/unclassified module execution observed')
        result.update(status='two_original_readout_GPU_probes_reproduced_exactly' if
            all(row['saved_GPU_probe_bitwise_equal'] for row in result['models']) else
            'original_readout_GPU_probe_differences_observed_not_exact',finished_unix=time.time(),
            direct_CUDA_method_calls=sum(len(row['direct_method_calls']) for row in result['models']),
            module_forward_hooks_do_not_observe_these_direct_methods=True,
            preserved_original_probe_reused_without_another_array_copy=True,
            full58_active_field_windows_per_fit_still_required=True)
    except BaseException as error:
        result.update(status='original_macro_readout_GPU_probe_replay_failed',error_type=type(error).__name__)
        raise
    finally:
        observer.close();result['module_forward_guard'] = observer.evidence();save(OUTPUT,result)
    print(result['status'],[(row['fit_item_id'],row['probe_relative_l2']) for row in result['models']],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','run'))
    if parser.parse_args().stage == 'prepare':
        prepare()
    else:
        run()
