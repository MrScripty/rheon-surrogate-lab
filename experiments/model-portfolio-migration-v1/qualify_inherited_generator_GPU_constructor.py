"""Untimed CUDA-only constructor pilot for the original K2 endpoint decoder.

The captured trainer/model files are immutable. Scoped process-local hooks move
registered parameters/buffers to CUDA before any neural unit-response query;
the explicit default device owns otherwise implicit matrices, bases and inputs.
GPU-built derived maps are an execution/precision variant, not an exact replay
of the former CPU-built maps. No inference-speed claim is made.
"""
from collections import Counter
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import time

from captured_temporal_review import audit
from full_state_gpu_timing import check_owned, cpu_views, field_sizes, hashes, numpy_outputs
from joint_generator_gpu_benchmark import score
from reduce_sharing_generator_gpu_benchmark import saved_probe_check
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = RUN/'inherited-generator-CUDA-construction-pilot-01'
READINESS = DOC/'reports/temporal-GPU-construction-readiness-01.json'
FIT = 'fit-0a92490241d21911'
FIELDS = ('pressure','vx','vy','vz','tracer')


def run():
    require(not OUTPUT.exists() and os.environ.get('TULDOK_TORCH_DEVICE') == 'cuda',
            'Fresh CUDA pilot required; reconcile any original result without automatic replay')
    OUTPUT.mkdir()
    result = dict(status='original_constructor_pilot_starting',pid=os.getpid(),created_unix=time.time(),
        fit_item_id=FIT,CPU_neural_forwards=0,forbidden_forward_attempts=0,
        neural_calls_by_phase={},registered_CPU_buffers_moved=0,registered_CPU_parameters_moved=0,
        new_parameter_fit=False,new_reference_generation=False,inference_latency_qualified=False,
        setup_resource_observations_only=True,concurrent_native_metadata_work=True,
        precision_variant='GPU-built F64 learned unit-response maps and cosine bases; original saved learned tensor bytes unchanged.',
        bitwise_equivalence_to_original_CPU_derived_maps_claimed=False,output_arrays_written=False,
        native_mutations=0,full_portfolio_complete=False,producer=binding(Path(__file__)))
    handles = []
    started = time.monotonic()
    calls = Counter()
    phase = 'construction'

    def checkpoint():
        result['elapsed_seconds'] = time.monotonic()-started
        (OUTPUT/'result.json').write_bytes(canonical(result)+b'\n')

    checkpoint()
    try:
        readiness = read(READINESS)
        row = next(r for r in readiness['rows'] if r['fit_item_id'] == FIT)
        require(row['identifier'] == 'b2-endpoint-decoder' and not row['full_unit_query_counts'],
                'Exact first constructor family required')
        for spec in [row['receipt'],*row['original_sources']]:
            require(binding(ROOT/spec['path']) == spec,'Readiness source or original receipt changed')
        config_path = ROOT/next(s['path'] for s in row['original_sources'] if Path(s['path']).name == 'config.json')
        project = config_path.parent
        recipe = read(project/'train-recipe.json')
        trainer_path = ROOT/recipe['entrypoint']
        item = dict(project_path=str(project.relative_to(ROOT)),receipt=row['receipt'])
        captured = audit(item,trainer_path.name)
        config,receipt = captured['config'],captured['receipt']
        require(config['horizons'] == 2 and config['architecture'] == 'endpoint','Exact original endpoint decoder required')
        result.update(identifier=row['identifier'],K=2,original_receipt=row['receipt'],
            captured_sources=captured['facts']['sources'],original_weights=binding(captured['train']/'weights.npz'),
            original_readiness=binding(READINESS),source_closure_verified=True)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        os.sched_setaffinity(0,{6})
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(config['seed'])
        module_hooks = torch.nn.modules.module

        def buffer_device(module,name,value):
            if value is not None and value.device.type != 'cuda':
                result['registered_CPU_buffers_moved'] += 1
                return value.to('cuda')

        def parameter_device(module,name,value):
            if value is not None and value.device.type != 'cuda':
                result['registered_CPU_parameters_moved'] += 1
                return torch.nn.Parameter(value.detach().to('cuda'),requires_grad=value.requires_grad)

        def forward_guard(module,args):
            tensors = [v for v in args if isinstance(v,torch.Tensor)]
            local = [*module.parameters(recurse=False),*module.buffers(recurse=False)]
            if not tensors or any(v.device.type != 'cuda' for v in [*tensors,*local]):
                result['forbidden_forward_attempts'] += 1
                raise ValueError('CPU or unclassified neural execution blocked before forward')
            require(time.monotonic()-started < 300 and torch.cuda.max_memory_allocated() <= 2*1024**3,
                    'Bounded300second/2GiB constructor pilot exceeded')
            calls[(phase,type(module).__name__)] += 1

        handles.extend([module_hooks.register_module_buffer_registration_hook(buffer_device),
            module_hooks.register_module_parameter_registration_hook(parameter_device),
            module_hooks.register_module_forward_pre_hook(forward_guard)])
        sys.path.insert(0,str(project))
        spec = importlib.util.spec_from_file_location('captured_endpoint_decoder_trainer',trainer_path)
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        require(binding(trainer_path)['sha256'] == receipt['closure']['sources'][trainer_path.name],
                'Executed trainer changed')
        tick = time.perf_counter()
        with torch.device('cuda'):
            network = trainer.construct_network(config,'cuda',statistics_path=project/'step-scales.json',
                asset_directory=project/'expert-assets')
        torch.cuda.synchronize()
        result['constructor_wall_seconds_under_native_host_load'] = time.perf_counter()-tick
        require(calls[('construction','FrozenSpatial')] == row['fine_root_queries_in_batches_of4'] == 453,
                'All453original fine unit-query batches must actually execute on CUDA')
        require(all(v.device.type == 'cuda' for v in [*network.parameters(),*network.buffers()]),
                'Every parameter and derived/constant buffer must be CUDA after construction')
        trainer.load_weights(network,captured['train']/'weights.npz',receipt['final_inventory'])
        require(trainer.inventory(network) == receipt['final_inventory'],'Original learned final tensor bytes changed')
        network.eval()
        maps = []
        for name,value in network.named_buffers():
            if name.endswith('_fine_matrix'):
                require(value.dtype == torch.float64 and value.device.type == 'cuda' and bool(torch.isfinite(value).all()),
                        'Finite CUDAF64derived unit map required')
                maps.append(dict(name=name,shape=list(value.shape),bytes=value.numel()*value.element_size(),
                    sha256=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()))
        require(sum(m['bytes'] for m in maps) == row['fine_derived_matrix_bytes'],
                'Exact original derived-map extent required')
        result.update(all_learned_final_tensors_bit_exact=True,derived_GPU_maps=maps,
            constructor_CUDA_FrozenSpatial_calls=453,all_parameters_and_buffers_CUDA=True)
        case_path = DOC/'reports/joint-generator-case-preflight-01.json'
        prepared = read(case_path)
        case = next(c for c in prepared['cases'] if c['identifier'] == 'temporal-g0-cpm8-r0' and c['start'] == 0)
        require(binding(ROOT/case['current']['path'])['sha256'] == case['current']['sha256'],
                'Original current native fields changed')
        with np.load(ROOT/case['current']['path'],allow_pickle=False) as arrays:
            native = {name:arrays[name].copy() for name in FIELDS}
        initial = hashes(native)
        sizes = field_sizes(case['scene']['counts'])
        rows,previous_hashes,retained = [],None,None
        phase = 'prediction'
        for repetition in range(2):
            with torch.inference_mode(),torch.device('cuda'):
                generated = network.predict(cpu_views(np,torch,native,sizes),case['scene'],case['start'],captured['norm'])
                cpu = [{name:r[name].to('cpu',copy=True).contiguous() for name in FIELDS} for r in generated]
                returned = numpy_outputs(torch,cpu,2,sizes)
                torch.cuda.synchronize()
            require(all(np.isfinite(r[name]).all() for r in returned for name in FIELDS)
                    and hashes(native) == initial,'Nonfinite output or modified current fields')
            check_owned(np,returned,native,retained)
            actual_hashes = [hashes(r) for r in returned]
            if previous_hashes is not None:
                require(actual_hashes == previous_hashes,'Repeated complete outputs changed bytes')
            else:
                for lead,output in enumerate(returned,1):
                    reference = case['reference_leads'][lead-1]
                    require(binding(ROOT/reference['path'])['sha256'] == reference['sha256'],'Original physical reference changed')
                    with np.load(ROOT/reference['path'],allow_pickle=False) as arrays:
                        truth = {name:arrays[name] for name in (*FIELDS,'rhs')}
                        rows.append(dict(lead=lead,time_s=reference['time_s'],values=score(np,output,truth,case['scene'])))
            previous_hashes,retained = actual_hashes,returned
        document = read(project/'project.json')
        norm_path = ROOT/next(r['path'] for r in document['data'] if r['id'] == 'normalization')
        model = dict(identifier=row['identifier'],receipt=row['receipt'],K=2,normalization=binding(norm_path))
        comparison = saved_probe_check(model,prepared,dict(rows=[dict(identifier=case['identifier'],start=0,metrics=rows)]))
        require(trainer.inventory(network) == receipt['final_inventory'] and result['forbidden_forward_attempts'] == 0,
                'Learned final tensors or neural device boundary changed during prediction')
        for source in captured['facts']['sources']:
            require(binding(ROOT/source['path']) == source,'Captured source changed during constructor qualification')
        result.update(status='one_inherited_constructor_CUDA_setup_and_first_case_qualified',
            original_saved_probe_metric_comparison=comparison,physical_case=dict(identifier=case['identifier'],start=0,
                counts_xyz=case['scene']['counts'],cells_per_metre=case['scene']['cells_per_metre'],output_dt_s=1/60),
            complete_output_hashes=previous_hashes,complete_all_field_metrics=rows,repeat_and_ownership_checks_passed=True,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),GPU=torch.cuda.get_device_name(),
            torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            setup_qualification_only=True,all64case_benchmark_complete=False,all16constructors_qualified=False,
            supporting_helpers=[binding(Path(__file__).with_name(n)) for n in
                ('captured_temporal_review.py','full_state_gpu_timing.py','joint_generator_gpu_benchmark.py',
                 'reduce_sharing_generator_gpu_benchmark.py')])
        print('CUDA constructor pilot passed:453 original unit-query batches,exact final tensors,allK2fields and saved probe metrics; no speed claim.',flush=True)
    except BaseException as error:
        result.update(status='failed_original_constructor_pilot_requires_reconciliation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        for handle in reversed(handles):
            handle.remove()
        result['neural_calls_by_phase'] = {p:{kind:count for (stage,kind),count in calls.items() if stage == p}
            for p in {stage for stage,_ in calls}}
        result['scoped_global_hooks_removed'] = True
        checkpoint()


if __name__ == '__main__':
    run()
