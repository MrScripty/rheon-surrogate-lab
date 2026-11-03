"""Scoped CUDA restoration of captured inherited-pressure generators.

The exact original trainer constructs the model and restores its saved weights.
Registration hooks place implicit CPU assets on CUDA before any unit-response
query. Derived maps/bases are explicitly GPU-built precision variants. Each
caller owns a fresh process, so captured imports and class map caches are fresh.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

from review_water_learning_sequence import ROOT, binding, read, require


def route(component,counts_xyz):
    """Name the captured buffer/CNN branch actually selected by this geometry."""
    counts = tuple(reversed(counts_xyz))
    if hasattr(component,'routes'):
        require(counts in component.routes,'Complete pressure map has no captured geometry')
        return 'complete_F64_unit_map' if component.routes[counts] is not None else 'core_CNN_fallback'
    require(counts in component.geometry,'Fine pressure core has no captured geometry')
    prefix,_ = component.geometry[counts]
    return 'fine_F64_unit_map' if hasattr(component,prefix+'_fine_matrix') else 'fine_F32_CNN'


@contextmanager
def restore(model):
    """Yield (original model, device/setup evidence), keeping guards until exit."""
    import torch
    project = ROOT/model['project_path']
    receipt = read(ROOT/model['receipt']['path'])
    config = read(ROOT/model['configuration']['path'])
    trainer_path = ROOT/model['trainer']['path']
    for spec in [model['receipt'],model['configuration'],model['trainer'],model['weights'],*model['sources']]:
        require(binding(ROOT/spec['path']) == spec,'Captured CUDA restoration input changed')
    require(not any(n.startswith('bundle_model') or n in ('frozen_temporal_pressure_core','full_learned_pressure_map')
                    for n in sys.modules),'Fresh process without a previous captured model required')
    observed = dict(phase='construction',CPU_neural_forwards=0,forbidden_forward_attempts=0,
        CPU_buffers_moved=0,CPU_parameters_moved=0,neural_calls_by_phase={},scoped_hooks_removed=False,
        precision_variant='CUDA-built F64 unit-response maps/bases; original learned F32final tensors preserved.',
        bitwise_CPU_derived_map_equivalence_qualified=False)
    calls = Counter()
    handles = []
    begun = time.monotonic()

    def buffer_device(owner,name,value):
        if value is not None and value.device.type != 'cuda':
            observed['CPU_buffers_moved'] += 1
            return value.to('cuda')

    def parameter_device(owner,name,value):
        if value is not None and value.device.type != 'cuda':
            observed['CPU_parameters_moved'] += 1
            return torch.nn.Parameter(value.detach().to('cuda'),requires_grad=value.requires_grad)

    def guard(owner,args):
        tensors = [v for v in args if isinstance(v,torch.Tensor)]
        local = [*owner.parameters(recurse=False),*owner.buffers(recurse=False)]
        if not tensors or any(v.device.type != 'cuda' for v in [*tensors,*local]):
            observed['forbidden_forward_attempts'] += 1
            raise ValueError('CPU/unclassified neural execution blocked before forward')
        require(time.monotonic()-begun < 600 and torch.cuda.max_memory_allocated() <= 2*1024**3,
                'Constructor/accuracy worker exceeds600seconds or2GiBCUDA')
        calls[(observed['phase'],type(owner).__name__)] += 1

    module = torch.nn.modules.module
    try:
        handles.extend([module.register_module_buffer_registration_hook(buffer_device),
            module.register_module_parameter_registration_hook(parameter_device),
            module.register_module_forward_pre_hook(guard)])
        sys.path.insert(0,str(project))
        spec = importlib.util.spec_from_file_location('captured_inherited_trainer',trainer_path)
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        full = sys.modules.get('full_learned_pressure_map')
        require(full is None or not full.FullLearnedPressureMap._cache,'Fresh complete-map class cache required')
        tick = time.perf_counter()
        with torch.device('cuda'):
            network = trainer.construct_network(config,'cuda',statistics_path=project/'step-scales.json',
                asset_directory=project/'expert-assets')
            trainer.load_weights(network,ROOT/model['weights']['path'],receipt['final_inventory'])
        torch.cuda.synchronize()
        observed['constructor_and_restore_wall_seconds_under_host_load'] = time.perf_counter()-tick
        require(trainer.inventory(network) == receipt['final_inventory'],'Original final learned tensor inventory differs')
        require(all(v.device.type == 'cuda' for v in [*network.parameters(),*network.buffers()]),
                'All constructed parameters and constant/derived buffers must be CUDA')
        readiness = model['readiness']
        core = read(project/'expert-assets/pressure/configuration.json')
        extra = sum((n+3)//4 for n in readiness['full_unit_query_counts'] if n > 2048)*core['steps']
        expected = readiness['fine_root_queries_in_batches_of4']+extra
        require(calls[('construction','FrozenSpatial')] == expected,
                'Captured fine/full-map unit-query CUDA work differs')
        derived = []
        for name,value in network.named_buffers():
            if name.endswith(('_fine_matrix','_full_matrix')):
                require(value.dtype == torch.float64 and bool(torch.isfinite(value).all()),'Finite CUDAF64unit map required')
                derived.append(dict(name=name,shape=list(value.shape),bytes=value.numel()*8,
                    sha256=hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()))
        require(sum(v['bytes'] for v in derived if v['name'].endswith('_fine_matrix')) == readiness['fine_derived_matrix_bytes']
                and sum(v['bytes'] for v in derived if v['name'].endswith('_full_matrix')) == readiness['full_derived_matrix_bytes'],
                'Derived fine/full map extents differ from captured construction')
        observed.update(expected_constructor_FrozenSpatial_calls=expected,derived_GPU_maps=derived,
            all_parameters_and_buffers_CUDA=True,original_final_tensor_inventory_exact=True)
        # Later prediction does not need registration interception. The forward
        # device guard remains installed through every physical prediction.
        handles[0].remove()
        handles[1].remove()
        observed['phase'] = 'prediction'
        network.eval()
        with torch.device('cuda'):
            yield network,observed
        require(trainer.inventory(network) == receipt['final_inventory'],'Prediction changed final learned tensors')
        for spec in model['sources']:
            require(binding(ROOT/spec['path']) == spec,'Captured source changed during CUDA execution')
    finally:
        for handle in reversed(handles):
            handle.remove()
        observed['neural_calls_by_phase'] = {phase:{name:n for (p,name),n in calls.items() if p == phase}
            for phase in {p for p,_ in calls}}
        observed['scoped_hooks_removed'] = True
