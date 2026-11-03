"""Freeze original35generator identities and new scalar-only replay sources."""
from pathlib import Path
import time

from import_common_generator_measurements import ROOT, RUN, DOC, read, save, binding, require


def run():
    output = RUN/'generator-scalar-replay-protocol-01.json'
    require(not output.exists(), 'Preserve original scalar replay protocol')
    old_path = RUN/'complete-generator-GPU-protocol-01.json'
    old = read(old_path)
    accepted_path = DOC/'reports/complete-generator-GPU-benchmark-01.json'
    accepted = read(accepted_path)
    changed = []
    dependencies = [binding(old_path),binding(accepted_path)]
    for spec in old['dependencies']:
        actual = binding(ROOT/spec['path'])
        if actual != spec:
            changed.append(dict(original=spec,current=actual))
        dependencies.append(actual)
    require(len(changed) == 1 and changed[0]['original']['path'] == 'projects/rheon-model-portfolio-unified-04/notes/portfolio-index.json',
        'Only the already accepted evolving portfolio index may differ from historical timing dependencies')
    current_index = read(ROOT/changed[0]['current']['path'])
    require({m['fit_item_id'] for m in old['models']} == {m['fit_item_id'] for m in accepted['models']}
        <= {m['item_id'] for m in current_index['fits']}, 'Every original fit identity preserved in accepted current index')
    allowed = {}
    for path in Path(__file__).parent.glob('*.py'):
        spec = binding(path)
        allowed[spec['path']] = spec['sha256']
    for selected in old['models']:
        for spec in selected['bound_inputs']:
            require(binding(ROOT/spec['path']) == spec, 'Original model material changed')
            if spec['path'].endswith('.py'):
                allowed[spec['path']] = spec['sha256']
    for spec in old['dependencies']:
        if spec['path'].endswith('.py'):
            allowed[spec['path']] = spec['sha256']
    # Runtime launcher is itself a bound source; only interpreted source paths
    # are recorded here, no model/data payload copies.
    common = ROOT/'experiments/pressure-four-pass-distillation-v1/common.py'
    allowed[str(common.relative_to(ROOT))] = binding(common)['sha256']
    save(output,dict(status='35original_generator_fits_admitted_for_scalar_only_accuracy_replay',created_unix=time.time(),
        producer=binding(Path(__file__)),dependencies=dependencies,models=old['models'],cases=old['cases'],
        accepted_accuracy=binding(accepted_path),mutable_index_evolution=changed,
        admitted_lab_python=allowed,expected_GPU='NVIDIA GeForce RTX 5090 Laptop GPU',host_cpu=4,
        worker_deadline_seconds=900,maximum_CUDA_allocated_bytes=4*1024**3,
        output_root='runs/model-portfolio-migration-v1/generator-scalar-replay-01',
        worker='experiments/model-portfolio-migration-v1/replay_generator_scalar_statistics.py',
        runtime_manifest=old['runtime_manifest'],primary_timing_samples=0,
        exposure=old['exposure'],reference_binary_sha256=old['reference_binary_sha256'],
        inference_policy='All learned constructors/queries/forwards onCUDA;fixed data/scoring onCPU. GPUsharing allowed foraccuracy only. Everyphysical field must match the original accepted output hash.',
        work_policy='Record scoped Python andmodule counts without interpreting overlapping calls as independentpasses orallocating joint cost toleads.',
        scalar_tests=binding(Path(__file__).with_name('test_full_state_scalar_diagnostics.py')),
        literal_scalar_tests_passed=7,expected_cases=2240,expected_outputs=5184,
        prediction_arrays_saved=False,new_parameter_fits=0,new_reference_generation=0,native_mutations=0))
    print('Admitted35originalfits/2240blocks/5184outputs;allscientificbindings unchanged;index evolution explicit.',flush=True)


if __name__ == '__main__':
    run()
