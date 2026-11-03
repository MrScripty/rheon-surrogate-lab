"""Bind all16original inherited generators and the same64retained physical cases."""
from pathlib import Path
import time

from captured_temporal_review import audit
from review_water_learning_sequence import ROOT, binding, canonical, read, require

RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = RUN/'inherited-generator-CUDA-qualification-protocol-01.json'


def run():
    require(not OUTPUT.exists(),'Preserve original family qualification protocol')
    readiness_path = DOC/'reports/temporal-GPU-construction-readiness-01.json'
    readiness = read(readiness_path)
    require(len(readiness['rows']) == 16,'All16captured inherited constructors required')
    pilot_path = RUN/'inherited-generator-CUDA-construction-pilot-01/result.json'
    pilot = read(pilot_path)
    require(pilot['status'] == 'one_inherited_constructor_CUDA_setup_and_first_case_qualified'
            and pilot['CPU_neural_forwards'] == pilot['forbidden_forward_attempts'] == 0
            and pilot['scoped_global_hooks_removed'],'Completed CUDA-only first pilot required')
    cases_path = DOC/'reports/joint-generator-case-preflight-01.json'
    cases = read(cases_path)
    require(len(cases['cases']) == 64 and cases['unique_native_state_files'] == 320,
            'Exact original common physical population required')
    dependencies = {}

    def add(path):
        spec = binding(path)
        dependencies[spec['path']] = spec
        return spec

    for path in [readiness_path,pilot_path,ROOT/pilot['producer']['path'],cases_path,
            ROOT/'runs/dependency-migration-v1/runtime/current-manifest-v2.json',
            ROOT/'experiments/pressure-four-pass-distillation-v1/common.py']:
        add(path)
    for name in ['prepare_inherited_generator_qualification.py','inherited_generator_cuda.py',
            'qualify_inherited_generator_family.py','captured_temporal_review.py','review_water_learning_sequence.py',
            'prepare_method_review_revisions.py','full_state_gpu_timing.py','joint_generator_gpu_benchmark.py',
            'reduce_sharing_generator_gpu_benchmark.py','verify_joint_generator_retained_probes.py',
            'review_direction_pressure_fits.py','run_joint_generator_gpu_benchmark.py']:
        add(Path(__file__).with_name(name))
    for case in cases['cases']:
        for spec in [case['current'],*case['reference_leads']]:
            require(add(ROOT/spec['path'])['sha256'] == spec['sha256'],'Original retained state changed')
        require(add(ROOT/case['reference_receipt']['path']) == case['reference_receipt'],'Original native receipt changed')
    models = []
    for row in readiness['rows']:
        for spec in [row['receipt'],*row['original_sources']]:
            require(add(ROOT/spec['path']) == spec,'Original readiness source changed')
        config_path = ROOT/next(s['path'] for s in row['original_sources'] if Path(s['path']).name == 'config.json')
        project = config_path.parent
        recipe_path = project/'train-recipe.json'
        recipe = read(recipe_path)
        trainer = ROOT/recipe['entrypoint']
        context = audit(dict(project_path=str(project.relative_to(ROOT)),receipt=row['receipt']),trainer.name)
        receipt,config = context['receipt'],context['config']
        require(config['identifier'] == row['identifier'],'Original model identifier differs')
        for spec in [*context['facts']['sources'],*context['facts']['training_inputs']]:
            require(add(ROOT/spec['path'])['sha256'] == spec['sha256'],'Audited source/input changed')
        for path in [recipe_path,project/'project.json',project/'step-scales.json',
                context['train']/'receipt.json',context['train']/'weights.npz',context['train']/'probe.npy']:
            add(path)
        if 'continuation_weights_sha256' in config:
            require(add(project/'continuation-weights.npz')['sha256'] == config['continuation_weights_sha256'],
                    'Original continuation initialization changed')
        norm = next(i for i in context['facts']['training_inputs'] if i['slot'] == 'normalization')
        models.append(dict(identifier=row['identifier'],fit_item_id=row['fit_item_id'],K=config['horizons'],
            architecture=config['architecture'],width=config['width'],project_path=str(project.relative_to(ROOT)),
            configuration=add(config_path),trainer=add(trainer),receipt=row['receipt'],
            weights=add(context['train']/'weights.npz'),normalization=add(ROOT/norm['path']),
            sources=context['facts']['sources'],readiness=row,
            original_final_inventory_parameter_count=receipt['parameters']))
    require(len(models) == len({m['fit_item_id'] for m in models}) == len({m['identifier'] for m in models}) == 16,
            'Complete distinct original model population required')
    # Exercise both constructor branches early; each fit still owns a fresh
    # process, so this changes no geometry/learned-map cache sharing policy.
    models.sort(key=lambda m:(bool(m['readiness']['full_unit_query_counts']),m['identifier']))
    result = dict(status='prepared_not_dispatched',created_unix=time.time(),models=models,cases=add(cases_path),
        dependencies=sorted(dependencies.values(),key=lambda s:s['path']),
        output='runs/model-portfolio-migration-v1/inherited-generator-CUDA-qualification-01',
        host_cpu=6,worker_deadline_seconds=600,models_count=16,cases_per_model=64,repetitions=2,
        expected_prediction_calls=16*64*2,expected_physical_outputs_including_repeat=sum(m['K'] for m in models)*64*2,
        original_reference_binary_sha256=cases['reference_native_probe_binary_sha256'],
        inference_latency_qualified=False,concurrent_native_metadata_work=True,
        precision_variant='GPU-built derived F64maps/bases; exact original learned final tensors.',
        scope='CUDA construction,all64old-reference native cases,allKfields,repeatability/ownership and saved first-case probe metrics.',
        new_fits=0,new_reference_generation=False,output_arrays_written=False,native_mutations=0,full_portfolio_complete=False)
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Prepared all16inherited fits:',result['expected_prediction_calls'],'complete predictions/',
        result['expected_physical_outputs_including_repeat'],'physical outputs;no speed claim or dispatch.')


if __name__ == '__main__':
    run()
