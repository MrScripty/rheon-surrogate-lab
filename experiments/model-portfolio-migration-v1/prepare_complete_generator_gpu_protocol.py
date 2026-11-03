"""Bind every one of the35reviewed generator fits to the same64native cases."""
from pathlib import Path
import time

from review_water_learning_sequence import ROOT, binding, canonical, read, require

RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = RUN/'complete-generator-GPU-protocol-01.json'


def prepare():
    require(not OUTPUT.exists(),'Preserve the original complete-generator protocol')
    cases_path = DOC/'reports/joint-generator-case-preflight-01.json'
    cases = read(cases_path)
    inherited_path = RUN/'inherited-generator-CUDA-qualification-protocol-01.json'
    inherited = read(inherited_path)
    available_path = DOC/'reports/full-generator-review-input-availability-01.json'
    available = read(available_path)
    index_path = ROOT/'projects/rheon-model-portfolio-unified-04/notes/portfolio-index.json'
    index = read(index_path)
    expected = {r['item_id'] for r in index['fits'] if r['task'] == 'full_state_generator'}
    models = {m['fit_item_id']:dict(m,constructor_kind='inherited_CUDA') for m in inherited['models']}
    original_norm = cases['models'][0]['normalization']
    for row in available['rows']:
        if row['fit_item_id'] in models:
            continue
        project = ROOT/row['project_path']
        config_path = project/'config.json'
        config,receipt = read(config_path),read(ROOT/row['receipt']['path'])
        require(binding(config_path)['sha256'] == row['configuration']['sha256'] == receipt['closure']['config_sha256'],
                'Original configuration identity differs')
        recipe = read(project/'train-recipe.json')
        trainer = ROOT/recipe['entrypoint']
        source = {str((ROOT/s['path']).relative_to(project)):s['sha256'] for s in row['sources']}
        require(source == receipt['closure']['sources'],'Complete captured source closure required')
        sources = [binding(ROOT/s['path']) for s in row['sources']]
        require(all(s['sha256'] == row_source['sha256'] for s,row_source in zip(sources,row['sources'],strict=True)),
                'Original captured source changed')
        document = read(project/'project.json')
        norm = next(d for d in document['data'] if d['id'] == 'normalization')
        require(binding(ROOT/norm['path'])['sha256'] == original_norm['sha256'], 'Common physical normalization required')
        weights = Path(row['receipt']['path']).parent/'weights.npz'
        require(binding(ROOT/weights)['sha256'] == receipt['outputs']['weights.npz']['sha256'],
                'Original final weights changed')
        models[row['fit_item_id']] = dict(fit_item_id=row['fit_item_id'],identifier=config['identifier'],
            K=config['horizons'],architecture=config.get('architecture','direct'),width=config['width'],
            constructor_kind='plain_captured',project_path=row['project_path'],
            configuration=binding(config_path),receipt=binding(ROOT/row['receipt']['path']),weights=binding(ROOT/weights),
            sources=sources,trainer=binding(trainer),normalization=binding(ROOT/norm['path']))
    require(len(models) == 35 and set(models) == expected,'Every original full-generator fit required')
    prior_qualified = {m['fit_item_id'] for m in inherited['models']}
    for path in [cases_path,RUN/'sharing-generator-GPU-cases-01.json']:
        prior_qualified.update(m['fit_item_id'] for m in read(path)['models'])
    require(len(prior_qualified) == 26,'Previously evaluated26fit population differs')
    for model in models.values():
        # Canonical byte bindings remain small; no model, corpus or project copy.
        project = ROOT/model['project_path']
        bound = {s['path']:binding(ROOT/s['path']) for s in
            [model['configuration'],model['receipt'],model['weights'],model['normalization'],model['trainer'],*model['sources']]}
        for path in [project/'project.json',project/'train-recipe.json',project/'step-scales.json',
                     (ROOT/model['receipt']['path']).parent/'probe.npy']:
            if path.exists():
                bound[str(path.relative_to(ROOT))] = binding(path)
        model['bound_inputs'] = sorted(bound.values(),key=lambda s:s['path'])
        model['previous64case_accuracy_qualified'] = model['fit_item_id'] in prior_qualified
    ordered = sorted(models.values(),key=lambda m:(m['previous64case_accuracy_qualified'],m['K'],m['fit_item_id']))
    dependencies = [binding(p) for p in [cases_path,inherited_path,available_path,index_path,
        RUN/'../dependency-migration-v1/runtime/current-manifest-v2.json']]
    for name in ['prepare_complete_generator_gpu_protocol.py','complete_generator_gpu_benchmark.py',
            'run_complete_generator_gpu_benchmark.py','test_complete_generator_timing_boundary.py',
            'inherited_generator_cuda.py','full_state_gpu_timing.py','joint_generator_gpu_benchmark.py',
            'reduce_sharing_generator_gpu_benchmark.py','reduce_joint_generator_gpu_benchmark.py',
            'verify_joint_generator_retained_probes.py','review_water_learning_sequence.py',
            'review_direction_pressure_fits.py','run_joint_generator_gpu_benchmark.py']:
        dependencies.append(binding(Path(__file__).with_name(name)))
    dependencies.append(binding(ROOT/'experiments/pressure-four-pass-distillation-v1/common.py'))
    for case in cases['cases']:
        for spec in [case['current'],*case['reference_leads']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'],'Original native case changed')
    blocks = [dict(fit_item_id=m['fit_item_id'],identifier=m['identifier'],phase=phase,
        result=f'runs/model-portfolio-migration-v1/complete-generator-GPU-benchmark-01/{m["fit_item_id"]}-phase{phase}.json')
        for phase,sequence in [(0,ordered),(1,list(reversed(ordered)))] for m in sequence]
    result = dict(status='preregistered_not_dispatched',created_unix=time.time(),models=ordered,
        cases=binding(cases_path),dependencies=dependencies,process_blocks=blocks,
        worker='experiments/model-portfolio-migration-v1/complete_generator_gpu_benchmark.py',
        output='runs/model-portfolio-migration-v1/complete-generator-GPU-benchmark-01',
        runtime_manifest='runs/dependency-migration-v1/runtime/current-manifest-v2.json',
        host_cpu=4,worker_deadline_seconds=600,max_CUDA_allocated_bytes=4*1024**3,
        warmups=3,samples=10,calibration_pairs=4,cases_per_model=64,device='CUDA',
        timing='Complete native CPU5fields to allKowned CPU5fields; same captured packing,known forcing,H2D,predict,reconstruction,D2H/allocation/synchronization. Primary calls have no validation hook. Setup,first,warmups,hashes/ownership/scoring outside primary.',
        instrumentation='Separate four alternating hook-on/off pairs percase; phase1reversespair order. Guard-observed first/warm/on calls distinguished from logical uninstrumented primary/off calls.',
        constructor='Exact captured trainers;16inherited constructors use separately qualified CUDA map variant,all guards removed before primary. Other constructors forbid unexpected neural setup calls.',
        reference_binary_sha256=cases['reference_native_probe_binary_sha256'],
        contention='Refuse dispatch with another CUDA compute PID; record before/after GPU/load. No continuous exclusivity claim; no native mutation/browser during campaign. External host work must remain disclosed.',
        expected_primary_calls=35*2*64*10,
        expected_primary_physical_outputs=sum(m['K'] for m in ordered)*2*64*10,
        expected_logical_complete_calls=35*2*64*(1+3+10+8),
        exposure='All16consumed training scenes,four starts,dt1/60;no independent holdout or long drift claim.',
        target='Diagnostic GPU speed/accuracy;1.2msobsolete. Divide only by Kphysical outputs.',
        CPU_neural_forwards=0,output_arrays_written=False,new_reference_generation=False,
        native_mutations=0,full_portfolio_complete=False,producer=binding(Path(__file__)))
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('All35fit identities/70freshworkers bound;',result['expected_primary_calls'],'primary calls;no execution.')


if __name__ == '__main__':
    prepare()
