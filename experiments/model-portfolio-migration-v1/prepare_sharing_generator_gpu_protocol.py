"""Bind six reviewed sharing models to the already qualified 64 native cases."""
from pathlib import Path
import time

from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/model-portfolio-migration-v1'
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
CASES = RUN/'sharing-generator-GPU-cases-01.json'
OUTPUT = RUN/'sharing-generator-GPU-benchmark-protocol-01.json'


def prepare():
    require(not OUTPUT.exists() and not CASES.exists(), 'Preserve original sharing protocol/cases')
    prior_path = RUN/'joint-generator-GPU-benchmark-protocol-01.json'
    prior = read(prior_path)
    original_cases_path = ROOT/prior['cases']['path']
    require(binding(original_cases_path) == prior['cases'], 'Original qualified cases changed')
    for spec in prior['dependencies']:
        require(binding(ROOT/spec['path']) == spec, 'Original benchmark dependency changed')
    reviewed_path = DOC/'reports/captured-sharing-temporal-generator-method-reviews-01.json'
    reviewed = read(reviewed_path)
    require(reviewed['source_reviewed_fits'] == 6 and len(reviewed['rows']) == 6,
            'All six reviewed endpoint/reuse fits required')
    models, inputs = [], set()
    original = read(original_cases_path)
    original_norm = original['models'][0]['normalization']
    for row in reviewed['rows']:
        receipt_path = ROOT/row['original_receipt']['path']
        require(binding(receipt_path) == row['original_receipt'], 'Original sharing fit receipt changed')
        receipt = read(receipt_path)
        weights = receipt_path.parent/'weights.npz'
        require(binding(weights)['sha256'] == receipt['outputs']['weights.npz']['sha256'],
                'Original sharing weights changed')
        normalization = next(i for i in row['training_inputs'] if i['slot'] == 'normalization')
        require(normalization['sha256'] == original_norm['sha256'], 'Shared physical normalization differs')
        model = dict(fit_item_id=row['fit_item_id'],identifier=row['generator_identifier'],K=row['horizons'],
            architecture=row['architecture'],project_path=str(Path(row['original_configuration']['path']).parent),
            configuration=row['original_configuration'],receipt=row['original_receipt'],weights=binding(weights),
            normalization=normalization,sources=row['sources'])
        for spec in [model['configuration'],model['receipt'],model['weights'],normalization,*model['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Reviewed sharing input changed')
            inputs.add(spec['path'])
        models.append(model)
    models.sort(key=lambda m:(m['K'],m['architecture']))
    require({(m['K'],m['architecture']) for m in models} ==
            {(k,a) for k in (2,3,4) for a in ('endpoint','reuse')}, 'Complete paired architecture set required')
    cases = dict(original,models=models,created_unix=time.time(),
        selection='Exact original64native cases,320original state files; model selection replaced by six reviewed sharing fits.',
        source_case_qualification=binding(original_cases_path),sharing_review=binding(reviewed_path),
        producer=binding(Path(__file__)))
    CASES.write_bytes(canonical(cases)+b'\n')
    inputs.update(spec['path'] for spec in prior['dependencies'])
    inputs.update(str(p.relative_to(ROOT)) for p in [prior_path,original_cases_path,reviewed_path,Path(__file__),
        ROOT/'experiments/model-portfolio-migration-v1/sharing_generator_gpu_worker.py',
        ROOT/'experiments/model-portfolio-migration-v1/run_generator_gpu_campaign.py',
        ROOT/'experiments/model-portfolio-migration-v1/run_joint_generator_gpu_benchmark.py'])
    blocks = [dict(identifier=m['identifier'],phase=phase,
        result=f'runs/model-portfolio-migration-v1/sharing-generator-GPU-benchmark-01/{m["identifier"]}-phase{phase}.json')
        for phase,ordered in [(0,models),(1,list(reversed(models)))] for m in ordered]
    protocol = dict(prior,created_unix=time.time(),cases=binding(CASES),models=6,process_blocks=blocks,
        worker='experiments/model-portfolio-migration-v1/sharing_generator_gpu_worker.py',
        dependencies=[binding(ROOT/p) for p in sorted(inputs)],expected_timed_network_calls=7680,
        expected_timed_physical_outputs=23040,expected_all_network_calls_including_first_and_warmups=10752,
        expected_all_physical_outputs_including_first_and_warmups=32256,
        constructor='Exact captured BundleNet(K,width,architecture); process-local alias before setup/timing; no neural math changes.',
        boundary='Unchanged frozen joint_generator_gpu_benchmark.run, including source predict and all native CPU input/output costs.',
        prior_protocol=binding(prior_path),case_order='Original64cases ascending in phase0 and reverse in phase1. Models sorted(K,architecture), then reverse.',
        producer=binding(Path(__file__)))
    OUTPUT.write_bytes(canonical(protocol)+b'\n')
    print('Six sharing models/12process blocks preregistered; 7,680timed calls/23,040physical outputs; no execution.')


if __name__ == '__main__':
    prepare()
