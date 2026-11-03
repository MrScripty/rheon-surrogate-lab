"""Identify captured CPU neural setup that must be qualified before CUDA replay."""
import math
from pathlib import Path
import time

from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
OUTPUT = DOC/'reports/temporal-GPU-construction-readiness-01.json'


def run():
    require(not OUTPUT.exists(), 'Preserve original constructor readiness observation')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    available = read(availability_path)
    candidates = [r for r in available['rows'] if any(Path(s['path']).name == 'frozen_temporal_pressure_core.py' for s in r['sources'])]
    # The availability worklist intentionally excluded the one generator already
    # source-reviewed at that time. Include its original captured constructor too.
    core_review_path = DOC/'reports/core-component-method-reviews-01.json'
    base = next(r for r in read(core_review_path)['rows'] if r['fit_item_id'] == 'fit-718037c1d106335e')
    candidates.append(dict(fit_item_id=base['fit_item_id'],project_path='runs/full-ai-next-priorities-v1/b1-fixed-viscosity-project-01',
        receipt=base['original_receipt']))
    require(len(candidates) == len({r['fit_item_id'] for r in candidates}) == 16, 'All16inherited-pressure generator constructors required')
    rows = []
    for item in candidates:
        project = ROOT/item['project_path']
        receipt_path = ROOT/item['receipt']['path']
        require(binding(receipt_path)['sha256'] == item['receipt']['sha256'], 'Original fit receipt differs')
        receipt = read(receipt_path)
        config = read(project/'config.json')
        recipe = read(project/'train-recipe.json')
        trainer = ROOT/recipe['entrypoint']
        frozen = project/'frozen_temporal_pressure_core.py'
        require(binding(frozen)['sha256'] == receipt['closure']['sources']['frozen_temporal_pressure_core.py']
                and binding(trainer)['sha256'] == receipt['closure']['sources'][trainer.name], 'Captured constructor/trainer source changed')
        source = frozen.read_text()
        require('spatial=copy.deepcopy(self.fine).double().eval()' in source
                and 'value=torch.zeros((batch,1,*counts),dtype=torch.float64)' in source
                and 'matrix[:,start:start+batch]=spatial(value)' in source
                and 'asset_directory=assets).to(device)' in trainer.read_text(),
                'Actual eager unit-response-before-device-transfer pattern differs')
        assets = project/'expert-assets'
        geometries = read(assets/'pressure/configuration.json')['geometries']
        fine = [math.prod(g['counts_zyx']) for g in geometries if max(g['counts_zyx']) <= 16 and math.prod(g['counts_zyx']) <= 2048]
        full_map = project/'full_learned_pressure_map.py'
        uses_full_map = config['architecture'] in ('coupled','displacement')
        full = [math.prod(g['counts_zyx']) for g in geometries if max(g['counts_zyx']) <= 16] if uses_full_map else []
        sources = [binding(frozen),binding(trainer),binding(project/'config.json'),binding(assets/'pressure/configuration.json')]
        if uses_full_map:
            require(binding(full_map)['sha256'] == receipt['closure']['sources']['full_learned_pressure_map.py']
                    and 'matrix[:,start:start+batch]=self.linear_graph' in full_map.read_text(), 'Actual complete learned-map constructor differs')
            sources.append(binding(full_map))
        rows.append(dict(fit_item_id=item['fit_item_id'],identifier=config['identifier'],receipt=binding(receipt_path),
            original_sources=sources,fine_unit_query_counts=fine,fine_root_queries_in_batches_of4=sum((n+3)//4 for n in fine),
            fine_derived_matrix_bytes=sum(n*n*8 for n in fine),full_unit_query_counts=full,
            full_derived_matrix_bytes=sum(n*n*8 for n in full),construction_execution='CPU under original default tensor allocation before .to(device)',
            observed_new_neural_forwards=0,actual_GPU_only_setup_qualified=False,
            interpretation='Static source/default-device analysis, not a new constructor execution or timing measurement. Full-map construction contains the earlier fine-map preparation too.'))
    value = dict(status='16_captured_generator_constructors_require_GPU_only_setup_qualification',created_unix=time.time(),rows=rows,
        inference_policy='Every new neural forward must be CUDA, including constructor unit-response queries and publication/restore probes.',
        existing_frozen_backend_timings_invalidated=False,
        existing_backend_scope='Previously qualified component-composition backends have their own source/guard evidence; this report does not execute or invalidate those different constructors.',
        proposed_variant='Construct parameters, registered buffers, unit inputs, matrices and basis tensors on CUDA before any neural call; guard setup as well as prediction. Keep original learned tensor bytes. Record GPU-built derived buffers as an explicit precision/execution variant and compare against retained outputs without new CPU inference.',
        qualification_required=['No CPU/unclassified neural call during setup or replay.',
            'Original final tensor inventory and learned asset identities remain exact.',
            'First-case saved-probe metric crosschecks, independent original-reference field scoring and repeated outputs.',
            'Report constructor/basis/map preparation time and memory separately from complete native warmed block time.',
            'Retain all precision/route differences rather than claiming bitwise equivalence from equal weights.'],
        new_model_forwards=0,new_reference_generation=False,payload_copies=0,native_mutations=0,full_portfolio_complete=False,
        producer=binding(Path(__file__)),evidence=[binding(availability_path),binding(core_review_path)])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('16captured constructors identified for GPU-only setup qualification; no neural execution.')


if __name__ == '__main__':
    run()
