"""Review C44's matched gradient-path fits using the already audited code family."""
import math
from pathlib import Path
import time

from prepare_method_review_revisions import encode, revised_card, verified_review_receipt
from review_detached_C37_fits import ROOT, BASE, PROJECT, binding, read, require

DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
OUTPUT = DOC / 'reports/captured-detached-C44-method-reviews-01.json'


def run():
    require(not OUTPUT.exists(), 'Fresh authored review output required')
    prior = read(DOC / 'reports/captured-detached-C37-method-reviews-01.json')
    require(binding(ROOT / prior['producer']['path'])['sha256'] == prior['producer']['sha256'],
            'Original source-review producer changed')
    previous = dict(zip(('U', 'A', 'D'), prior['rows'], strict=True))
    cards = {row['item_id']: row for row in read(PROJECT / 'notes/portfolio-index.json')['fits']}
    scores = read(DOC / 'reports/pressure-common-score-01.json')['models']
    rows = []
    initialization = pattern_order = depth_order = amplitude = None
    for mode, label in [('U', 'C72'), ('A', 'C73'), ('D', 'C74')]:
        arm = 'c44-' + mode.lower()
        project = BASE / (arm + '-project-01')
        train = BASE / (arm + '-study-01/train')
        receipt, history = read(train / 'receipt.json'), read(train / 'history.json')
        witness, manifest = read(project / 'source-witness.json'), read(project / 'project.json')
        fine, config = read(project / 'fine-tuning.json'), read(project / 'config.json')
        old = previous[mode]
        audited_sources = {Path(entry['path']).name: entry for entry in old['sources']}
        sources = [binding(project / name) for name in sorted(witness['sources'])]
        require(len(sources) == 32, 'Original32source closure required')
        shared_python = []
        for source in sources:
            name = Path(source['path']).name
            require(source['sha256'] == witness['sources'][name], 'Captured source changed')
            if name.endswith('.py'):
                require(source['sha256'] == audited_sources[name]['sha256'],
                        'Numerical/fitting code differs from the audited C37family: ' + name)
                shared_python.append(name)
        require(binding(project / 'config.json')['sha256'] == receipt['closure']['experiment_sha256']
                and witness['architecture'] == receipt['architecture']
                and witness['inputs'] == receipt['closure']['inputs'], 'Original witness/configuration differs')
        references = {entry['id']: entry for entry in manifest['data']}
        inputs = []
        for slot, expected in receipt['closure']['inputs'].items():
            path = ROOT / references[slot]['path']
            observed = binding(path)
            require(dict(sha256=observed['sha256'], bytes=path.stat().st_size) == expected,
                    'Static original input differs: ' + slot)
            inputs.append(dict(observed, slot=slot, bytes=path.stat().st_size))
        recipe = read(project / 'train-recipe.json')
        require(len(recipe['sources']) == 35 and recipe['entrypoint'] == str((project / 'evidence.py').relative_to(ROOT)),
                'Original recipe differs')
        for source in [*recipe['sources'], *manifest['recipes']]:
            require(binding(ROOT / source['path'])['sha256'] == source['sha256'], 'Recipe source changed')
        old_fine = read(BASE / ('c37-' + mode.lower() + '-project-01/fine-tuning.json'))
        require(fine['fitting_bank'] == old_fine['fitting_bank']
                and fine['rollout_feedback'] == old_fine['rollout_feedback']
                and all(fine[key] == old_fine[key] for key in
                        ('training_passes', 'teacher_passes', 'teacher_weight', 'early_weight', 'relative_equation_target')),
                'Original fitting objective or eight-pattern stream differs')
        require(receipt['status'] == 'complete' and receipt['mode'] == 'train'
                and receipt['optimizer_updates'] == config['training']['updates'] == 1600
                and fine['initial_weights_sha256'] == scores['C44']['weights_sha256']
                and sum(math.prod(value['shape']) for value in fine['initial_inventory'].values()) == 370016,
                'Original C44initialization or fitting budget differs')
        if initialization is not None:
            require(initialization == fine['initial_inventory'], 'Matched arms do not share exact initial tensors')
        initialization = fine['initial_inventory']
        bank = receipt['fine_tuning_statistics']['fundamental_bank']
        feedback = receipt['fine_tuning_statistics']['rollout_feedback']
        require(len(bank['cases']) == 8 and bank['optimizer_examples'] == 1600
                and bank['expected_exposures'] == bank['optimizer_exposures'] == [200] * 8
                and feedback['depth_exposures'] == [[50] * 4] * 8
                and feedback['auxiliary_losses'] == (0 if mode == 'U' else 1600)
                and receipt['fine_tuning_statistics']['teacher_forwards'] == 0
                and history['optimizer_work']['network_forwards'] == (6400 if mode == 'U' else 8000),
                'Actual optimizer population, auxiliary depths or counted work differs')
        if pattern_order is not None:
            require(pattern_order == bank['sampling_order_sha256'] and depth_order == feedback['depth_order_sha256']
                    and amplitude == bank['amplitude_rms_pa'], 'Matched stream/depth/amplitude differs')
        pattern_order, depth_order, amplitude = bank['sampling_order_sha256'], feedback['depth_order_sha256'], bank['amplitude_rms_pa']
        population = read(train / 'splits.json')
        require(len(population['parents']) == 16 and all(parent['entry']['split'] == 'train' for parent in population['parents']),
                'Physical calibration population differs')
        for name in ('weights.npz', 'checkpoint-1600.npz'):
            observed = binding(train / name)
            require(dict(sha256=observed['sha256'], bytes=(train / name).stat().st_size)
                    == {key: receipt['outputs'][name][key] for key in ('sha256', 'bytes')}, 'Checkpoint bytes changed')
        score = scores[label]
        evidence = ROOT / score['result']['path']
        require(binding(evidence)['sha256'] == score['result']['sha256']
                and score['weights_sha256'] == receipt['outputs']['weights.npz']['sha256']
                and len(read(evidence)['rows']) == 128, 'Common accuracy evidence selects another checkpoint')
        worst, parent = score['worst'], scores['C44']['worst']
        require(worst['pressure_mean_free_relative_l2'] > parent['pressure_mean_free_relative_l2']
                and worst['common_equation_relative_l2'] > parent['common_equation_relative_l2'],
                'Observed regression statement does not hold')
        fit = 'fit-' + binding(train / 'receipt.json')['sha256'][:16]
        card = cards[fit]
        require(card.get('method_review_status') != 'captured_source_and_receipt_reviewed', 'Do not replace an existing native review')
        row = dict(fit_item_id=fit, note_path=card['card_path'],
            method=old['method'].replace('C37', 'C44'), purpose=old['purpose'], experiment=old['experiment'],
            outcome=f"{label}, on the same retained 128 pressure requests: worst mean-free pressure relative L2 "
                    f"{worst['pressure_mean_free_relative_l2']:.17g}, equation relative L2 "
                    f"{worst['common_equation_relative_l2']:.17g}. Parent C44 is better on both: "
                    f"{parent['pressure_mean_free_relative_l2']:.17g} and {parent['common_equation_relative_l2']:.17g}, respectively. "
                    'No new timing or own-history drift result is implied.',
            lesson='All three eight-pattern fine-tunes regress against the already stronger C44 parent on both '
                   'worst pressure and equation error under the unchanged common benchmark. The matched C37 '
                   'family improved instead, so that improvement does not transfer automatically to another '
                   'initialization. Among these C44 continuations, A has the lowest worst pressure error and '
                   'D the lowest worst equation residual; neither supplies an accuracy reason to replace the parent. '
                   'A narrow analytical fitting objective can disagree with broad native-case results. '
                   'The data establish the regression, not its sole cause; preserve the parent and validate '
                   'matched cases before promotion. Equal update counts also conceal the A/D extra forward.',
            reproduction=old['reproduction'].replace('C37', 'C44'),
            original_receipt=binding(train / 'receipt.json'), history=binding(train / 'history.json'),
            source_witness=binding(project / 'source-witness.json'), sources=sources,
            training_population=binding(train / 'splits.json'), training_inputs=inputs,
            retained_accuracy=binding(evidence), effective_optimizer_patterns=8, calibration_trajectories=16,
            network_forwards=history['optimizer_work']['network_forwards'],
            numerical_python_sources_identical_to_reviewed_C37=shared_python,
            parent_accuracy=scores['C44']['result'])
        original = verified_review_receipt(row)
        raw = (PROJECT / row['note_path']).read_bytes()
        require(raw.count(original) == 1 and original in revised_card(raw, row)[1], 'Original native receipt would be lost')
        rows.append(row)
    report = dict(schema='rheon.captured-model-method-reviews.v1', created_unix=time.time(),
        classification='authored_source_and_receipt_interpretation_not_retraining', rows=rows,
        overview_addition='C72/C73/C74 reuse the exact audited C37 fitting Python code but start from C44. '
        'They train on the same eight single-mode patterns with matched initialization/order/depths within '
        'their family. All three regress against C44 on both common worst pressure and equation error. '
        'The C37 improvement therefore cannot justify promoting these C44 continuations.',
        new_model_forwards=0, native_mutations=0, clean_reproduction_qualified=False,
        producer=binding(Path(__file__)), reused_source_interpretation=binding(DOC / 'reports/captured-detached-C37-method-reviews-01.json'))
    OUTPUT.write_bytes(encode(report))
    print('Three C44 source reviews complete; all regress against the parent on both common worst errors.')


if __name__ == '__main__':
    run()
