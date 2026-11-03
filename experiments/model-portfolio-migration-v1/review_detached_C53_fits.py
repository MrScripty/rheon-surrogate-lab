"""Review three C53 continuations, preserving the unnamed U execution identity."""
import math
from pathlib import Path
import time

from prepare_method_review_revisions import encode, revised_card, verified_review_receipt
from review_detached_C37_fits import ROOT, BASE, PROJECT, binding, read, require

DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
OUTPUT = DOC / 'reports/captured-detached-C53-method-reviews-01.json'


def run():
    require(not OUTPUT.exists(), 'Preserve the original authored review')
    prior_path = DOC / 'reports/captured-detached-C37-method-reviews-01.json'
    prior = read(prior_path)
    require(binding(ROOT / prior['producer']['path'])['sha256'] == prior['producer']['sha256'],
            'Audited source-family interpretation changed')
    previous = dict(zip(('U', 'A', 'D'), prior['rows'], strict=True))
    numerical_path = DOC / 'reports/pressure-correction-work-01.json'
    numerical = read(numerical_path)
    require(numerical['status'] == 'all_7680_retained_pressure_policy_fields_independently_rescored'
            and binding(ROOT / numerical['producer']['path'])['sha256'] == numerical['producer']['sha256'],
            'Independent numerical evidence required')
    cards = {row['item_id']:row for row in read(PROJECT / 'notes/portfolio-index.json')['fits']}
    common = read(DOC / 'reports/pressure-common-score-01.json')['models']
    rows = []
    shared_initialization = shared_order = None
    for mode, alias in [('U', None), ('A', 'C70'), ('D', 'C71')]:
        identifier = 'C53-' + mode
        project = BASE / ('c53-' + mode.lower() + '-project-01')
        train = BASE / ('c53-' + mode.lower() + '-study-01/train')
        receipt, history = read(train / 'receipt.json'), read(train / 'history.json')
        witness, manifest = read(project / 'source-witness.json'), read(project / 'project.json')
        fine, config = read(project / 'fine-tuning.json'), read(project / 'config.json')
        audited = previous[mode]
        audited_sources = {Path(entry['path']).name:entry for entry in audited['sources']}
        sources = [binding(project / name) for name in sorted(witness['sources'])]
        require(len(sources) == 32, 'Original complete 32-source closure required')
        shared_python = []
        for source in sources:
            name = Path(source['path']).name
            require(source['sha256'] == witness['sources'][name], 'Captured source changed')
            if name.endswith('.py'):
                require(source['sha256'] == audited_sources[name]['sha256'], 'Audited numerical Python differs: ' + name)
                shared_python.append(name)
        require(binding(project / 'config.json')['sha256'] == receipt['closure']['experiment_sha256']
                and witness['architecture'] == receipt['architecture']
                and witness['inputs'] == receipt['closure']['inputs'], 'Original captured closure differs')
        references = {entry['id']:entry for entry in manifest['data']}
        inputs = []
        for slot, expected in receipt['closure']['inputs'].items():
            path = ROOT / references[slot]['path']
            observed = binding(path)
            require(dict(sha256=observed['sha256'],bytes=path.stat().st_size) == expected,
                    'Original static training input differs: ' + slot)
            inputs.append(dict(observed,slot=slot,bytes=path.stat().st_size))
        recipe = read(project / 'train-recipe.json')
        require(len(recipe['sources']) == 35 and recipe['entrypoint'] == str((project/'evidence.py').relative_to(ROOT)),
                'Captured original recipe differs')
        for source in [*recipe['sources'], *manifest['recipes']]:
            require(binding(ROOT/source['path'])['sha256'] == source['sha256'], 'Recipe executable changed')
        old_fine = read(BASE / ('c37-' + mode.lower() + '-project-01/fine-tuning.json'))
        require(all(fine[key] == old_fine[key] for key in ('fitting_bank','rollout_feedback','training_passes',
                    'teacher_passes','teacher_weight','early_weight','relative_equation_target')),
                'Captured fitting objective differs from the audited mode')
        require(receipt['status'] == 'complete' and receipt['mode'] == 'train'
                and receipt['optimizer_updates'] == config['training']['updates'] == 1600
                and fine['initial_weights_sha256'] == common['C53']['weights_sha256']
                and sum(math.prod(value['shape']) for value in fine['initial_inventory'].values()) == 370016,
                'Original C53 initialization or optimizer budget differs')
        bank = receipt['fine_tuning_statistics']['fundamental_bank']
        feedback = receipt['fine_tuning_statistics']['rollout_feedback']
        require(len(bank['cases']) == 8 and bank['optimizer_examples'] == 1600
                and bank['expected_exposures'] == bank['optimizer_exposures'] == [200]*8
                and feedback['depth_exposures'] == [[50]*4]*8
                and feedback['auxiliary_losses'] == (0 if mode == 'U' else 1600)
                and receipt['fine_tuning_statistics']['teacher_forwards'] == 0
                and history['optimizer_work']['network_forwards'] == (6400 if mode == 'U' else 8000),
                'Actual fitting population or counted work differs')
        stream = (bank['sampling_order_sha256'],feedback['depth_order_sha256'],bank['amplitude_rms_pa'])
        if shared_initialization is not None:
            require(shared_initialization == fine['initial_inventory'] and shared_order == stream,
                    'C53 arms do not have matched initial tensors and optimizer streams')
        shared_initialization, shared_order = fine['initial_inventory'], stream
        population = read(train/'splits.json')
        require(len(population['parents']) == 16 and all(p['entry']['split']=='train' for p in population['parents']),
                'Physical calibration population differs')
        for name in ('weights.npz','checkpoint-1600.npz'):
            require(dict(sha256=binding(train/name)['sha256'],bytes=(train/name).stat().st_size)
                    == {k:receipt['outputs'][name][k] for k in ('sha256','bytes')}, 'Original checkpoint changed')
        score = numerical['models'][identifier]
        evidence = ROOT / score['result']['result']
        captured = read(evidence)
        require(binding(evidence)['sha256'] == score['result']['sha256']
                and captured['weights_sha256'] == receipt['outputs']['weights.npz']['sha256']
                and captured['custody']['kind'] == 'training_checkpoint', 'Replay selects another training artifact')
        if alias:
            require(common[alias]['weights_sha256'] == captured['weights_sha256'], 'Named alias selects another checkpoint')
        else:
            require(common['C66']['weights_sha256'] == captured['weights_sha256']
                    and score['original_default_output_parity_cases'] == 128, 'C53-U/C66 tested equivalence differs')
        for policy in ('fixed_1','fixed_2','fixed_4','fixed_8'):
            ratios = numerical['matched_policy_parent_comparisons'][identifier][policy]['ratio_of_worst_to_parent']
            require(ratios['common_equation_relative_l2'] > 1 and ratios['pressure_mean_free_relative_l2'] > 1,
                    'Matched-budget regression statement differs')
        worst = score['summary']['captured_default']['worst']
        fit = 'fit-' + binding(train/'receipt.json')['sha256'][:16]
        card = cards[fit]
        require(card.get('method_review_status') != 'captured_source_and_receipt_reviewed', 'Already reviewed native fit')
        label = alias or 'C53-U (unnamed completed fit)'
        row = dict(fit_item_id=fit,note_path=card['card_path'],
            method=audited['method'].replace('C37','C53'),purpose=audited['purpose'],experiment=audited['experiment'],
            outcome=f"{label}: worst mean-free pressure relative L2 {worst['pressure_mean_free_relative_l2']:.17g} "
                f"and equation relative L2 {worst['common_equation_relative_l2']:.17g} at the captured two-correction default. "
                'All 128 consumed old-solver equations were also replayed at one, two, four and eight corrections; '
                'all three C53 fine-tunes are worse than C53 on both worst metrics at every matched budget. '
                + ('Its default outputs match C66 byte-for-byte on all 128 equations; this does not equate their fit histories. '
                   if alias is None else '') + 'No timing or own-history result is implied.',
            lesson='The eight-pattern objective fails to improve this stronger parent on the consumed native pressure bank, '
                'including at its trained four-correction depth. Extra corrections reduce absolute errors but do not reverse '
                'the parent ranking. The earlier C37 default comparison also changed one to two corrections; the new matched '
                'matrix confirms that C37 benefits only conditionally on budget and metric. Preserve distinct fit identities '
                'even when checkpoint bytes and tested outputs coincide. No sole causal explanation or unseen-scene claim follows.',
            reproduction=audited['reproduction'].replace('C37','C53') +
                ' The original CUDA training receipt is sufficient for the declared training-checkpoint replay; '
                'do not invent a CPU-restore receipt or a cohort alias for C53-U.',
            original_receipt=binding(train/'receipt.json'),history=binding(train/'history.json'),
            source_witness=binding(project/'source-witness.json'),sources=sources,
            training_population=binding(train/'splits.json'),training_inputs=inputs,
            retained_accuracy=binding(evidence),independent_accuracy=binding(numerical_path),
            effective_optimizer_patterns=8,calibration_trajectories=16,
            network_forwards=history['optimizer_work']['network_forwards'],
            numerical_python_sources_identical_to_reviewed_C37=shared_python,
            named_alias=alias,parent_accuracy=common['C53']['result'])
        original = verified_review_receipt(row)
        raw = (PROJECT/row['note_path']).read_bytes()
        require(raw.count(original)==1 and original in revised_card(raw,row)[1], 'Original native receipt would be lost')
        rows.append(row)
    report = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        classification='authored_source_and_receipt_interpretation_not_retraining',
        overview_addition='The three C53 detached continuations reuse the audited fitting code and eight-pattern stream. '
            'All regress against C53 on both worst errors at matched 1/2/4/8 correction budgets. C53-U remains a distinct '
            'unnamed completed fit; its weight bytes and 128 tested default outputs coincide with C66.',
        producer=binding(Path(__file__)),reused_source_interpretation=binding(prior_path),
        independent_accuracy=binding(numerical_path),new_model_forwards=0,native_mutations=0,
        clean_reproduction_qualified=False)
    OUTPUT.write_bytes(encode(report))
    print('Three C53 source reviews complete, including the unnamed original U fit.')


if __name__ == '__main__':
    run()
