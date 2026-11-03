"""Review the original matched C37 attached/detached fits without model imports."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from prepare_method_review_revisions import revised_card, verified_review_receipt

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
BASE = ROOT / 'runs/pressure-detached-correction-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC / 'reports/captured-detached-C37-method-reviews-01.json'


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run():
    require(not OUTPUT.exists(), 'Fresh source review required')
    index = read(PROJECT / 'notes/portfolio-index.json')
    cards = {fit['item_id']: fit for fit in index['fits']}
    aliases = {alias['item_id']: alias for alias in index['aliases']}
    scores = read(DOC / 'reports/pressure-common-score-01.json')['models']
    rows = []
    initial = sample_order = depth_order = None
    for mode, label, fit in [('U', 'C75', 'fit-0595218bf00cecf7'),
                             ('A', 'C76', 'fit-4dd23f41a4aa0d5d'),
                             ('D', 'C77', 'fit-db6062f2bb250ad4')]:
        arm = 'c37-' + mode.lower()
        project = BASE / (arm + '-project-01')
        train = BASE / (arm + '-study-01/train')
        receipt = read(train / 'receipt.json')
        history = read(train / 'history.json')
        config = read(project / 'config.json')
        fine = read(project / 'fine-tuning.json')
        witness = read(project / 'source-witness.json')
        manifest = read(project / 'project.json')
        recipe = read(project / 'train-recipe.json')
        population = read(train / 'splits.json')
        sources = [binding(project / name) for name in sorted(witness['sources'])]
        require(len(sources) == 32 and all(row['sha256'] == witness['sources'][Path(row['path']).name]
                for row in sources), 'Original captured source closure differs')
        require(binding(project / 'source-witness.json')['sha256'] == receipt['closure']['source_witness_sha256']
                and binding(project / 'config.json')['sha256'] == receipt['closure']['experiment_sha256']
                and witness['inputs'] == receipt['closure']['inputs']
                and witness['architecture'] == receipt['architecture'], 'Receipt/witness closure differs')
        require(len(recipe['sources']) == 35 and recipe['entrypoint'] == str((project / 'evidence.py').relative_to(ROOT)),
                'Original executed recipe differs')
        for entry in [*recipe['sources'], *manifest['recipes']]:
            require(binding(ROOT / entry['path'])['sha256'] == entry['sha256'], 'Recipe source changed')
        references = {item['id']: item for item in manifest['data']}
        inputs = []
        for slot, expected in receipt['closure']['inputs'].items():
            path = ROOT / references[slot]['path']
            observed = binding(path)
            require(dict(sha256=observed['sha256'], bytes=path.stat().st_size) == expected,
                    'Actual static training input changed: ' + slot)
            inputs.append(dict(observed, slot=slot, bytes=path.stat().st_size))
        require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
                and receipt['optimizer_updates'] == config['training']['updates'] == 1600
                and fine['training_passes'] == 4 and fine['teacher_passes'] == 8
                and fine['teacher_weight'] == 0 and fine['early_weight'] == .05
                and fine['relative_equation_target'] == 1e-6
                and fine['rollout_feedback'] == dict(mode=mode, weight=.05, depth_seed=2026101807),
                'Original matched fitting objective differs')
        require(sum(math.prod(item['shape']) for item in fine['initial_inventory'].values()) == 370016
                and all(item['dtype'] == '<f4' for item in fine['initial_inventory'].values())
                and fine['initial_weights_sha256'] == scores['C37']['weights_sha256'],
                'Original C37 initialization/architecture differs')
        if initial is not None:
            require(initial == fine['initial_inventory'], 'Arms do not share the complete initialization')
        initial = fine['initial_inventory']
        bank = receipt['fine_tuning_statistics']['fundamental_bank']
        feedback = receipt['fine_tuning_statistics']['rollout_feedback']
        require(len(bank['cases']) == 8 and bank['optimizer_examples'] == 1600
                and bank['expected_exposures'] == bank['optimizer_exposures'] == [200] * 8
                and len(fine['fitting_bank']['grids']) == 8
                and fine['fitting_bank']['modes_zyx'] == [[0, 0, 1]]
                and feedback['mode'] == mode and feedback['depth_exposures'] == [[50] * 4] * 8
                and feedback['auxiliary_losses'] == (0 if mode == 'U' else 1600),
                'Actual eight-pattern optimizer population/depth coverage differs')
        if sample_order is not None:
            require(sample_order == bank['sampling_order_sha256'] and depth_order == feedback['depth_order_sha256'],
                    'Matched pattern/depth ordering differs')
        sample_order, depth_order = bank['sampling_order_sha256'], feedback['depth_order_sha256']
        require(history['optimizer_work']['network_forwards'] == (6400 if mode == 'U' else 8000)
                and receipt['fine_tuning_statistics']['teacher_forwards'] == 0
                and len(population['parents']) == 16
                and all(parent['entry']['split'] == 'train' for parent in population['parents']),
                'Original counted forwards or calibration population differs')
        for name in ('initial_weights',):
            require(references[name]['sha256'] == fine['initial_weights_sha256'], 'Initial input identity differs')
        for name in ('weights.npz', 'checkpoint-1600.npz'):
            expected = receipt['outputs'][name]
            observed = binding(train / name)
            require(dict(sha256=observed['sha256'], bytes=(train / name).stat().st_size)
                    == {key: expected[key] for key in ('sha256', 'bytes')}, 'Final checkpoint bytes differ')
        score = scores[label]
        evidence = ROOT / score['result']['path']
        require(score['weights_sha256'] == receipt['outputs']['weights.npz']['sha256']
                and binding(evidence)['sha256'] == score['result']['sha256']
                and len(read(evidence)['rows']) == 128, 'Common accuracy selects another checkpoint')
        card = cards[fit]
        require(any(aliases[item]['label'] == label and aliases[item]['fit_item_id'] == fit
                    for item in card['named_entries']), 'Actual alias label differs from fit linkage')
        distinction = {'U': 'Retain the four-pass control without the auxiliary loss.',
                       'A': 'Add a one-correction loss at a balanced depth, with gradients through the carried pressure.',
                       'D': 'Use the same auxiliary loss but detach the carried pressure before the additional correction.'}[mode]
        worst = score['worst']
        row = dict(fit_item_id=fit, note_path=card['card_path'],
            method='A 370,016-parameter float32 3D pressure U-Net with encoder widths8/16/24/32, '
                   'bottleneck48 and reversed decoder; physical residual reconstruction is float64. '
                   'It starts from the exact C37 checkpoint. Training unrolls four pressure corrections '
                   'on one equation. These corrections are not future physical simulation steps.',
            purpose='Compare unchanged four-pass fine tuning with attached and detached auxiliary '
                    'gradient paths under the same initialization, pattern order, depth schedule and optimizer budget. ' + distinction,
            experiment='The optimizer uses eight synthetic analytical patterns: one [0,0,1] mode on each '
                       'of eight rectangular grids, at8or16cells/m. Each gets200exposures, with50at each '
                       'of four auxiliary depths. The original16train trajectories calibrate amplitude/normalization; '
                       'they are not the optimizer sample stream. No companion physical, direction or current-policy '
                       'loss runs in the captured fitting-bank updater. Teacher weight is zero and counted teacher '
                       'forwards are zero. All arms run1,600updates; U counts6,400network forwards and A/D8,000. '
                       'The loss combines analytical pressure loss, log-scaled final equation residual and '
                       '0.05mean early residual feedback; A/D add0.05auxiliary loss.',
            outcome=f'{label}, on the retained common128pressure requests: worst mean-free pressure '
                    f'relative L2={worst["pressure_mean_free_relative_l2"]:.17g}, equation relative L2='
                    f'{worst["common_equation_relative_l2"]:.17g}. The parent C37 values are '
                    '0.05382213638741031 and0.04891706555322945. These are pressure-only request '
                    'results, not trajectory drift, full-state accuracy or fresh qualified latency.',
            lesson='This matched experiment changes gradient flow, not architecture or time advancement. '
                   'Its effective fitting variety is eight single-mode patterns, despite retaining physical '
                   'datasets and broader feedback declarations. A has the lowest worst pressure error of '
                   'the three arms, while U has the lowest worst equation residual; no arm dominates both. '
                   'Training forward counts differ, so equal optimizer updates do not imply equal compute.',
            reproduction='Use the exact captured project.json/train-recipe.json and evidence.py entrypoint '
                         'with the original input bindings, source witness, C37 checkpoint, fitting bank '
                         'and fine-tuning.json. Clean retraining, fresh qualified latency and new-Rheon '
                         'data equivalence remain unqualified; no checkpoint can be retired on this review alone.',
            original_receipt=binding(train / 'receipt.json'), history=binding(train / 'history.json'),
            source_witness=binding(project / 'source-witness.json'), sources=sources,
            training_population=binding(train / 'splits.json'), training_inputs=inputs,
            retained_accuracy=binding(evidence), effective_optimizer_patterns=8,
            calibration_trajectories=16, network_forwards=history['optimizer_work']['network_forwards'])
        verified = verified_review_receipt(row)
        raw = (PROJECT / row['note_path']).read_bytes()
        require(raw.count(verified) == 1, 'Card does not retain the exact receipt once')
        _, suffix = revised_card(raw, row)
        require(verified in suffix, 'Review would discard original receipt')
        rows.append(row)
    report = dict(schema='rheon.captured-model-method-reviews.v1', created_unix=time.time(),
                  classification='authored_source_and_receipt_interpretation_not_retraining', rows=rows,
                  overview_addition='C75/C76/C77 are matched C37 four-correction fits with no/attached/detached '
                  'auxiliary gradients. Each optimizer sees eight single-mode synthetic patterns, not the retained '
                  'physical trajectory population; equal updates also have different forward counts.',
                  new_model_forwards=0, native_mutations=0, clean_reproduction_qualified=False,
                  producer=binding(Path(__file__)))
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='three_C37_detached_fit_source_reviews_complete', report=str(OUTPUT),
                         effective_patterns=8, calibration_trajectories=16)))


if __name__ == '__main__':
    run()
