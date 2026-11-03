"""Review C125/C126/C127 from their captured head-only training closures.

Read existing sources, receipts and the small offline teacher dataset. Never
import a model, execute a forward, train, simulate, or change a native project.
The authored interpretation is separate from the preserved original evidence.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from prepare_method_review_revisions import revised_card, verified_review_receipt

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
BASE = ROOT / 'runs/full-ai-next-priorities-v1'
OUTPUT = DOC / 'reports/captured-model-method-reviews-04.json'


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def head(name):
    return name == 'coarse_logits' or name.startswith('shape_head.')


def run():
    if OUTPUT.exists():
        raise ValueError('A fresh authored review report is required')
    previous = read(DOC / 'reports/captured-model-method-reviews-03.json')
    index = read(ROOT / 'runs/model-portfolio-migration-v1/native-composition-primary-01/root/project/notes/portfolio-index.json')
    cards = {r['item_id']: r for r in index['fits']}
    scores = read(DOC / 'reports/pressure-common-score-01.json')['models']
    teacher_path = BASE / 'head-teacher-bank-01/manifest.json'
    teacher = read(teacher_path)
    archive = BASE / 'head-teacher-bank-01/teacher.npz'
    generator = ROOT / 'experiments/full-ai-next-priorities-v1/prepare_head_teacher.py'
    if (binding(archive)['sha256'] != teacher['archive_sha256']
            or archive.stat().st_size != teacher['archive_bytes']
            or binding(generator)['sha256'] != teacher['source_sha256']):
        raise ValueError('Original teacher or generator differs')
    # Dataset arrays only; neither checkpoint deserialization nor model import.
    with np.load(archive, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if set(arrays) != set(teacher['inventory']):
        raise ValueError('Teacher array membership differs')
    for name, value in arrays.items():
        entry = teacher['inventory'][name]
        if (value.dtype.str != entry['dtype'] or list(value.shape) != entry['shape']
                or hashlib.sha256(value.tobytes()).hexdigest() != entry['sha256']):
            raise ValueError('Teacher array identity differs: ' + name)
    geometry = {tuple(int(n) for n in shape):
                ('development' if int(split) else 'train', 'small' if int(small) else 'large')
                for shape, split, small in zip(arrays['counts_zyx'], arrays['split'], arrays['tier_small'])}
    counts = Counter(phase + '_' + tier for phase, tier in geometry.values())
    if (len(geometry) != 6728 or counts != teacher['counts']
            or counts != dict(train_small=1526, train_large=4478,
                              development_small=202, development_large=522)
            or int(arrays['counts_zyx'].min()) != 5 or int(arrays['counts_zyx'].max()) != 128):
        raise ValueError('Original whole-geometry partition differs')
    rows, witnesses = [], []
    initial_reference = None
    for label, arm, fit in (
        ('C125', 'a-head-calibration', 'fit-c80b7d18912c7d9d'),
        ('C126', 'a-head-calibration-s2', 'fit-ddb5458432c1351b'),
        ('C127', 'a-head-calibration-s3', 'fit-a8e19469f5232c7e'),
    ):
        project = BASE / (arm + '-project-01')
        train = BASE / (arm + '-pair-01/train')
        restore_path = BASE / (arm + '-pair-01/restore/receipt.json')
        receipt, history = read(train / 'receipt.json'), read(train / 'history.json')
        witness = read(project / 'source-witness.json')
        manifest, recipe = read(project / 'project.json'), read(project / 'train-recipe.json')
        calibration, restore = read(project / 'calibration.json'), read(restore_path)
        sources = [binding(project / name) for name in sorted(witness['sources'])]
        if (len(sources) != 53
                or any(s['sha256'] != witness['sources'][Path(s['path']).name] for s in sources)
                or {Path(s['path']).name: s['sha256'] for s in sources if Path(s['path']).suffix == '.py'} != receipt['closure']['sources']
                or len(receipt['closure']['sources']) != 44
                or binding(project / 'config.json')['sha256'] != receipt['closure']['experiment_sha256']
                or binding(project / 'calibration.json')['sha256'] != receipt['closure']['calibration_sha256']
                or witness['inputs'] != receipt['closure']['inputs']):
            raise ValueError('Captured source or static closure differs')
        for entry in [*manifest['recipes'], *recipe['sources']]:
            if binding(ROOT / entry['path'])['sha256'] != entry['sha256']:
                raise ValueError('Captured recipe source differs')
        if (len(recipe['sources']) != 56
                or recipe['entrypoint'] != str((project / 'head_train.py').relative_to(ROOT))
                or recipe['arguments'][0] != {'literal': 'train'}
                or {'source': str((project / 'calibration.json').relative_to(ROOT))} not in recipe['arguments']):
            raise ValueError('Actual head-only recipe differs')
        inputs = [dict(slot=e['id'], **binding(ROOT / e['path']), bytes=(ROOT / e['path']).stat().st_size)
                  for e in manifest['data']]
        if len(inputs) != 6 or {e['slot'] for e in inputs} != set(receipt['closure']['inputs']):
            raise ValueError('Actual typed training inputs differ')
        initial, final = receipt['initial_inventory'], receipt['final_inventory']
        if (initial != history['initial_inventory'] or final != history['final_inventory']
                or set(initial) != set(final)
                or sum(math.prod(v['shape']) for v in initial.values()) != 93758
                or sum(math.prod(v['shape']) for k, v in initial.items() if head(k)) != 1198
                or len([k for k in initial if head(k)]) != 5
                or {k: v for k, v in initial.items() if not head(k)} != {k: v for k, v in final.items() if not head(k)}
                or not receipt['fine_tensors_bitwise_unchanged']):
            raise ValueError('Actual parameter inventories or frozen fine tensors differ')
        if initial_reference is not None and initial_reference != initial:
            raise ValueError('Replicates do not share the exact complete initial inventory')
        initial_reference = initial
        if (receipt['parameters'] != 93758 or receipt['trainable_parameters'] != 1198
                or receipt['optimizer_updates'] != 10000
                or history['optimizer_work'] != dict(head_forwards=10000, fine_spatial_forwards_during_fitting=0,
                                                     teacher_examples=3840000)
                or history['physical_reference_fields_used'] or calibration['batch_size'] != 384
                or calibration != receipt['architecture']['head_calibration']
                or restore['optimizer_updates'] != 0 or not restore['full_cpu_probe_bitwise_verified']
                or restore['outputs']['weights.npz'] != receipt['outputs']['weights.npz']):
            raise ValueError('Executed calibration/restore budget differs')
        initial_input = next(i for i in inputs if i['slot'] == 'initial_weights')
        if initial_input['sha256'] != scores['C107']['weights_sha256']:
            raise ValueError('Calibration did not initialize from the exact scored C107 checkpoint')
        score = scores[label]
        result_path = ROOT / score['result']['path']
        if (score['weights_sha256'] != receipt['outputs']['weights.npz']['sha256']
                or binding(result_path)['sha256'] != score['result']['sha256']):
            raise ValueError('Common pressure evidence selects another checkpoint')
        cases = read(result_path)['rows']
        partition = Counter(geometry[tuple(reversed(case['counts_xyz']))][0] for case in cases)
        if len(cases) != 128 or partition != dict(train=118, development=10):
            raise ValueError('Common pressure teacher-geometry partition differs')
        worst = score['worst']
        row = dict(fit_item_id=fit, note_path=cards[fit]['card_path'],
            original_receipt=binding(train / 'receipt.json'), history=binding(train / 'history.json'),
            source_witness=binding(project / 'source-witness.json'), sources=sources, training_inputs=inputs,
            data_generation_source=dict(**binding(generator), manifest_slot='teacher_manifest'),
            parameters=93758, trainable_parameters=1198, frozen_parameters=92560, optimizer_updates=10000,
            training_geometries=6004, development_geometries=724, physical_training_scenes=0,
            common_pressure_teacher_geometry_partition=dict(partition),
            seed=calibration['seed'], sampling_seed=calibration['sampling_seed'],
            sampled_geometry_order_sha256=history['sampled_geometry_order_sha256'],
            calibration=binding(project / 'calibration.json'),
            reproduction_recipes=[binding(project / 'train-recipe.json'), binding(project / 'restore-recipe.json')],
            historical_restore=binding(restore_path), common_pressure_evidence=binding(result_path))
        row['method'] = (
            f'{label} keeps the complete 93,758-parameter C107 pressure graph and starts from its exact '
            'checkpoint. It freezes 92,560 fine-network parameters and calibrates only the five existing '
            'coarse-head tensors (1,198 parameters). The head adds a 3→16 tanh→63 shape network to63 '
            'modal logits; sigmoid gives positive gains bounded by max-axis-cells²/2. Separable cosine '
            'modes0..3 excluding DC restrict/prolong the physical residual in float64. Before each of '
            'two float32 fine-network corrections, a coarse proposal is accepted only if its actual '
            'fine-equation residual norm does not increase. This predicts pressure for one request, '
            'not velocity, tracer, future endpoints or simultaneous physical steps.')
        row['purpose'] = (
            'Test whether offline geometry-teacher calibration improves the C107 shape-dependent '
            'global response while preserving its fine tensors and inference graph. The three fits '
            'are siblings from the same complete initial inventory; the different torch seeds do '
            'not create different loaded initial weights. Their private sampling seeds differ. '
            'They are not a sequential continuation or three new architectures.')
        row['experiment'] = (
            'Actual recipe entrypoint head_train.py uses calibration.json, not the inherited scene '
            'trainer/configuration. The offline teacher has6,728 unique Z,Y,X geometries: all1,728 '
            'triples with axes5..16 plus5,000 sampled larger triples with axes up to128. Whole-geometry '
            'hashing gives1,526 small+4,478 large training geometries and202 small+522 large development '
            'geometries. Labels are log of the analytical sealed-Neumann modal inverse gain divided '
            'by max-axis-cells²/2; no physical fields or Rheon trajectories train this calibration. '
            '10,000 CUDA Adam updates each sample256 small+128 large geometries with replacement: '
            '3,840,000 teacher examples and zero fine spatial-network forwards during fitting. '
            'Loss is mean squared log-gain error plus0.1 times mean per-example maximum squared '
            'mode error; learning rate.001→.0003 at6001→.0001 at9001, gradient clipping1, deterministic '
            'float32 fitting, TF32/AMP disabled. Fixed physical pressure normalization remains C107’s.')
        row['outcome'] = (
            f'The retained common128-pressure-request diagnostic gives worst mean-free pressure '
            f'relative L2 {worst["pressure_mean_free_relative_l2"]:.17g} and worst equation relative L2 '
            f'{worst["common_equation_relative_l2"]:.17g}. Final teacher-development loss is '
            f'{history["history"][-1]["teacher_development_loss"]:.17g}; this is a distinct metric. '
            'The common pressure requests cover small grids at8/16cells per metre and dt1/60; '
            '118 have teacher-training geometries and10 teacher-development geometries. Geometry '
            'exposure is not a physical-scene held-out split. Exact historical zero-update CPU '
            'restore and bitwise probe evidence is retained; this review executes no new inference.')
        row['lesson'] = (
            'All three improve this diagnostic over C107’s worst pressure0.00815329, with modest '
            'seed variation around0.0024; C126 is best of these three on both worst pressure and '
            'equation error. Training on large geometry labels does not prove accurate physical '
            'pressure solves on large domains. No speed improvement follows from a calibration '
            'that keeps the same inference graph, and the coarse acceptance gate does not bound '
            'the final fine correction or own-history drift. Preserve separate teacher loss, '
            'pressure-field error, equation residual, runtime and rollout claims.')
        row['reproduction'] = (
            'Retain the original53-file witness, exact C107 initialization/normalization, six typed '
            'inputs, calibration and train/restore recipes. The teacher manifest binds the original '
            'prepare_head_teacher.py and all seven array inventories. Recreate a fresh admitted '
            'teacher output instead of overwriting the guarded bank. Original recipes include CPU '
            'deployment/restore probes; current reproduction must use separately identified '
            'GPU-only probe executions to respect the active inference policy. Clean retraining '
            'and complete-generator reproduction remain unqualified. New Rheon reference epochs '
            'must remain separate from these older pressure requests.')
        verified_review_receipt(row)
        revised_card((ROOT / 'runs/model-portfolio-migration-v1/native-composition-primary-01/root/project' / row['note_path']).read_bytes(), row)
        rows.append(row)
        witnesses.append(witness['sources'])
    differences = {name for name in witnesses[0] if any(w[name] != witnesses[0][name] for w in witnesses[1:])}
    if differences != {'config.json', 'calibration.json'}:
        raise ValueError('Replicate executable source differs')
    overview = previous['overview_addition'].replace(
        'Eight of 205 fit cards have retrospective source reviews prepared:',
        'Eleven of 205 fit cards have retrospective source reviews prepared:').replace(
        'parent, C106/C107 and the three C128/C129/C130 pressure-tail calibrations.',
        'parent, C106/C107, the three C125/C126/C127 head calibrations and the three '
        'C128/C129/C130 pressure-tail calibrations.')
    overview += ('\nC125/C126/C127 calibrate the existing1,198-parameter C107 coarse head, with '
                 '92,560 frozen fine parameters, using6,004 training and724 development teacher '
                 'geometries. All three start from the exact same C107 weights. Their teacher '
                 'covers axes through128, but the retained common pressure evidence covers small '
                 'grids only. This is separate from the later conditional511-mode tails.\n')
    output = dict(schema='rheon.captured-model-method-reviews.v1', created_unix=time.time(),
        classification='authored_retrospective_interpretation_original_evidence_preserved',
        rows=previous['rows'] + rows, overview_addition=overview,
        head_teacher_array_inventories_verified=7, head_replicate_executable_sources_equal=True,
        head_replicate_source_differences=sorted(differences),
        new_model_forwards=0, new_parameter_updates=0, native_project_written=False, payload_copies=0)
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='eleven_source_reviews_ready_for_guarded_note_preparation',
                          cards=len(output['rows']), new_cards=len(rows))), flush=True)


if __name__ == '__main__':
    run()
