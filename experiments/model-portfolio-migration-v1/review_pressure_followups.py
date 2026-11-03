"""Source-only interpretation of six C107 follow-ups, preserving original facts."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from prepare_method_review_revisions import revised_card, verified_review_receipt

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
BASE = ROOT / 'runs/full-ai-next-priorities-v1'
PRIMARY = ROOT / 'runs/model-portfolio-migration-v1/native-composition-primary-01/root/project'
OUTPUT = DOC / 'reports/captured-model-method-reviews-05.json'


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def optimizer_multiplier(path):
    """Inspect the actual two-group literal; never import captured model code."""
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'groups')
    returned = next(n.value for n in function.body if isinstance(n, ast.Return))
    if not isinstance(returned, ast.List) or len(returned.elts) != 2:
        raise ValueError('Expected original disjoint fine/head optimizer groups')
    groups = [dict((k.arg, k.value) for k in entry.keywords) for entry in returned.elts]
    return [ast.literal_eval(group['lr_multiplier']) for group in groups]


def run():
    if OUTPUT.exists():
        raise ValueError('Fresh authored source review required')
    previous = read(DOC / 'reports/captured-model-method-reviews-04.json')
    cards = {r['item_id']: r for r in read(PRIMARY / 'notes/portfolio-index.json')['fits']}
    scores = read(DOC / 'reports/pressure-common-score-01.json')['models']
    specs = [
        ('C119', 'a-control', '02', '02', 'fit-ddaf39b5dbadb100', 93758, 1200, 2, False, 10/3,
         'Provide a further C107 continuation control on recorded physical requests, with the original10/3 '
         'coarse-head learning-rate multiplier and unweighted train-frame sampling.'),
        ('C120', 'a-head', '01', '01', 'fit-aeef69c065d09a85', 93758, 1200, 2, False, 1.,
         'Test a lower coarse-head learning rate: use the fine-network rate rather than10/3 times '
         'that rate, paired with C119’s initial weights and sample order. This trains the full '
         'network; it is not the later head-only offline geometry calibration.'),
        ('C121', 'a-tail', '01', '01', 'fit-5abc12d93143490c', 93758, 1200, 2, True, 10/3,
         'Emphasize train frames ranked difficult by the frozen C107 diagnostic, while retaining '
         'the10/3 head rate. Here tail means hard-case resampling, not an added modal-tail network.'),
        ('C122', 'a-combined', '01', '01', 'fit-ac2153e16dfaa0f7', 93758, 1200, 2, True, 1.,
         'Combine the reduced head learning rate and hard-case train-frame sampling; compare '
         'with C119/C120/C121 as a small paired intervention grid.'),
        ('C123', 'a-redistributed', '02', '01', 'fit-844a28a6c7d29068', 137227, 1200, 2, True, 1.,
         'Test redistributing spatial channels toward deeper levels: encoder[3,8,16,20], '
         'bottleneck28, reversed decoder. Explicit channel duplication/pruning maps C107 into '
         'a new137,227-parameter initialization. This is a different architecture and more '
         'parameters than C107, despite the pruning of its first level.'),
        ('C124', 'a-one-stage', '01', '01', 'fit-2f3e853153936dd0', 126542, 2400, 1, True, 1.,
         'Test one internal pressure correction with a larger learned global response:511 '
         'non-DC modes0..7 and a3→64 tanh→511 shape head. The fine CNN remains C107’s width. '
         'Two adjacent repetitions of each original sampled frame give2,400 optimizer updates '
         'and2,400 fine-network forwards, matching the two-stage1,200-update arms’ forward '
         'count. This is one pressure solve, not one or more physical future steps.'),
    ]
    rows, observed = [], {}
    for label, arm, project_number, study_number, fit, parameters, updates, stages, tail, multiplier, purpose in specs:
        project = BASE / (arm + '-project-' + project_number)
        train = BASE / (arm + '-study-' + study_number) / 'train'
        receipt, history = read(train / 'receipt.json'), read(train / 'history.json')
        witness, config = read(project / 'source-witness.json'), read(project / 'config.json')
        fine, sampling = read(project / 'fine-tuning.json'), read(project / 'fit-sampling.json')
        manifest, recipe = read(project / 'project.json'), read(project / 'train-recipe.json')
        population = read(train / 'splits.json')
        variant = label in ('C123', 'C124')
        expected_sources, expected_recipes = (49, 52) if variant else (46, 49)
        sources = [binding(project / name) for name in sorted(witness['sources'])]
        if (len(sources) != expected_sources
                or any(s['sha256'] != witness['sources'][Path(s['path']).name] for s in sources)
                or binding(project / 'source-witness.json')['sha256'] != receipt['closure']['source_witness_sha256']
                or binding(project / 'config.json')['sha256'] != receipt['closure']['experiment_sha256']
                or binding(project / 'runtime-manifest.json')['sha256'] != receipt['closure']['runtime_manifest_sha256']
                or witness['architecture'] != receipt['architecture']
                or witness['inputs'] != receipt['closure']['inputs']
                or len(witness['inputs']) != 12
                or sum(Path(s['path']).suffix == '.py' for s in sources) != (40 if variant else 38)):
            raise ValueError('Original executed source closure differs: ' + label)
        if (recipe['entrypoint'] != str((project / 'evidence.py').relative_to(ROOT))
                or len(recipe['sources']) != expected_recipes):
            raise ValueError('Original recipe entrypoint or source membership differs')
        for entry in [*manifest['recipes'], *recipe['sources']]:
            if binding(ROOT / entry['path'])['sha256'] != entry['sha256']:
                raise ValueError('Captured recipe source differs')
        refs = {e['id']: e for e in manifest['data']}
        if len(refs) != 14 or any(refs[slot]['sha256'] != item['sha256'] for slot, item in witness['inputs'].items()):
            raise ValueError('Declared recipe inputs contradict the fit closure')
        initial_path = ROOT / refs['initial_weights']['path']
        initial_binding = dict(slot='initial_weights', **binding(initial_path), bytes=initial_path.stat().st_size)
        if (initial_binding['sha256'] != fine['initial_weights_sha256']
                or history['checkpoints']['checkpoint-000.npz']['sha256'] != initial_binding['sha256']
                or history['checkpoints']['checkpoint-000.npz']['inventory'] != fine['initial_inventory']
                or sum(math.prod(v['shape']) for v in fine['initial_inventory'].values()) != parameters):
            raise ValueError('Actual initialized parameter population differs')
        actual_multiplier = optimizer_multiplier(project / 'head_optimizer.py')
        if actual_multiplier != [1., multiplier] or "__import__('head_optimizer').groups" not in (project / 'train.py').read_text():
            raise ValueError('Executed optimizer does not use the reviewed head groups')
        parents = population['parents']
        if (len(parents) != 34 or sum(len(p['frames']) for p in parents) != 68
                or any(p['split'] != 'train' or p['dt_s'] != 1/60 or p['density_kg_m3'] != 1000 for p in parents)
                or receipt['optimizer_updates'] != updates or receipt['heldout_evaluations'] != 0
                or config['training']['updates'] != updates or config['training']['batch_size'] != 1
                or config['correction']['steps'] != stages or fine['training_passes'] != stages
                or fine['teacher_weight'] != 0 or sampling['emphasize_tail'] != tail
                or not sampling['training_only'] or receipt['fine_tuning_statistics']['distillation_teacher_forwards'] != 0
                or receipt['fine_tuning_statistics']['teacher_forwards'] != 0
                or history['current_feedback']['status'] != 'inherited_auxiliary_banks_disabled'
                or history['optimizer_work']['network_forwards'] != 2400
                or history['current_feedback']['sampling']['sample_order_sha256'] != receipt['closure']['sample_order_sha256']):
            raise ValueError('Executed population, sampling or optimizer budget differs')
        score = scores[label]
        result_path = ROOT / score['result']['path']
        if (score['weights_sha256'] != receipt['outputs']['weights.npz']['sha256']
                or binding(result_path)['sha256'] != score['result']['sha256']
                or len(read(result_path)['rows']) != 128):
            raise ValueError('Common pressure evidence selects another checkpoint')
        row = dict(fit_item_id=fit, note_path=cards[fit]['card_path'],
            original_receipt=binding(train / 'receipt.json'), history=binding(train / 'history.json'),
            source_witness=binding(project / 'source-witness.json'), sources=sources,
            training_population=binding(train / 'splits.json'), training_inputs=[initial_binding],
            parameters=parameters, trainable_parameters=parameters, optimizer_updates=updates,
            pressure_correction_passes=stages, fine_network_forwards=2400,
            training_parents=34, selected_training_frames=68,
            recorded_input_slot_declarations_reconciled=12,
            actual_initial_weights_rehashed=True,
            training_densities_cells_per_metre=dict(Counter(str(1/p['cell_size_m']) for p in parents)),
            shape_counts_xyz=dict(Counter(str(p['counts_xyz']) for p in parents)),
            runtime_contract=binding(project / 'runtime-manifest.json'),
            reproduction_recipes=[binding(project / 'train-recipe.json'), binding(project / 'restore-recipe.json')],
            common_pressure_evidence=binding(result_path),
            declared_head_lr_multiplier=receipt['architecture']['shape_informed_coarse']['head_learning_rate_multiplier'],
            executed_head_lr_multiplier=multiplier,
            clean_reproduction_qualified=False, fresh_complete_generator_benchmark_qualified=False)
        shape = receipt['architecture']
        row['method'] = (
            f'{label} is a pressure-only residual predictor with{parameters:,} trained parameters, '
            f'encoder widths{shape["encoder_widths"]}, bottleneck{shape["bottleneck_width"]}, reversed '
            f'decoder and{stages} internal pressure correction pass(es). The fine path uses bias-free '
            'linear convolutions, replicated boundaries, average pooling, nearest upsampling, skip '
            'concatenation and a global pooled addition. Input is normalized physical residual '
            'q=(rho/h)r. Pressure and equation arithmetic use float64 around the float32 CNN. '
            'A shape-conditioned positive learned cosine-mode proposal precedes each fine pass '
            'and is accepted only if its actual fine-equation residual norm does not increase. '
            'The gate does not bound the final neural correction. The model does not advance '
            'velocity/tracer or generate simultaneous future physical states.')
        row['purpose'] = purpose
        initial_description = ('the same exact C107 checkpoint' if not variant else
                               'an explicitly transformed C107-derived checkpoint with its own inventory')
        draw_description = ('1,600 small and800 larger draws, by repeating each original sampled pair twice'
                            if stages == 1 else '800 small and400 larger draws')
        row['experiment'] = (
            f'Initialize from{initial_description}: {initial_binding["sha256"]}. Use{updates:,} '
            f'CUDA Adam updates, batch1 and{draw_description}; larger draws rotate through32/64/128 '
            'cell tiers. The34 captured train-only parents provide68 selected frames at8/16/32 '
            'cells per metre, dt1/60s and fluid density1000kg/m³; the domains include rectangular '
            'shapes and cubes through128³. Hard-case sampling uses frozen training-only rank '
            f'weights up to4 and is {"enabled" if tail else "disabled"}. The actual head rate '
            f'multiplier is{multiplier:.17g}, while inherited architecture metadata says '
            f'{row["declared_head_lr_multiplier"]}; preserve this discrepancy. Fine rate3e-5 '
            f'falls to1e-5 at update{1801 if stages == 1 else 901}; seed2026101907, private '
            'sampling seed2026101908 and fixed C53 normalization. The canonical physical loss '
            'adds logarithmic final equation feedback and total0.05 earlier-stage feedback '
            'where applicable. Inherited auxiliary banks are disabled and both teacher-forward '
            'counts are zero. All arms record2,400 actual optimizer-loss CNN forwards; forward-cell '
            'counts differ by sampling. These are parent/frame counts, not proof of independent '
            'scenes. Trainer held-out evaluations are zero; the evidence wrapper’s historical '
            'post-fit consumed pressure probe does not create an independent validation split.')
        worst = score['worst']
        row['outcome'] = (
            f'On the same128 original consumed pressure requests, worst mean-free pressure '
            f'relative L2 is{worst["pressure_mean_free_relative_l2"]:.17g}, and worst equation '
            f'relative L2 is{worst["common_equation_relative_l2"]:.17g}. C119–C122 range about '
            '0.00752–0.00797 pressure error, C123 reaches0.01583 and C1240.04810. The control '
            'has the lowest worst pressure of those four paired arms; the combined arm has '
            'the lowest worst equation error among them. The larger redistributed and '
            'one-stage graphs do worse on these tails. The common requests cover only '
            'small grids at8/16cells per metre and dt1/60s. No larger-grid physical accuracy, '
            'complete-generator drift or freshly qualified speed follows from this review.')
        row['lesson'] = (
            'Reducing a head learning rate, resampling hard cases, changing channel allocation '
            'or increasing global modes does not guarantee better worst pressure error. '
            'Pressure-field and equation tails can rank the same four controlled arms '
            'differently. Equal CNN-forward counts do not imply equal FLOPs, memory traffic '
            'or measured latency when architectures and sampled grids differ. The one-stage '
            'accuracy loss is a real same-request result; its saved pressure pass is distinct '
            'from multistep physical-time prediction. Keep the old solver epoch and preserve '
            'negative outcomes rather than promote an attractive architecture label.')
        row['reproduction'] = (
            f'The original{expected_sources}-file witness,{expected_recipes} recipe source bindings, '
            'recorded twelve fitting-input declarations, exact rehashed initialization, train '
            'population, original train/restore recipes and runtime contract are bound below '
            'or in this review report. Dataset/input archives are not freshly decoded by this '
            'source-only review; clean reproduction must verify and consume their exact '
            'original bytes under the pinned old physics contract. Captured recipes include '
            'historical CPU deployment probes; follow current GPU-only inference policy via '
            'a separately identified execution rather than silently changing an original Run. '
            'Opening/importing a project must remain inert. Clean retraining, independent '
            'generalization and end-to-end reproduction remain unverified.')
        verified_review_receipt(row)
        revised_card((PRIMARY / row['note_path']).read_bytes(), row)
        rows.append(row)
        observed[label] = dict(receipt=receipt, sampling=sampling, initial=fine['initial_inventory'], population=population)
    for label in ('C120', 'C121', 'C122'):
        if (observed[label]['initial'] != observed['C119']['initial']
                or observed[label]['population'] != observed['C119']['population']):
            raise ValueError('Four-arm comparison does not share initial inventory/population')
    if (observed['C119']['receipt']['closure']['sample_order_sha256'] != observed['C120']['receipt']['closure']['sample_order_sha256']
            or observed['C121']['receipt']['closure']['sample_order_sha256'] != observed['C122']['receipt']['closure']['sample_order_sha256']):
        raise ValueError('Head-rate pairs do not share sample order')
    overview = previous['overview_addition'].replace('Eleven of 205', 'Seventeen of 205')
    overview += ('\nSix C119–C124 follow-ups now distinguish coarse-head learning-rate changes, '
                 'hard-case resampling, deeper channel redistribution and a larger-head one-pass '
                 'pressure predictor. The first four share the exact C107 initialization; '
                 'the two architecture variants bind separate transformed initializations. '
                 'All record2,400 fine-network forwards, with different sampling/architecture '
                 'costs. The paired improvements are small and metric-dependent; both architecture '
                 'variants worsen the common small-grid pressure tails. No new speed or '
                 'complete-generator qualification follows from this source review.\n')
    output = dict(schema='rheon.captured-model-method-reviews.v1', created_unix=time.time(),
        classification='authored_retrospective_interpretation_original_evidence_preserved',
        rows=previous['rows'] + rows, overview_addition=overview,
        reviewed_pressure_followups=6, same_first_four_initial_inventory=True,
        same_first_four_training_population=True, head_rate_pairs_same_sample_order=True,
        new_model_forwards=0, new_parameter_updates=0, native_project_written=False, payload_copies=0)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status='seventeen_source_reviews_ready_for_guarded_note_preparation',
                          cards=len(output['rows']), new_cards=len(rows))), flush=True)


if __name__ == '__main__':
    run()
