"""Inspect native pressure fits with flat populations and balanced replay.

The earlier pressure-review helper owns uniform parent sampling. This distinct
protocol preserves 1,600 small-grid draws and interleaves 800 larger-grid draws.
Captured code is read as evidence, never imported or executed.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np

from review_water_learning_sequence import binding, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'


def replay_order(parents, seed):
    require(parents and all(p['split'] == 'train' for p in parents), 'Replay requires training-only parents')
    small = [i for i, p in enumerate(parents) if max(p['counts_xyz']) <= 16]
    require(len(small) == 16, 'Replay requires the original sixteen small parents')
    tiers = {}
    for i, p in enumerate(parents):
        size = max(p['counts_xyz'])
        if size > 16:
            tiers.setdefault(32 if size <= 32 else 64 if size <= 64 else 128, []).append(i)
    require(tiers, 'Replay requires at least one larger size tier')
    generators = [np.random.Generator(np.random.PCG64(seed + offset)) for offset in (0, 1)]
    # Build the private small sequence before interleaving: large draws cannot
    # consume RNG values belonging to the original small sequence.
    def draw(indices, generator):
        p = indices[int(generator.integers(len(indices)))]
        return p, int(generator.integers(len(parents[p]['frames'])))
    small_stream = [draw(small, generators[0]) for _ in range(1600)]
    order = []
    sizes = sorted(tiers)
    for n, item in enumerate(small_stream):
        order.append([item])
        if n % 2:
            order.append([draw(tiers[sizes[(n // 2) % len(sizes)]], generators[1])])
    value = np.asarray(order, dtype='<i8')
    require(value.shape == (2400, 1, 2), 'Replay update shape differs')
    return value, np.asarray(small_stream, dtype='<i8')


def check_tensors(path, inventory):
    result = {}
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == set(inventory), 'Complete pressure checkpoint membership required')
        for name, spec in inventory.items():
            value = arrays[name]
            require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                    and np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'],
                    'Original pressure parameter tensor differs')
            result[name] = value.copy()
    return result


def load_fit(identifier, *, work, material, index, score):
    entry = material[identifier]
    project = ROOT / entry['project']
    aliases = [a for a in work['named_entries'] if a['label'] == identifier and a['item_id'].startswith('cohorts/')]
    require(len(aliases) == 1, 'One exact named replay alias required')
    fit_id = aliases[0]['fit_item_id']
    fit = next(f for f in work['fit_cards'] if f['item_id'] == fit_id)
    candidates = [ROOT / p for p in fit['receipt_paths'] if (ROOT / p).is_file()
                  and binding(ROOT / p)['sha256'] == fit['receipt_sha256']]
    require(len(candidates) == 1, 'One original completed replay receipt required')
    receipt_path = candidates[0]
    training = receipt_path.parent
    receipt, config = read(receipt_path), read(project / 'config.json')
    witness, history = read(project / 'source-witness.json'), read(training / 'history.json')
    split, fine = read(training / 'splits.json'), read(project / 'fine-tuning.json')
    require(receipt['status'] == 'complete' and receipt['mode'] == 'train'
            and receipt['optimizer_updates'] == config['training']['updates'] == 2400,
            'Original replay update budget differs')
    for name, key in [('config.json','experiment_sha256'), ('source-witness.json','source_witness_sha256'),
                      ('runtime-manifest.json','runtime_manifest_sha256')]:
        require(binding(project / name)['sha256'] == receipt['closure'][key], 'Replay source/config/runtime identity differs')
    sources = [binding(project / name) for name in sorted(witness['sources'])]
    require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources)
            and witness['inputs'] == receipt['closure']['inputs'], 'Complete replay source/input closure differs')
    document, recipe = read(project / 'project.json'), read(project / 'train-recipe.json')
    require(recipe['entrypoint'] == str((project / 'evidence.py').relative_to(ROOT))
            and recipe['arguments'][0] == dict(literal='train')
            and recipe['runtime'] == 'managed_torch_python_v1', 'Original typed replay recipe differs')
    for spec in [*document['recipes'], *recipe['sources']]:
        require(binding(ROOT / spec['path'])['sha256'] == spec['sha256'], 'Typed replay source/recipe changed')
    inputs = {r['id']: ROOT / r['path'] for r in document['data']}
    bound_inputs = []
    for slot, expected in receipt['closure']['inputs'].items():
        p = inputs[slot]
        b = binding(p)
        require(dict(sha256=b['sha256'], bytes=p.stat().st_size) == expected, 'Original replay input bytes differ')
        bound_inputs.append(dict(b, slot=slot, bytes=expected['bytes']))
    locations = []
    for name, expected in receipt['outputs'].items():
        path = training / ('training-profile.json' if name == 'profile.json' else name)
        require(dict(sha256=binding(path)['sha256'], bytes=path.stat().st_size)
                == {k:expected[k] for k in ('sha256','bytes')}, 'Original replay output/profile differs')
        locations.append(dict(original_name=name, retained_path=str(path.relative_to(ROOT)), sha256=expected['sha256']))
    require(receipt['outputs']['weights.npz']['sha256'] == score[identifier]['weights_sha256']
            == entry['weights_sha256'] == aliases[0]['declared_weights_sha256'], 'Named replay checkpoint differs')
    require(binding(training / 'normalization.json')['sha256'] == receipt['closure']['normalization_sha256']
            and receipt['diagnostic_population'] == history['diagnostic_population'] == 'train_only'
            and receipt['heldout_evaluations'] == history['heldout_evaluations'] == 0
            and not receipt['scientific_acceptance_or_model_promotion'], 'Normalization or diagnostic scope differs')
    parents = split['parents']
    require(len(parents) in (26, 34) and all(p['dt_s'] == 1/60 and p['density_kg_m3'] == 1000
            and p['split'] == 'train' and len(p['frames']) == 2 for p in parents), 'Physical replay population differs')
    order, small_order = replay_order(parents, config['training']['sampling_seed'])
    require(hashlib.sha256(order.tobytes()).hexdigest() == receipt['closure']['sample_order_sha256'],
            'Independent balanced replay sample order differs')
    stats = history['current_feedback']['sampling']
    tier_exposure, parent_exposure, frame_exposure, cells = Counter(), Counter(), Counter(), 0
    for ((parent, frame),) in order:
        p, f = int(parent), int(frame)
        size = max(parents[p]['counts_xyz'])
        tier = 16 if size <= 16 else 32 if size <= 32 else 64 if size <= 64 else 128
        tier_exposure[str(tier)] += 1
        parent_exposure[p] += 1
        frame_exposure[f'{parents[p]["trajectory_id"]}:{parents[p]["frames"][f]["index"]}'] += 1
        cells += math.prod(parents[p]['counts_xyz'])
    require(dict(tier_exposure) == stats['tier_exposures'] and dict(frame_exposure) == stats['frame_exposures']
            and [parent_exposure[i] for i in range(len(parents))] == stats['parent_exposures']
            and cells == stats['sampled_cells']
            and hashlib.sha256(small_order.tobytes()).hexdigest() == stats['small_subsequence_sha256'],
            'Independent replay exposure/cell accounting differs')
    initial = history['checkpoints']['checkpoint-000.npz']['inventory']
    final = history['checkpoints']['checkpoint-2400.npz']['inventory']
    require(initial == fine['initial_inventory'] and binding(inputs['initial_weights'])['sha256'] == fine['initial_weights_sha256']
            and fine['training_passes'] == 2 and fine['teacher_weight'] == 0
            and fine['early_weight'] == .05 and fine['relative_equation_target'] == 1e-6,
            'Fine-tuning initialization/objective differs')
    initial_values = check_tensors(training / 'checkpoint-000.npz', initial)
    input_values = check_tensors(inputs['initial_weights'], initial)
    require(all(np.array_equal(input_values[n], initial_values[n]) for n in initial), 'Loaded warm tensors differ')
    final_values = check_tensors(training / 'weights.npz', final)
    require(receipt['outputs']['weights.npz']['sha256'] == receipt['outputs']['checkpoint-2400.npz']['sha256'],
            'Selected replay checkpoint must be the final update')
    trainer = (project / 'train.py').read_text()
    evidence = (project / 'evidence.py').read_text()
    require('return mixed_order.order(parents, config)' in trainer
            and 'analytical_samples = supplemental_samples = direction_samples = current_samples = None' in trainer
            and "'status':'inherited_auxiliary_banks_disabled'" in trainer
            and 'train.evaluate = training_diagnostic.diagnostic' in evidence
            and '(output / "profile.json").rename(output / "training-profile.json")' in evidence,
            'Executed replay sampler/disabled banks/native profile scope differs')
    card = next(f for f in index['fits'] if f['item_id'] == fit_id)
    require(card['receipt_sha256'] == fit['receipt_sha256'] and not card.get('method_review_status'),
            'Exact unreviewed replay card required')
    metadata = dict(fit_item_id=fit_id, note_path=card['card_path'], named_checkpoint=identifier,
        original_receipt=binding(receipt_path), history=binding(training / 'history.json'), sources=sources,
        source_witness=binding(project / 'source-witness.json'), original_configuration=binding(project / 'config.json'),
        original_recipe=binding(project / 'train-recipe.json'), training_population=binding(training / 'splits.json'),
        training_inputs=bound_inputs, original_output_locations=locations,
        parameters=sum(math.prod(v['shape']) for v in initial.values()), initialized_tensor_count=len(initial),
        actual_training_parents=len(parents), actual_training_frames=2*len(parents),
        actual_parent_densities=dict(Counter(str(1/p['cell_size_m']) for p in parents)),
        actual_parent_shapes=dict(Counter('x'.join(map(str,p['counts_xyz'])) for p in parents)),
        tier_exposures=dict(tier_exposure), sampled_cells=cells, optimizer_updates=2400,
        sample_order_independently_rederived=True, inherited_auxiliary_banks_executed=False,
        new_model_forwards=0, clean_reproduction_qualified=False)
    return dict(metadata=metadata, project=project, training=training, receipt=receipt, config=config,
        history=history, fine=fine, parents=parents, inputs=inputs, initial=initial_values,
        final=final_values, order=order, small_order=small_order)
