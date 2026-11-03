"""Read source/data/tensor custody for expanded native pressure-fit reviews.

Callers own architecture and objective interpretation. This reader accepts an
explicit expected parent/parameter population; it never imports captured code,
decodes simulation fields or evaluates a neural network/numerical hierarchy.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np

from captured_replay_pressure_review import check_tensors, replay_order
from review_water_learning_sequence import binding, read, require

ROOT = Path(__file__).resolve().parents[2]


def load_fit(identifier, *, expected_parents, expected_parameters, work, material, index, score):
    entry = material[identifier]
    project = ROOT / entry['project']
    aliases = [a for a in work['named_entries'] if a['label'] == identifier and a['item_id'].startswith('cohorts/')]
    require(len(aliases) == 1, 'One exact named physical-pressure alias required')
    fit_id = aliases[0]['fit_item_id']
    fit = next(f for f in work['fit_cards'] if f['item_id'] == fit_id)
    receipts = [ROOT / p for p in fit['receipt_paths'] if (ROOT / p).is_file()
                and binding(ROOT / p)['sha256'] == fit['receipt_sha256']]
    require(len(receipts) == 1, 'One original physical-pressure training receipt required')
    receipt_path = receipts[0]
    training = receipt_path.parent
    receipt, history = read(receipt_path), read(training / 'history.json')
    config, witness = read(project / 'config.json'), read(project / 'source-witness.json')
    split, fine = read(training / 'splits.json'), read(project / 'fine-tuning.json')
    require(receipt['status'] == 'complete' and receipt['mode'] == 'train'
            and receipt['optimizer_updates'] == config['training']['updates'] == 2400,
            'Physical-pressure fit did not complete the declared budget')
    for name, key in [('config.json','experiment_sha256'), ('source-witness.json','source_witness_sha256'),
                      ('runtime-manifest.json','runtime_manifest_sha256')]:
        require(binding(project / name)['sha256'] == receipt['closure'][key], 'Original physical-pressure source/config/runtime differs')
    sources = [binding(project / n) for n in sorted(witness['sources'])]
    require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources)
            and witness['inputs'] == receipt['closure']['inputs'], 'Complete physical-pressure source/input closure differs')
    document, recipe = read(project / 'project.json'), read(project / 'train-recipe.json')
    require(recipe['entrypoint'] == str((project / 'evidence.py').relative_to(ROOT))
            and recipe['arguments'][0] == dict(literal='train')
            and recipe['runtime'] == 'managed_torch_python_v1', 'Typed physical-pressure recipe differs')
    for spec in [*document['recipes'], *recipe['sources']]:
        require(binding(ROOT / spec['path'])['sha256'] == spec['sha256'], 'Physical-pressure typed source/recipe changed')
    inputs = {r['id']:ROOT / r['path'] for r in document['data']}
    bound_inputs = []
    for slot, expected in receipt['closure']['inputs'].items():
        p = inputs[slot]
        b = binding(p)
        require(dict(sha256=b['sha256'], bytes=p.stat().st_size) == expected, 'Original physical-pressure input bytes differ')
        bound_inputs.append(dict(b, slot=slot, bytes=expected['bytes']))
    locations = []
    for name, expected in receipt['outputs'].items():
        p = training / ('training-profile.json' if name == 'profile.json' else name)
        require(dict(sha256=binding(p)['sha256'], bytes=p.stat().st_size)
                == {k:expected[k] for k in ('sha256','bytes')}, 'Original physical-pressure output/profile differs')
        locations.append(dict(original_name=name, retained_path=str(p.relative_to(ROOT)), sha256=expected['sha256']))
    require(receipt['outputs']['weights.npz']['sha256'] == score[identifier]['weights_sha256']
            == entry['weights_sha256'] == aliases[0]['declared_weights_sha256'], 'Named physical-pressure checkpoint differs')
    require(binding(training / 'normalization.json')['sha256'] == receipt['closure']['normalization_sha256']
            and receipt['diagnostic_population'] == history['diagnostic_population'] == 'train_only'
            and receipt['heldout_evaluations'] == history['heldout_evaluations'] == 0
            and not receipt['scientific_acceptance_or_model_promotion'], 'Physical-pressure normalization/diagnostic scope differs')
    parents = split['parents']
    require(len(parents) == expected_parents and all(p['split'] == 'train' and p['dt_s'] == 1/60
            and p['density_kg_m3'] == 1000 and len(p['frames']) in (2,4) for p in parents),
            'Explicit physical-pressure parent population differs')
    order, small_order = replay_order(parents, config['training']['sampling_seed'])
    require(hashlib.sha256(order.tobytes()).hexdigest() == receipt['closure']['sample_order_sha256'],
            'Independent expanded sample order differs')
    stats = history['current_feedback']['sampling']
    tiers, exposures, frames, cells = Counter(), Counter(), Counter(), 0
    for ((parent, frame),) in order:
        p, f = int(parent), int(frame)
        size = max(parents[p]['counts_xyz'])
        tier = 16 if size <= 16 else 32 if size <= 32 else 64 if size <= 64 else 128
        tiers[str(tier)] += 1
        exposures[p] += 1
        frames[f'{parents[p]["trajectory_id"]}:{parents[p]["frames"][f]["index"]}'] += 1
        cells += math.prod(parents[p]['counts_xyz'])
    require(dict(tiers) == stats['tier_exposures'] and dict(frames) == stats['frame_exposures']
            and [exposures[i] for i in range(len(parents))] == stats['parent_exposures']
            and cells == stats['sampled_cells']
            and hashlib.sha256(small_order.tobytes()).hexdigest() == stats['small_subsequence_sha256'],
            'Independent expanded exposure accounting differs')
    initial = history['checkpoints']['checkpoint-000.npz']['inventory']
    final = history['checkpoints']['checkpoint-2400.npz']['inventory']
    require(sum(math.prod(v['shape']) for v in initial.values()) == expected_parameters
            and initial == fine['initial_inventory']
            and binding(inputs['initial_weights'])['sha256'] == fine['initial_weights_sha256'],
            'Explicit physical-pressure initial parameter inventory differs')
    initial_values = check_tensors(training / 'checkpoint-000.npz', initial)
    warm_values = check_tensors(inputs['initial_weights'], initial)
    require(all(initial_values[n].tobytes() == warm_values[n].tobytes() for n in initial), 'Loaded physical-pressure warm values differ')
    final_values = check_tensors(training / 'weights.npz', final)
    require(receipt['outputs']['weights.npz']['sha256'] == receipt['outputs']['checkpoint-2400.npz']['sha256'],
            'Selected physical-pressure weights must equal final-update archive')
    trainer, evidence = (project / 'train.py').read_text(), (project / 'evidence.py').read_text()
    require('return mixed_order.order(parents, config)' in trainer
            and 'analytical_samples = supplemental_samples = direction_samples = current_samples = None' in trainer
            and 'train.evaluate = training_diagnostic.diagnostic' in evidence
            and '(output / "profile.json").rename(output / "training-profile.json")' in evidence,
            'Executed physical-pressure sampling/bank/diagnostic scope differs')
    card = next(c for c in index['fits'] if c['item_id'] == fit_id)
    require(card['receipt_sha256'] == fit['receipt_sha256'] and not card.get('method_review_status'),
            'Original unreviewed physical-pressure card required')
    metadata = dict(fit_item_id=fit_id, note_path=card['card_path'], named_checkpoint=identifier,
        original_receipt=binding(receipt_path), history=binding(training / 'history.json'), sources=sources,
        source_witness=binding(project / 'source-witness.json'), original_configuration=binding(project / 'config.json'),
        original_recipe=binding(project / 'train-recipe.json'), training_population=binding(training / 'splits.json'),
        training_inputs=bound_inputs, original_output_locations=locations, parameters=expected_parameters,
        actual_training_parents=len(parents), actual_training_frames=sum(len(p['frames']) for p in parents),
        actual_parent_densities=dict(Counter(str(1/p['cell_size_m']) for p in parents)),
        actual_parent_shapes=dict(Counter('x'.join(map(str,p['counts_xyz'])) for p in parents)),
        tier_exposures=dict(tiers), sampled_cells=cells, optimizer_updates=2400,
        sample_order_independently_rederived=True, simulation_arrays_decoded=False,
        new_model_forwards=0, clean_reproduction_qualified=False)
    return dict(metadata=metadata, project=project, training=training, receipt=receipt, history=history,
        config=config, fine=fine, split=split, inputs=inputs, parents=parents,
        initial=initial_values, final=final_values, order=order, small_order=small_order)
