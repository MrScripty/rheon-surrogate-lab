"""Original native pressure-fit closure, including the SDK evidence wrapper.

Native recipes use evidence.py and retain the trainer's profile under
training-profile.json. These semantics differ from legacy stage-00 receipts.
No captured code is imported and no native project or scientific file is written.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from review_water_learning_sequence import binding, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'


def load_fit(identifier, *, work, material, index, score):
    entry = material[identifier]
    project = ROOT/entry['project']
    aliases = [a for a in work['named_entries'] if a['label'] == identifier and a['item_id'].startswith('cohorts/')]
    require(len(aliases) == 1, 'One original native named pressure alias required')
    fit_id = aliases[0]['fit_item_id']
    fit = next(f for f in work['fit_cards'] if f['item_id'] == fit_id)
    candidates = [ROOT/p for p in fit['receipt_paths'] if (ROOT/p).is_file()
                  and binding(ROOT/p)['sha256'] == fit['receipt_sha256']]
    require(len(candidates) == 1, 'One original exact native training receipt required')
    receipt_path = candidates[0]
    receipt = read(receipt_path)
    training = receipt_path.parent
    config, witness = read(project/'config.json'), read(project/'source-witness.json')
    history, split = read(training/'history.json'), read(training/'splits.json')
    normalization = read(training/'normalization.json')
    require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
            and receipt['optimizer_updates'] == config['training']['updates'], 'Native fit did not finish declared updates')
    for name, key in [('config.json','experiment_sha256'),('source-witness.json','source_witness_sha256'),
                      ('runtime-manifest.json','runtime_manifest_sha256')]:
        require(binding(project/name)['sha256'] == receipt['closure'][key], 'Native original source/config/runtime differs')
    sources = [binding(project/name) for name in sorted(witness['sources'])]
    require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources), 'Native witnessed source differs')
    manifest, recipe = read(project/'project.json'), read(project/'train-recipe.json')
    require(recipe['entrypoint'] == str((project/'evidence.py').relative_to(ROOT))
            and recipe['arguments'][0] == dict(literal='train')
            and recipe['runtime'] == 'managed_torch_python_v1', 'Native typed evidence recipe differs')
    for spec in [*manifest['recipes'],*recipe['sources']]:
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Native typed source binding differs')
    inputs = {v['id']:v for v in manifest['data']}
    bound_inputs = []
    for slot, expected in receipt['closure']['inputs'].items():
        path = ROOT/inputs[slot]['path']
        observed = binding(path)
        require(dict(sha256=observed['sha256'],bytes=path.stat().st_size) == expected, 'Native original input differs')
        bound_inputs.append(dict(observed,slot=slot,bytes=path.stat().st_size))
    evidence = (project/'evidence.py').read_text()
    require('(output / "profile.json").rename(output / "training-profile.json")' in evidence
            and 'train.evaluate = training_diagnostic.diagnostic' in evidence,
            'Native profile custody or explicit diagnostic owner differs')
    output_locations = []
    for name, expected in receipt['outputs'].items():
        selected = 'training-profile.json' if name == 'profile.json' else name
        path = training/selected
        require(dict(sha256=binding(path)['sha256'],bytes=path.stat().st_size)
                == {k:expected[k] for k in ('sha256','bytes')}, 'Native original output/profile differs')
        output_locations.append(dict(original_output=name,retained_path=str(path.relative_to(ROOT)),sha256=expected['sha256']))
    require(receipt['outputs']['weights.npz']['sha256'] == score[identifier]['weights_sha256']
            == entry['weights_sha256'] == aliases[0]['declared_weights_sha256'], 'Native named final checkpoint differs')
    require(binding(training/'normalization.json')['sha256'] == receipt['closure']['normalization_sha256'],
            'Native fitted normalization differs')
    require(history['diagnostic_population'] == receipt['diagnostic_population'] == 'train_only'
            and history['heldout_evaluations'] == receipt['heldout_evaluations'] == 0
            and not history['scientific_acceptance_or_model_promotion']
            and not receipt['scientific_acceptance_or_model_promotion'], 'Native training diagnostics scope differs')
    parents = split['parents']
    eligible = [i for i,p in enumerate(parents) if p['entry']['split'] == 'train']
    rng = np.random.Generator(np.random.PCG64(config['training']['sampling_seed']))
    batches = []
    for _ in range(config['training']['updates']):
        batch = []
        for _ in range(config['training']['batch_size']):
            parent = eligible[int(rng.integers(len(eligible)))]
            batch.append((parent,int(rng.integers(len(parents[parent]['frames'])))))
        batches.append(batch)
    order = np.asarray(batches,dtype='<i8')
    require(hashlib.sha256(order.tobytes()).hexdigest() == receipt['closure']['sample_order_sha256'],
            'Native optimizer sample order differs')
    inventory = history['checkpoints']['checkpoint-000.npz']['inventory']
    card = next(f for f in index['fits'] if f['item_id'] == fit_id)
    require(card['receipt_sha256'] == fit['receipt_sha256'] and not card.get('method_review_status'),
            'Native fit already reviewed or card receipt differs')
    metadata = dict(fit_item_id=fit_id,note_path=card['card_path'],named_checkpoint=identifier,
        original_receipt=binding(receipt_path),history=binding(training/'history.json'),sources=sources,
        source_witness=binding(project/'source-witness.json'),training_population=binding(training/'splits.json'),
        training_inputs=bound_inputs,original_configuration=binding(project/'config.json'),
        original_recipe=binding(project/'train-recipe.json'),retained_accuracy=score[identifier]['result'],
        common_score_source=binding(DOC/'reports/pressure-common-score-01.json'),
        actual_parameter_count=sum(math.prod(v['shape']) for v in inventory.values()),initialized_tensor_count=len(inventory),
        actual_train_trajectories=len(eligible),actual_train_frames=sum(len(parents[i]['frames']) for i in eligible),
        actual_validation_trajectories=sum(p['entry']['split'] == 'validation' for p in parents),
        actual_validation_frames=sum(len(p['frames']) for p in parents if p['entry']['split'] == 'validation'),
        sample_order_independently_rederived=True,clean_reproduction_qualified=False,new_model_forwards=0,
        training_diagnostic_population='train_only',original_output_locations=output_locations)
    return dict(metadata=metadata,project=project,training=training,receipt=receipt,config=config,history=history,
                split=split,normalization=normalization,inventory=inventory,sample_order=order)
