"""Read and validate original pressure-fit facts without importing a model.

This helper owns receipt/recipe/input/output consistency for retrospective
reviews. Callers still explain and check their family's actual transformations.
It neither writes the native project nor confers clean-reproduction authority.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from review_water_learning_sequence import binding, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'


def load_fit(identifier, *, work, material, index, score):
    entry = material[identifier]
    project = ROOT/entry['project']
    aliases = [a for a in work['named_entries'] if a['label'] == identifier and a['item_id'].startswith('cohorts/')]
    require(len(aliases) == 1, 'Exactly one original named pressure alias required')
    fit_id = aliases[0]['fit_item_id']
    fit = next(f for f in work['fit_cards'] if f['item_id'] == fit_id)
    candidates = [ROOT/name for name in fit['receipt_paths'] if name.startswith(str(project.parent.relative_to(ROOT))+'/')
                  and name.endswith('/stage-00/artifacts/receipt.json') and (ROOT/name).exists()]
    require(candidates, 'Original canonical training receipt unavailable')
    receipt_path = candidates[0]
    receipt = read(receipt_path)
    training = receipt_path.parent
    require(binding(receipt_path)['sha256'] == fit['receipt_sha256'] and receipt['mode'] == 'train'
            and receipt['status'] == 'complete', 'Original fit receipt is not the indexed completed training attempt')
    config, witness = read(project/'config.json'), read(project/'source-witness.json')
    history, split = read(training/'history.json'), read(training/'splits.json')
    normalization = read(training/'normalization.json')
    require(receipt['optimizer_updates'] == config['training']['updates'], 'Configured and actual optimizer budget differ')
    for name,key in [('config.json','experiment_sha256'),('source-witness.json','source_witness_sha256'),
                     ('runtime-manifest.json','runtime_manifest_sha256')]:
        require(binding(project/name)['sha256'] == receipt['closure'][key], 'Original source/config/runtime closure differs')
    sources = [binding(project/name) for name in sorted(witness['sources'])]
    require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources), 'Original source changed')
    manifest, recipe = read(project/'project.json'), read(project/'train-recipe.json')
    require(recipe['entrypoint'] == str((project/'train.py').relative_to(ROOT))
            and recipe['arguments'][0] == dict(literal='train'), 'Original train recipe entrypoint differs')
    for spec in [*manifest['recipes'],*recipe['sources']]:
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Original typed recipe source differs')
    inputs = {e['id']:e for e in manifest['data']}
    bound_inputs = []
    for slot, expected in receipt['closure']['inputs'].items():
        path = ROOT/inputs[slot]['path']
        observed = binding(path)
        require(dict(sha256=observed['sha256'],bytes=path.stat().st_size) == expected, 'Original typed input differs')
        bound_inputs.append(dict(**observed,slot=slot,bytes=path.stat().st_size))
    for name, expected in receipt['outputs'].items():
        path = training/name
        require(dict(sha256=binding(path)['sha256'],bytes=path.stat().st_size)
                == {k:expected[k] for k in ('sha256','bytes')}, 'Original retained output differs')
    require(receipt['outputs']['weights.npz']['sha256'] == score[identifier]['weights_sha256']
            == entry['weights_sha256'] == aliases[0]['declared_weights_sha256'], 'Named checkpoint differs')
    require(binding(training/'normalization.json')['sha256'] == receipt['closure']['normalization_sha256'],
            'Original normalization differs')
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
            'Independently derived original optimizer sample order differs')
    inventory = history['checkpoints']['checkpoint-000.npz']['inventory']
    count = sum(math.prod(v['shape']) for v in inventory.values())
    card = next(f for f in index['fits'] if f['item_id'] == fit_id)
    require(card['receipt_sha256'] == fit['receipt_sha256'] and not card.get('method_review_status'),
            'Card receipt differs or fit is already source-reviewed')
    metadata = dict(fit_item_id=fit_id,note_path=card['card_path'],named_checkpoint=identifier,
        original_receipt=binding(receipt_path),history=binding(training/'history.json'),sources=sources,
        source_witness=binding(project/'source-witness.json'),training_population=binding(training/'splits.json'),
        training_inputs=bound_inputs,original_configuration=binding(project/'config.json'),
        original_recipe=binding(project/'train-recipe.json'),retained_accuracy=score[identifier]['result'],
        common_score_source=binding(DOC/'reports/pressure-common-score-01.json'),
        actual_parameter_count=count,initialized_tensor_count=len(inventory),
        actual_train_trajectories=len(eligible),actual_train_frames=sum(len(parents[i]['frames']) for i in eligible),
        actual_validation_trajectories=sum(p['entry']['split'] == 'validation' for p in parents),
        actual_validation_frames=sum(len(p['frames']) for p in parents if p['entry']['split'] == 'validation'),
        sample_order_independently_rederived=True,clean_reproduction_qualified=False,new_model_forwards=0)
    return dict(metadata=metadata,project=project,training=training,receipt=receipt,config=config,history=history,
                split=split,normalization=normalization,inventory=inventory,sample_order=order)
