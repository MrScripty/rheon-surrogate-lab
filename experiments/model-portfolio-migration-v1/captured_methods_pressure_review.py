"""Receipt closure and explicit sampler owners for the methods-v1/v2 reviews.

Normalizes metadata in memory only; captured code is never imported. Uniform
physical indices in capacity probes are retained bookkeeping, not optimizer use.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np

from captured_replay_pressure_review import check_tensors
from review_water_learning_sequence import binding, read, require

ROOT = Path(__file__).resolve().parents[2]


def sample_order(parents,config,policy):
    count = config['training']['updates']
    seed = config['training']['sampling_seed']
    require(config['training']['batch_size'] == 1 and all(p['entry']['split'] == 'train' for p in parents),
            'Methods sampler requires serial train-only parents')
    rng = np.random.Generator(np.random.PCG64(seed))
    def draw(indices,generator):
        i = indices[int(generator.integers(len(indices)))]
        return i,int(generator.integers(len(parents[i]['frames'])))
    if policy == 'uniform':
        return np.asarray([[draw(list(range(len(parents))),rng)] for _ in range(count)],dtype='<i8'),None
    small = [i for i,p in enumerate(parents) if max(p['counts_xyz']) <= 16]
    tiers = {}
    for i,p in enumerate(parents):
        tier = 16 if max(p['counts_xyz']) <= 16 else 32 if max(p['counts_xyz']) <= 32 else 64
        tiers.setdefault(tier,[]).append(i)
    if policy == 'equal_tiers':
        names = sorted(tiers)
        tier_order = np.concatenate([rng.permutation(names) for _ in range(math.ceil(count/len(names)))])[:count]
        return np.asarray([[draw(tiers[int(t)],rng)] for t in tier_order],dtype='<i8'),None
    require(policy == 'small_replay' and small,'Explicit supported methods sampling policy required')
    if len(small) == len(parents):
        subsequence = [draw(small,rng) for _ in range(count)]
        return np.asarray([[v] for v in subsequence],dtype='<i8'),np.asarray(subsequence,dtype='<i8')
    require(count in (1600,2400) and 32 in tiers and 64 in tiers,'Declared replay expansion budget/tiers required')
    retained = count == 2400
    subsequence = [draw(small,rng) for _ in range(1600 if retained else 800)]
    other = np.random.Generator(np.random.PCG64(seed+1))
    order = []
    for n,item in enumerate(subsequence):
        order.append([item])
        if not retained or n%2:
            k = n//2 if retained else n
            order.append([draw(tiers[32 if k%2 == 0 else 64],other)])
    return np.asarray(order,dtype='<i8'),np.asarray(subsequence,dtype='<i8')


def load_fit(fid,project,*,policy,work,index):
    fit = next(f for f in work['fit_cards'] if f['item_id'] == fid)
    card = next(f for f in index['fits'] if f['item_id'] == fid)
    receipts = [ROOT/p for p in fit['receipt_paths']]
    require(all(binding(p)['sha256'] == fit['receipt_sha256'] == card['receipt_sha256'] for p in receipts)
            and not card.get('method_review_status'),'Original unreviewed methods fit/receipt differs')
    receipt_path = receipts[0]
    training = receipt_path.parent
    receipt,history = read(receipt_path),read(training/'history.json')
    config,witness = read(project/'config.json'),read(project/'source-witness.json')
    manifest,recipe = read(project/'project.json'),read(project/'train-recipe.json')
    fine,split = read(project/'fine-tuning.json'),read(training/'splits.json')
    require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
            and receipt['optimizer_updates'] == config['training']['updates'],'Completed methods fitting budget differs')
    for name,key in [('config.json','experiment_sha256'),('source-witness.json','source_witness_sha256'),
                     ('runtime-manifest.json','runtime_manifest_sha256')]:
        require(binding(project/name)['sha256'] == receipt['closure'][key],'Original methods closure differs')
    sources = [binding(project/n) for n in sorted(witness['sources'])]
    require(all(b['sha256'] == witness['sources'][Path(b['path']).name] for b in sources),'Complete witnessed methods source differs')
    require(recipe['entrypoint'] == str((project/'evidence.py').relative_to(ROOT))
            and recipe['arguments'][0] == dict(literal='train')
            and recipe['runtime'] == 'managed_torch_python_v1','Actual methods evidence recipe differs')
    for spec in [*manifest['recipes'],*recipe['sources']]:
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'],'Typed methods recipe/source differs')
    inputs = {r['id']:ROOT/r['path'] for r in manifest['data']}
    bound_inputs = []
    for slot,expected in receipt['closure']['inputs'].items():
        p = inputs[slot]
        b = binding(p)
        require(dict(sha256=b['sha256'],bytes=p.stat().st_size) == expected,'Original methods input differs')
        bound_inputs.append(dict(b,slot=slot,bytes=expected['bytes']))
    locations = []
    for name,expected in receipt['outputs'].items():
        p = training/('training-profile.json' if name == 'profile.json' else name)
        require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size)
                == {k:expected[k] for k in ('sha256','bytes')},'Original methods output/profile differs')
        locations.append(dict(original_name=name,retained_path=str(p.relative_to(ROOT)),sha256=expected['sha256']))
    require(binding(training/'normalization.json')['sha256'] == receipt['closure']['normalization_sha256']
            and history['diagnostic_population'] == receipt['diagnostic_population'] == 'train_only'
            and receipt['heldout_evaluations'] == history['heldout_evaluations'] == 0
            and not receipt['scientific_acceptance_or_model_promotion'],'Methods scales/diagnostic scope differs')
    parents = [dict(p,entry=p.get('entry',dict(trajectory_id=p.get('trajectory_id'),split=p.get('split'),dt_s=p.get('dt_s'))))
               for p in split['parents']]
    order,subsequence = sample_order(parents,config,policy)
    require(order.shape == (config['training']['updates'],1,2)
            and hashlib.sha256(order.tobytes()).hexdigest() == receipt['closure']['sample_order_sha256'],
            'Independent methods physical index order differs')
    exposures = Counter(int(i) for ((i,j),) in order)
    cells = sum(math.prod(parents[int(i)]['counts_xyz']) for ((i,j),) in order)
    tier_counts = Counter(str(16 if max(parents[int(i)]['counts_xyz']) <= 16 else 32
        if max(parents[int(i)]['counts_xyz']) <= 32 else 64) for ((i,j),) in order)
    if policy != 'uniform':
        original = ROOT/('runs/pressure-methods-v1/teacher-coarse-project-01/mixed_order.py' if policy == 'equal_tiers'
                         else 'runs/pressure-methods-v2/v2-small16-project-02/mixed_order.py')
        require((project/'mixed_order.py').read_bytes() == original.read_bytes(),'Selected sampler implementation differs')
        sampling = history['current_feedback']['sampling']
        require(history['current_feedback']['status'] == 'inherited_auxiliary_banks_disabled'
                and sampling['parent_exposures'] == [exposures[i] for i in range(len(parents))]
                and sampling['tier_exposures'] == dict(tier_counts),'Methods recorded exposure accounting differs')
        if subsequence is not None:
            require(sampling['sampled_cells'] == cells and hashlib.sha256(subsequence.tobytes()).hexdigest()
                    == sampling['small_subsequence_sha256'],'Methods replay subsequence/cell accounting differs')
        require(history['analytical_feedback'] is None,'Inherited manufactured banks must be disabled')
    inventory = history['checkpoints']['checkpoint-000.npz']['inventory']
    require(inventory == fine['initial_inventory'] and binding(inputs['initial_weights'])['sha256'] == fine['initial_weights_sha256'],
            'Methods declared initialization differs')
    initial = check_tensors(training/'checkpoint-000.npz',inventory)
    warm = check_tensors(inputs['initial_weights'],inventory)
    require(all(initial[n].tobytes() == warm[n].tobytes() for n in initial),'Actual methods warm values differ')
    final = check_tensors(training/'weights.npz',history['checkpoints'][f'checkpoint-{config["training"]["updates"]:03d}.npz']['inventory'])
    metadata = dict(fit_item_id=fid,note_path=card['card_path'],named_checkpoint=fit['name'],original_receipt=binding(receipt_path),
        history=binding(training/'history.json'),sources=sources,source_witness=binding(project/'source-witness.json'),
        original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
        original_fine_tuning_contract=binding(project/'fine-tuning.json'),training_inputs=bound_inputs,
        training_population=binding(training/'splits.json'),original_output_locations=locations,
        actual_parameter_count=sum(v.size for v in initial.values()),initialized_tensor_count=len(initial),
        actual_train_trajectories=len(parents),actual_train_frames=sum(len(p['frames']) for p in parents),
        actual_validation_trajectories=0,actual_validation_frames=0,sampler_policy=policy,
        sampled_cells=cells,tier_exposures=dict(tier_counts),sample_order_independently_rederived=True,
        actual_cells_per_metre=dict(Counter(str(1/p['cell_size_m']) for p in parents)),
        actual_grid_shapes=dict(Counter('x'.join(map(str,p['counts_xyz'])) for p in parents)),
        clean_reproduction_qualified=False,new_model_forwards=0,simulation_arrays_decoded=False)
    return dict(metadata=metadata,project=project,training=training,receipt=receipt,history=history,config=config,
        fine=fine,split=split,parents=parents,initial=initial,final=final,order=order,subsequence=subsequence,inputs=inputs)
