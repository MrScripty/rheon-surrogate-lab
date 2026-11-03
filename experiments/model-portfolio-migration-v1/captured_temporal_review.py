"""Verify immutable temporal fit custody without importing captured Python.

This owns byte identities, complete tensor inventories, fixed training-corpus
membership and private sample order. Family reviewers own architecture meaning,
parameter formulas, actual optimizer work and method/outcome/reproduction prose.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

from review_water_learning_sequence import binding, read, require
from prepare_method_review_revisions import verified_review_receipt as verify_legacy_receipt

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'


def audit(item, trainer_name):
    project = ROOT/item['project_path']
    receipt_path = ROOT/item['receipt']['path']
    require(binding(receipt_path)['sha256'] == item['receipt']['sha256'], 'Original temporal receipt changed')
    config, receipt = read(project/'config.json'), read(receipt_path)
    train = receipt_path.parent
    history = read(train/'history.json')
    require(receipt['status'] == 'complete' and receipt['mode'] == 'train'
            and receipt['optimizer_updates'] == config['updates']
            and receipt['horizons'] == config['horizons'], 'Original completed fit required')
    require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Original config changed')
    sources = [binding(project/n) for n in sorted(receipt['closure']['sources'])]
    require(all(s['sha256'] == receipt['closure']['sources'][str((ROOT/s['path']).relative_to(project))]
                for s in sources), 'Executed source closure changed')
    document, recipe = read(project/'project.json'), read(project/'train-recipe.json')
    require(recipe['entrypoint'] == str((project/trainer_name).relative_to(ROOT)), 'Actual trainer selection differs')
    for spec in [*document['recipes'],*recipe['sources']]:
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Typed recipe/source changed')
    prior_path = DOC/'reports/captured-joint-temporal-generator-method-reviews-01.json'
    prior = read(prior_path)
    require(prior['training_archive_audit']['all_decoded_float32_field_hashes_verified'], 'Full original-corpus audit required')
    expected_inputs = {r['slot']:{k:r[k] for k in ('sha256','bytes')} for r in prior['rows'][0]['training_inputs']}
    inputs = {r['id']:ROOT/r['path'] for r in document['data']}
    require(receipt['closure']['inputs'] == expected_inputs, 'Original16scene fitting corpus/runtime differs')
    bound_inputs = []
    for slot,expected in expected_inputs.items():
        spec = binding(inputs[slot])
        require(dict(sha256=spec['sha256'],bytes=inputs[slot].stat().st_size) == expected, 'Original training input changed')
        bound_inputs.append(dict(spec,slot=slot,bytes=expected['bytes']))
    manifest, norm = read(inputs['manifest']), read(inputs['normalization'])
    require(len(manifest['parents']) == history['valid_parent_count'] == 16 and history['train_only']
            and not manifest['development_or_reserved_used'] and not norm['development_or_reserved_used'],
            'Original consumed train-only population differs')
    stats_path = project/'step-scales.json'
    stats = read(stats_path)
    scale_review_path = DOC/'reports/captured-step-scaled-temporal-generator-method-reviews-01.json'
    scale_review = read(scale_review_path)
    require(binding(stats_path)['sha256'] == config['step_scale_statistics_sha256'] == scale_review['rows'][0]['fixed_statistics']['sha256']
            and stats['original_normalization_sha256'] == config['original_normalization_sha256'] == binding(inputs['normalization'])['sha256']
            and scale_review['independent_statistics_verification']['all_parent_increment_scales_verified']
            and config['physical_scales'] == norm['scales'], 'Independently audited scales/physical units differ')
    for name,expected in receipt['outputs'].items():
        p = train/name
        require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size) == expected, 'Original fit output changed')
    for filename,key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
        inventory = receipt[key]
        require(inventory == history[key], 'Receipt/history tensor inventory differs')
        with np.load(train/filename,allow_pickle=False) as arrays:
            require(set(arrays.files) == set(inventory), 'Complete retained tensor inventory required')
            for name,spec in inventory.items():
                value = arrays[name]
                require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                        and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'], 'Original tensor bytes/type/shape changed')
    count = sum(math.prod(s['shape']) for s in receipt['final_inventory'].values())
    require(count == receipt['parameters'], 'Retained parameter count differs')
    frozen = receipt['frozen_parameter_inventory']
    require(all(receipt['initial_inventory'].get(n) == receipt['final_inventory'].get(n) == s for n,s in frozen.items()),
            'Frozen expert tensors changed during fitting')
    frozen_count = sum(math.prod(s['shape']) for s in frozen.values())
    require(receipt['trainable_parameters'] == count-frozen_count, 'Trainable/frozen count differs')
    assets = project/'expert-assets'
    asset_specs = [binding(p) for p in sorted(assets.rglob('*')) if p.is_file()]
    require({str((ROOT/s['path']).relative_to(assets)):s['sha256'] for s in asset_specs} == config['expert_asset_sha256'],
            'Complete expert asset inventory changed')
    rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
    order = np.asarray([(int(rng.integers(16)),int(rng.integers(121)),1 if u <= config['single_block_updates'] else 2)
                        for u in range(1,config['updates']+1)],dtype='<i8')
    require(hashlib.sha256(order.tobytes()).hexdigest() == history['sample_order_sha256'], 'Private sample/curriculum order differs')
    require(history['curriculum'] == dict(single_block_updates=config['single_block_updates'],later_blocks=2,
        second_block_uses_own_predicted_history=True,feedback_encoding='physical_reconstruction_then_deployed_mean_centering_v2'),
        'Deployment-equivalent curriculum differs')
    facts = dict(original_receipt=binding(receipt_path),history=binding(train/'history.json'),sources=sources,
        source_closure_semantics='project_relative_paths_including_expert_assets',
        original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
        training_inputs=bound_inputs,training_manifest=binding(inputs['manifest']),fixed_statistics=binding(stats_path),
        expert_assets=asset_specs,retained_parameter_count=count,trainable_parameters=count-frozen_count,
        frozen_parameters=frozen_count,optimizer_updates=config['updates'],actual_train_trajectories=16,
        available_train_states=2064,physical_output_dt_s=1/60,block_physical_advance_s=config['horizons']/60,
        private_sample_order_verified=True,frozen_tensors_unchanged=True,
        corpus_audit=binding(prior_path),statistics_audit=binding(scale_review_path))
    return dict(project=project,train=train,receipt=receipt,config=config,history=history,manifest=manifest,
                norm=norm,order=order,facts=facts)


def verified_review_receipt(review):
    """Validate complete project-relative closures, then shared receipt fields.

    Historical generic reviews use basename-only source keys. These temporal
    receipts also own nested expert assets, so basename lookup is insufficient.
    This explicit variant preserves every original key/path; it neither flattens
    assets nor changes the already executed legacy validation implementation.
    """
    require(review['source_closure_semantics'] == 'project_relative_paths_including_expert_assets',
            'Explicit temporal closure contract required')
    receipt_path = ROOT/review['original_receipt']['path']
    require(binding(receipt_path) == review['original_receipt'], 'Original temporal receipt changed')
    receipt = read(receipt_path)
    configuration = ROOT/review['original_configuration']['path']
    require(binding(configuration) == review['original_configuration']
            and binding(configuration)['sha256'] == receipt['closure']['config_sha256'], 'Source-owning configuration changed')
    owner = configuration.parent
    observed = {}
    for spec in review['sources']:
        path = ROOT/spec['path']
        require(path.is_relative_to(owner) and binding(path) == spec, 'Temporal source path/bytes differ')
        key = str(path.relative_to(owner))
        require(key not in observed, 'Duplicate temporal source closure key')
        observed[key] = spec['sha256']
    require(observed == receipt['closure']['sources'], 'Complete project-relative source closure required')
    # Source validation is owned above. Reuse unchanged history/input/receipt
    # checks without asking the old basename-only validator to reinterpret it.
    require('source_witness' not in review, 'This explicit temporal variant has no legacy source witness')
    return verify_legacy_receipt(dict(review,sources=[]))
