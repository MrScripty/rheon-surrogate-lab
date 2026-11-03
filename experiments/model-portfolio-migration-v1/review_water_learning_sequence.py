"""Review C06--C10 training diversity and budgets from captured facts only.

NumPy rederives PCG64 sample order; no captured Python model is imported and no
neural forward, simulation, fit, project write or new physical reference occurs.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from prepare_method_review_revisions import revised_card, verified_review_receipt

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC / 'reports/captured-water-learning-sequence-method-reviews-01.json'
SPECS = [('C06', 256, 512, 2000, 2), ('C07', 1, 1, 10000, 1),
         ('C08', 256, 512, 10000, 2), ('C09', 256, 256, 10000, 2),
         ('C10', 1024, 1024, 10000, 2)]


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path.relative_to(ROOT)), sha256=digest)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def population_identity(parents):
    # Selection namespaces changed when the original packet was resharded.
    # Compare physical scene/fields/frames, without equating catalog ancestry.
    return [dict(scene=p['scene'], counts_xyz=p['counts_xyz'], cell_size_m=p['cell_size_m'],
                 frames=p['frames'], dt_s=p['entry']['dt_s']) for p in parents]


def run():
    require(not OUTPUT.exists(), 'Preserve the original authored source review')
    index_path = PROJECT / 'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    index = json.loads(index_raw)
    material = {r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score_path = DOC / 'reports/pressure-common-score-01.json'
    scores = read(score_path)['models']
    rows, populations, orders, initial, note_bindings = [], {}, {}, {}, []
    for label, expected_trajectories, expected_frames, updates, batch in SPECS:
        entry = material[label]
        project = ROOT / entry['project']
        training = (ROOT / entry['restore']).parent.parent / 'stage-00/artifacts'
        receipt_path = training / 'receipt.json'
        receipt = read(receipt_path)
        history = read(training/'history.json')
        config = read(project/'config.json')
        witness = read(project/'source-witness.json')
        config_training = config['training']
        require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
                and receipt['optimizer_updates'] == config_training['updates'] == updates
                and config_training['batch_size'] == batch and config_training['learning_rate'] == .001
                and config_training['seed'] == 20261015 and config_training['sampling_seed'] == 2026102501,
                'Captured optimizer budget/seed differs')
        for name, key in [('config.json','experiment_sha256'), ('source-witness.json','source_witness_sha256'),
                          ('runtime-manifest.json','runtime_manifest_sha256')]:
            require(binding(project/name)['sha256'] == receipt['closure'][key], 'Captured training closure differs')
        sources = [binding(project/name) for name in sorted(witness['sources'])]
        require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources), 'Captured source changed')
        require(binding(project/'portable_model.py')['sha256'] == '64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935',
                'Shared seven-channel pressure implementation differs')
        manifest = read(project/'project.json')
        recipe = read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/'train.py').relative_to(ROOT))
                and recipe['arguments'][0] == dict(literal='train'), 'Captured recipe entrypoint differs')
        for source in [*manifest['recipes'], *recipe['sources']]:
            require(binding(ROOT/source['path'])['sha256'] == source['sha256'], 'Typed recipe source differs')
        inputs = {r['id']:r for r in manifest['data']}
        bound_inputs = []
        for slot, identity in receipt['closure']['inputs'].items():
            path = ROOT/inputs[slot]['path']
            bound = binding(path)
            require(dict(sha256=bound['sha256'],bytes=path.stat().st_size) == identity, 'Original typed input differs')
            bound_inputs.append(dict(**bound,slot=slot,bytes=path.stat().st_size))
        for name in ['weights.npz','normalization.json','history.json','splits.json',
                     *[f'checkpoint-{update:03d}.npz' for update in config_training['checkpoints']]]:
            path = training/name
            require(dict(sha256=binding(path)['sha256'],bytes=path.stat().st_size)
                    == {k:receipt['outputs'][name][k] for k in ('sha256','bytes')}, 'Original retained training output differs')
        require(receipt['outputs']['weights.npz']['sha256'] == scores[label]['weights_sha256'], 'Named selected checkpoint differs')
        split = read(training/'splits.json')
        parents = split['parents']
        trains = [p for p in parents if p['entry']['split'] == 'train']
        validation = [p for p in parents if p['entry']['split'] == 'validation']
        require(len(trains) == expected_trajectories and sum(len(p['frames']) for p in trains) == expected_frames
                and len(validation) == 64 and sum(len(p['frames']) for p in validation) == 128
                and len(trains)+len(validation) == len(parents), 'Actual physical optimizer/validation population differs')
        require(all(p['entry']['dt_s'] == 1/60 and p['density_kg_m3'] == 1000
                and 1/p['cell_size_m'] in (8,16) and all(5 <= n <= 16 for n in p['counts_xyz'])
                for p in parents), 'Physical units/density/domain population differs')
        normalization = read(training/'normalization.json')
        require(binding(training/'normalization.json')['sha256'] == receipt['closure']['normalization_sha256']
                and normalization['training_provenance']['frames'] == expected_frames
                and normalization['training_provenance']['trajectory_ids'] == [p['entry']['trajectory_id'] for p in trains],
                'Train-only normalization provenance differs')
        inventory = history['checkpoints']['checkpoint-000.npz']['inventory']
        require(len(inventory) == 38 and sum(math.prod(v['shape']) for v in inventory.values()) == 371729
                and receipt['architecture']['input_channels'] == 7
                and receipt['architecture']['block_normalization']['name'] == 'GroupNorm', 'Actual parameter/architecture inventory differs')
        initial[label] = inventory
        eligible = [i for i,p in enumerate(parents) if p['entry']['split'] == 'train']
        rng = np.random.Generator(np.random.PCG64(config_training['sampling_seed']))
        order = []
        for _ in range(updates):
            selected = []
            for _ in range(batch):
                parent = eligible[int(rng.integers(len(eligible)))]
                selected.append((parent,int(rng.integers(len(parents[parent]['frames'])))))
            order.append(selected)
        order = np.asarray(order,dtype='<i8')
        require(hashlib.sha256(order.tobytes()).hexdigest() == receipt['closure']['sample_order_sha256'],
                'Independently derived optimizer sampling order differs')
        orders[label], populations[label] = order, population_identity(parents)
        train_source = (project/'train.py').read_text()
        require('torch.optim.Adam(network.parameters()' in train_source and 'clip_grad_norm_' in train_source
                and 'final update candidate; no optimizer resume' in str(history['scope']), 'Optimizer/update selection contract differs')
        schedule = config_training['schedule']
        require(schedule == ([dict(learning_rate=.0001,update=1000)] if label == 'C06' else
                [dict(learning_rate=.0001,update=1000),dict(learning_rate=.00001,update=6000)]), 'Original learning-rate schedule differs')
        receipt_binding = binding(receipt_path)
        card = next(f for f in index['fits'] if f['receipt_sha256'] == receipt_binding['sha256'])
        require(not card.get('method_review_status'), 'Fit is already source-reviewed')
        note = PROJECT/card['card_path']
        raw = note.read_bytes()
        note_bindings.append(binding(note))
        purposes = {
            'C06':'Establish a CUDA-trained pressure baseline using 256 simulated training scenes and two saved frames per scene.',
            'C07':'Test whether the unchanged network can memorize one pressure frame, separating basic fitting capacity from broad generalization.',
            'C08':'Increase the optimization budget on the original 256-scene, two-frame physical training population.',
            'C09':'Establish a 256-scene, one-frame-per-scene data-scaling control with the long optimization schedule.',
            'C10':'Expand the one-frame-per-scene control to 1,024 training scenes across four independently seeded capture cohorts.'}
        worst = scores[label]['worst']
        last = history['history'][-1]
        train_densities = dict(Counter(str(1/p['cell_size_m']) for p in trains))
        frames = sorted({frame['index'] for p in trains for frame in p['frames']})
        if label == 'C07':
            lesson = ('A final sampled training loss near 1.6e-6 on one frame coexists with materially larger held-out loss and common pressure error. '
                      'This is a narrow fitting-capacity control; its training population has only 8 cells/metre on one 16x16x16 scene. '
                      'The two-density validation population must not be described as two-density training.')
        elif label == 'C08':
            lesson = ('The physical frames, initial tensors and first 2,000 sampled update batches match C06 despite packet resharding/namespaces. '
                      'Additional updates reduce the captured validation objective, but the final candidates still have substantial common worst error. '
                      'This is evidence about fitting budget on small grids, not generalization to larger domains or new timesteps.')
        elif label == 'C09':
            lesson = ('One saved frame per scene reduces the training frame count from 512 to 256 while retaining 256 physical scenes. '
                      'Selected frames include both native indices 15 and 63 across scenes; one frame per scene does not mean one common timestamp. '
                      'The train-derived scales change with this selection, so compare the same common physical errors rather than logged normalized losses alone.')
        elif label == 'C10':
            lesson = ('This increases independent scene count rather than merely repeating the original 256 scenes at more cadences. '
                      'Its broader fitting population and train-only scales still cover only 8/16 cells per metre, 5–16 cells per axis and one 1/60 s step size. '
                      'More small-grid scenes do not establish large-grid/high-density or variable-timestep accuracy; compare common errors to the 256-scene control.')
        else:
            lesson = ('This is already a 256-scene physical population, distinct from the much smaller early variable-grid dataset. '
                      'Two snapshots per trajectory remain sparse temporal supervision. Common worst pressure/equation errors are substantial despite the wider scene count; '
                      'do not infer larger-domain or timestep generalization from parameter sharing alone.')
        row = dict(fit_item_id=card['item_id'],note_path=card['card_path'],named_checkpoint=label,
            method='A 371,729-parameter, 38-tensor F32 multiscale 3D pressure-correction CNN with seven input channels: residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep. '
                   'Encoder widths 8/16/24/32, bottleneck 48, decoder 32/24/16/8, affine-free four-group GroupNorm, SiLU, global-mean broadcast and nearest-to-skip resizing. '
                   'The linear head supplies a pressure-scaled cell-zero-gauge correction added to the previous pressure. FV arithmetic/output remain F64. '
                   'This is a pressure request, not a complete simulation advance. The captured cell budget is 65,536.',
            purpose=purposes[label],
            experiment=f'{updates:,} whole-network Adam updates, batch {batch}, initial learning rate 0.001, seed 20261015 and sampling seed 2026102501. '
                f'Learning-rate schedule: {schedule}. Gradient norm is clipped at 1.0. '
                f'{expected_trajectories:,} training trajectories/{expected_frames:,} saved pressure frames; 64 validation trajectories/128 frames. '
                f'Actual training densities, in scenes: {train_densities}. Native training snapshot indices present: {frames}. '
                'The sealed-water captures use variable rectangular small domains, randomized Gaussian/tapered force position, direction, amplitude, motion, swirl and temporal variation. '
                'The objective combines normalized pressure-field and FV equation losses, with scales fitted only on training frames. '
                'Reserved-test input bindings are retained separately and do not enter the optimizer parent selection.',
            outcome=f'The final sampled training loss is {last["loss"]:.17g}; the retained validation objective changes from {history["validation"]["initial"]:.17g} to {history["validation"]["final"]:.17g}. '
                f'The independently rescored 128-request GPU comparison has worst mean-free pressure relative L2 {worst["pressure_mean_free_relative_l2"]:.17g} and equation relative L2 {worst["common_equation_relative_l2"]:.17g}. '
                'These are different metrics/populations from the logged training loss. This source review adds no latency or own-history drift measurement.',
            lesson=lesson,
            reproduction='Use the exact captured project/train recipe, config, source witness, runtime, typed input bindings, normalization and seeded checkpoint inventory. '
                'Captured training runs execute CUDA gradients/Adam updates, then perform historical CPU deployment/parity checks. '
                'Do not run that recipe unchanged under the current GPU-only neural-forward policy. A clean GPU-only retraining/replay path and new-Rheon data regeneration remain to be qualified. '
                'Final-update selection and no optimizer-resume state are explicit; preserve the old physics epoch and original receipts.',
            original_receipt=receipt_binding,history=binding(training/'history.json'),source_witness=binding(project/'source-witness.json'),
            sources=sources,training_population=binding(training/'splits.json'),training_inputs=bound_inputs,
            original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            retained_accuracy=scores[label]['result'],common_score_source=binding(score_path),
            parameter_count=371729,physical_train_trajectories=expected_trajectories,physical_train_frames=expected_frames,
            physical_validation_trajectories=64,physical_validation_frames=128,training_density_scene_counts=train_densities,
            sample_order_independently_rederived=True,initialized_inventory_sha256=hashlib.sha256(canonical(inventory)).hexdigest(),
            source_review_only=True,new_model_forwards=0,clean_reproduction_qualified=False)
        original = verified_review_receipt(row)
        rewritten, suffix = revised_card(raw,row)
        require(raw.count(original) == rewritten.count(original) == 1 and rewritten.endswith(suffix), 'Original evidence suffix would change')
        rows.append(row)
    require(all(value == initial['C06'] for value in initial.values()), 'Claimed shared initialized tensors differ')
    require(populations['C06'] == populations['C08'] and np.array_equal(orders['C06'],orders['C08'][:2000]),
            'C06/C08 physical population or sample prefix differs')
    require(populations['C09'][:256] == populations['C10'][:256], 'Scaling baseline selection differs')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/b['path']) == b for b in note_bindings),
            'Native project notes/index changed during review')
    require('torch' not in sys.modules, 'No neural runtime is permitted for source review')
    report = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        shared_initialized_inventory_exact=True,C06_C08_physical_population_and2000batch_prefix_exact=True,
        C09_C10_first256physical_selections_exact=True,source_reviewed_fits=5,
        overview_addition='\n\n## Early physical water learning sequence\n\nC06/C08 share 256 training scenes and 512 frames, with 2,000 versus 10,000 updates. C07 is a one-frame memorization control. C09/C10 compare 256 versus 1,024 scenes with one selected frame per scene. All share the same 371,729-parameter initialization and 7-channel pressure architecture; their training density/domain/timestep coverage remains narrow. Different train-derived normalization scales make common physical error metrics preferable to comparing logged losses alone. Clean GPU-only reproduction and new-Rheon generators remain unqualified.\n',
        original_note_bindings=note_bindings,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        clean_reproduction_qualified=False,producer=binding(Path(__file__)))
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Five early physical water fits source-reviewed; no model execution or native mutation.',flush=True)


if __name__ == '__main__':
    run()
