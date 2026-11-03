"""Explain five conditioning experiments from captured source/receipts only.

No captured model module is imported. NumPy only rederives sample order.
The report is authored context for original fits, not new training or lineage.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_water_learning_sequence import binding, canonical, population_identity, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC / 'reports/captured-water-conditioning-method-reviews-01.json'
SPECS = [('C11','constant-low','managed'), ('C12','constant-high','managed-train'),
         ('C13','delayed-decay','managed-train'), ('C14','scalar-units','managed-full'),
         ('C15','no-groupnorm','managed-full')]


def run():
    require(not OUTPUT.exists(), 'Preserve original authored source review')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    score_path = DOC/'reports/pressure-common-score-01.json'
    scores = read(score_path)['models']
    material = {e['id']:e for e in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    rows, populations, initial, scales, orders, notes = [], {}, {}, {}, {}, []
    for label, folder, managed in SPECS:
        project = ROOT/'runs/water-conditioning-v1'/folder/'project'
        training = project.parent/managed/'stage-00/artifacts'
        config, witness = read(project/'config.json'), read(project/'source-witness.json')
        receipt_path = training/'receipt.json'
        receipt, history = read(receipt_path), read(training/'history.json')
        require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
                and receipt['optimizer_updates'] == config['training']['updates'] == 10000
                and config['training']['batch_size'] == 1 and config['training']['seed'] == 20261015
                and config['training']['sampling_seed'] == 2026102501,
                'Original single-frame optimizer budget/seed differs')
        for name, key in [('config.json','experiment_sha256'), ('source-witness.json','source_witness_sha256'),
                          ('runtime-manifest.json','runtime_manifest_sha256')]:
            require(binding(project/name)['sha256'] == receipt['closure'][key], 'Original training closure differs')
        sources = [binding(project/name) for name in sorted(witness['sources'])]
        require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources), 'Captured source changed')
        require(binding(project/'portable_model.py')['sha256'] == '64fae71821a497932fb73b270b75d06d76f6dd9e5438d29d2245d0ca2ae87935',
                'Underlying seven-channel implementation differs')
        manifest, recipe = read(project/'project.json'), read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/'train.py').relative_to(ROOT))
                and recipe['arguments'][0] == dict(literal='train'), 'Original train recipe differs')
        for spec in [*manifest['recipes'], *recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Typed recipe source differs')
        inputs = {e['id']:e for e in manifest['data']}
        bound_inputs = []
        for slot, expected in receipt['closure']['inputs'].items():
            path = ROOT/inputs[slot]['path']
            observed = binding(path)
            require(dict(sha256=observed['sha256'],bytes=path.stat().st_size) == expected, 'Original input differs')
            bound_inputs.append(dict(**observed,slot=slot,bytes=path.stat().st_size))
        for name, expected in receipt['outputs'].items():
            path = training/name
            require(dict(sha256=binding(path)['sha256'],bytes=path.stat().st_size)
                    == {k:expected[k] for k in ('sha256','bytes')}, 'Original retained training output differs')
        require(receipt['outputs']['weights.npz']['sha256'] == scores[label]['weights_sha256']
                == material[label]['weights_sha256'], 'Named checkpoint differs from original final fit')
        split, normalization = read(training/'splits.json'), read(training/'normalization.json')
        parents = split['parents']
        trains = [p for p in parents if p['entry']['split'] == 'train']
        validation = [p for p in parents if p['entry']['split'] == 'validation']
        require(len(trains) == 1 and len(trains[0]['frames']) == 1 and len(validation) == 64
                and sum(len(p['frames']) for p in validation) == 128 and len(parents) == 65,
                'Single-frame versus held-out population differs')
        only = trains[0]
        require(only['counts_xyz'] == [16,16,16] and only['cell_size_m'] == .125
                and only['frames'][0]['index'] == 15 and all(p['entry']['dt_s'] == 1/60 for p in parents),
                'Original optimizer scene/domain/density/timestep differs')
        require(normalization['training_provenance']['frames'] == 1
                and normalization['training_provenance']['cells'] == 4096
                and binding(training/'normalization.json')['sha256'] == receipt['closure']['normalization_sha256'],
                'Train-only normalization differs')
        scales[label] = normalization['scales']
        populations[label] = population_identity(parents)
        initial[label] = history['checkpoints']['checkpoint-000.npz']['inventory']
        require(len(initial[label]) == 38 and sum(math.prod(v['shape']) for v in initial[label].values()) == 371729,
                'Initialized parameter inventory differs')
        eligible = [i for i,p in enumerate(parents) if p['entry']['split'] == 'train']
        rng = np.random.Generator(np.random.PCG64(2026102501))
        order = []
        for _ in range(10000):
            parent = eligible[int(rng.integers(len(eligible)))]
            order.append([(parent,int(rng.integers(len(parents[parent]['frames']))))])
        orders[label] = np.asarray(order,dtype='<i8')
        require(hashlib.sha256(orders[label].tobytes()).hexdigest() == receipt['closure']['sample_order_sha256'],
                'Independently derived sample sequence differs')
        require(receipt['architecture']['input_channels'] == 7
                and receipt['architecture']['block_normalization']['name'] == ('Identity' if label == 'C15' else 'GroupNorm'),
                'Executed normalization architecture differs')
        model_source = (project/'model.py').read_text()
        if label in ('C14','C15'):
            require(config['conditioning'] == 'scalar-units'
                    and "values[:,5] *= scale['rho']/1000." in model_source
                    and "values[:,6] *= scale['dt']/(1/60)" in model_source,
                    'Actual scalar feature transformation differs')
        if label == 'C15':
            require(config['normalization'] == 'none'
                    and 'if isinstance(layer,torch.nn.GroupNorm):block.layers[index]=torch.nn.Identity()' in model_source,
                    'Actual GroupNorm replacement differs')
        training_source = (project/'train.py').read_text()
        require('torch.optim.Adam(network.parameters()' in training_source
                and 'clip_grad_norm_' in training_source, 'Whole-network Adam/gradient clipping differs')
        card = next(f for f in index['fits'] if f['receipt_sha256'] == binding(receipt_path)['sha256'])
        require(not card.get('method_review_status'), 'Existing source review cannot be replaced')
        note = PROJECT/card['card_path']
        raw = note.read_bytes()
        notes.append(binding(note))
        purpose = {
            'C11':'Test a constant 0.0001 learning rate on the single-frame fitting control instead of its original decaying schedule.',
            'C12':'Test a constant 0.001 learning rate on the same single-frame fitting control.',
            'C13':'Delay learning-rate reductions to updates 5,000 and 8,000 on the same single-frame control.',
            'C14':'Condition the scalar input channels in physical units: divide fluid density by 1,000 kg/m3 and timestep by 1/60 s, leaving fixed training RMS amplitude scales.',
            'C15':'Remove parameter-free GroupNorm from the scalar-conditioned network while preserving its trainable tensor inventory.'}[label]
        lesson = {
            'C11':'A lower constant learning rate fits the one frame well but has worse common worst pressure/equation errors than C07. Lower training loss is not evidence of better scene generalization.',
            'C12':'The higher constant rate improves common worst pressure error versus C07, while captured average validation loss is higher. Keep average training objectives and worst field-error comparisons separate.',
            'C13':'Delaying decay gives a small common worst pressure improvement over the constant-high candidate, despite a larger final sampled training loss. Neither schedule resolves the large error on held-out scenes.',
            'C14':'Dimensionless scalar features are plausible conditioning, but this captured run worsens common worst pressure relative L2 above 1.6 while still fitting its training frame. The observation does not establish a universal failure of physical scaling; it establishes that this change did not improve this trained candidate.',
            'C15':'Removing GroupNorm substantially improves common pressure error relative to C14 and lowers the sampled training loss, with the same 371,729 trainable parameters. The execution strategy also changes to resident inputs and combined finite checks, so training throughput is not attributable solely to GroupNorm. It remains a one-frame fitting experiment, without broad training or own-history qualification.'}[label]
        method = ('The same seven-channel multiscale 3D pressure-correction CNN as C07: 371,729 F32 parameters in 38 tensors, '
                  'encoder widths 8/16/24/32, bottleneck 48, decoder widths 32/24/16/8, SiLU and a linear pressure head. '
                  'Residual, previous pressure, metre-valued XYZ coordinates, fluid density and physical timestep are inputs. '
                  'The cell-zero-gauged correction is added to previous pressure using F64 reconstruction and FV arithmetic. '
                  'One call proposes pressure; it does not advance a simulation.')
        method += (' GroupNorm is replaced by Identity; its affine-free form had no trainable parameters.' if label == 'C15' else
                   ' Affine-free four-group GroupNorm remains in each convolutional block.')
        if label in ('C14','C15'):
            method += ' The rho/dt channels use the fixed physical units described in the purpose; residual/output amplitude is not normalized per sample.'
        rate = config['training']['learning_rate']
        schedule = config['training']['schedule']
        last, validation_loss = history['history'][-1], history['validation']
        common = scores[label]['worst']
        row = dict(fit_item_id=card['item_id'],note_path=card['card_path'],named_checkpoint=label,
            method=method,purpose=purpose,
            experiment=f'10,000 whole-network Adam updates, batch 1, initial learning rate {rate:g}, schedule {schedule}, '
                'seed 20261015, PCG64 sample seed 2026102501 and gradient clipping at norm 1.0. '
                'Training uses just native frame 15 from one 16x16x16 scene at 8 cells/metre and a 1/60 s physical timestep. '
                'The 64 validation scenes supply 128 pressure frames at 8/16 cells per metre, with small rectangular grids. '
                'The equal-weight objective combines normalized squared pressure-field and FV equation errors. '
                'Preprocessing RMS scales are fitted on the one training frame; reserved-test bindings remain separate. '
                'The final configured update is selected; no optimizer-resume state is saved.',
            outcome=f'The last sampled training loss is {last["loss"]:.17g}; captured validation objective '
                f'{validation_loss["initial"]:.17g} initially and {validation_loss["final"]:.17g} finally. '
                f'On the common 128 native pressure requests, worst mean-free pressure relative L2 is '
                f'{common["pressure_mean_free_relative_l2"]:.17g} and equation relative L2 is '
                f'{common["common_equation_relative_l2"]:.17g}. These common errors differ from the training objective. '
                'This review adds no timing measurement or drift result.',lesson=lesson,
            reproduction='Use the captured typed train recipe, exact config/source witness/runtime, original input slots, train-only normalization and seeded tensor inventory. '
                'The historical recipe trains on CUDA then performs CPU deployment/parity forwards. A clean reproduction must first qualify a GPU-only path under the current execution policy. '
                'Retain the old-Rheon capture epoch and exact receipts; new-Rheon regeneration, clean retraining and timestep/large-grid generalization remain unqualified.',
            original_receipt=binding(receipt_path),history=binding(training/'history.json'),
            sources=sources,source_witness=binding(project/'source-witness.json'),
            training_population=binding(training/'splits.json'),training_inputs=bound_inputs,
            original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            retained_accuracy=scores[label]['result'],common_score_source=binding(score_path),
            actual_train_trajectories=1,actual_train_frames=1,actual_validation_trajectories=64,actual_validation_frames=128,
            training_cells_per_metre=[8],validation_cells_per_metre=sorted({1/p['cell_size_m'] for p in validation}),
            parameter_count=371729,physical_timestep_s=1/60,trainable_initialized_inventory_sha256=hashlib.sha256(canonical(initial[label])).hexdigest(),
            sample_order_independently_rederived=True,clean_reproduction_qualified=False,new_model_forwards=0)
        original = verified_review_receipt(row)
        rewritten, suffix = revised_card(raw,row)
        require(raw.count(original) == rewritten.count(original) == 1 and rewritten.endswith(suffix), 'Original receipt/configuration suffix changed')
        rows.append(row)
    require(all(initial[k] == initial['C11'] and populations[k] == populations['C11']
                and scales[k] == scales['C11'] and np.array_equal(orders[k],orders['C11']) for k in initial),
            'Claimed shared physical data, initial tensors, scale values or sample order differs')
    low_history = read(ROOT/'runs/water-conditioning-v1/constant-high/managed-train/stage-00/artifacts/history.json')
    decay_history = read(ROOT/'runs/water-conditioning-v1/delayed-decay/managed-train/stage-00/artifacts/history.json')
    require([r for r in low_history['history'] if r['update'] < 5000]
            == [r for r in decay_history['history'] if r['update'] < 5000], 'Common pre-decay training trace differs')
    require(index_path.read_bytes() == index_bytes and all(binding(ROOT/n['path']) == n for n in notes), 'Project changed during source review')
    require('torch' not in sys.modules, 'No neural runtime may be imported by source review')
    report = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        shared_original_population_and_initial_tensors_exact=True,shared_scale_values_and_sample_sequence_exact=True,
        constant_high_and_delayed_decay_logged_updates_before5000_exact=True,
        overview_addition='\n\n## Single-frame conditioning controls\n\nC11–C15 retain one physical training frame and the same initial trainable tensors. Constant/delayed learning-rate schedules alter fitting and common worst errors. Scalar unit conditioning (C14) worsens worst pressure error despite fitting the training frame; removing parameter-free GroupNorm (C15) improves that candidate. Validation contains both 8 and 16 cells/metre, but training contains only 8. None establishes broad-scene, large-grid or own-history accuracy.\n',
        original_note_bindings=notes,source_reviewed_fits=5,new_model_forwards=0,native_mutations=0,
        new_reference_generation=False,clean_reproduction_qualified=False,
        producer=binding(Path(__file__)),helper=binding(ROOT/'experiments/model-portfolio-migration-v1/review_water_learning_sequence.py'))
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Five original conditioning fits source-reviewed; no neural execution or project mutation.',flush=True)


if __name__ == '__main__':
    run()
