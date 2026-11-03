"""Review early temporal feedback, MSE, capacity and 16-update cost controls."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-early-temporal-control-method-reviews-01.json'
IDS = {'fit-af80092df200c9ac','fit-7f855afaded9ff1a','fit-b032c5e5e7166623','fit-945f908e1cc207b8'}


def run():
    require(not OUTPUT.exists(), 'Preserve original early temporal review')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    selected = [r for r in read(availability_path)['rows'] if r['fit_item_id'] in IDS]
    require(len(selected) == 4, 'All four exact historical controls required')
    prior_path = DOC/'reports/captured-joint-temporal-generator-method-reviews-01.json'
    prior = read(prior_path)
    require(prior['training_archive_audit']['all_decoded_float32_field_hashes_verified'], 'Prior full-corpus decoding required')
    prior_by_k = {r['horizons']:r for r in prior['rows']}
    corpus = {r['slot']:{k:r[k] for k in ('sha256','bytes')} for r in prior['rows'][0]['training_inputs']}
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {r['item_id']:r for r in read(index_path)['fits']}
    rows, notes = [], []
    for item in sorted(selected,key=lambda r:r['fit_item_id']):
        project = ROOT/item['project_path']
        receipt_path = ROOT/item['receipt']['path']
        require(binding(receipt_path)['sha256'] == item['receipt']['sha256'], 'Original receipt changed')
        receipt, config = read(receipt_path), read(project/'config.json')
        train = receipt_path.parent
        history = read(train/'history.json')
        k, updates = config['horizons'],config['updates']
        pilot = config['identifier'] == 'b4-cost-pilot'
        global_model = config['identifier'] == 'b1-global'
        mse = config['identifier'] == 'b1-relative-mse'
        old_feedback = config['schema'] == 'rheon-temporal-fit-v1'
        require(receipt['status'] == 'complete' and receipt['mode'] == 'train'
                and updates == receipt['optimizer_updates'] == (16 if pilot else 10000)
                and config['single_block_updates'] == (8 if pilot else 2000)
                and config['seed'] == 2026101921 and config['sampling_seed'] == 2026101922
                and config['sample_step_range'] == [0,120] and config['learning_rate'] == .001
                and config['physical_context']['dt_s'] == 1/60
                and config['schedule'] == ([] if pilot else [dict(update=6001,learning_rate=.0003),dict(update=9001,learning_rate=.0001)]),
                'Original budget/cadence/seed schedule differs')
        require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Original config differs')
        sources = [binding(project/n) for n in sorted(receipt['closure']['sources'])]
        require(all(s['sha256'] == receipt['closure']['sources'][Path(s['path']).name] for s in sources), 'Executed source closure differs')
        document, recipe = read(project/'project.json'), read(project/'train-recipe.json')
        trainer_name = 'temporal_train.py' if old_feedback else ('temporal_train_global.py' if global_model else 'temporal_train_v2.py')
        require(recipe['entrypoint'] == str((project/trainer_name).relative_to(ROOT)), 'Actual trainer differs')
        for spec in [*document['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Typed recipe source differs')
        inputs = {r['id']:ROOT/r['path'] for r in document['data']}
        require(receipt['closure']['inputs'] == corpus, 'Original shared full-trajectory corpus/runtime differs')
        bound_inputs = []
        for slot,expected in corpus.items():
            spec = binding(inputs[slot])
            require(dict(sha256=spec['sha256'],bytes=inputs[slot].stat().st_size) == expected, 'Current original training input differs')
            bound_inputs.append(dict(spec,slot=slot,bytes=expected['bytes']))
        manifest = read(inputs['manifest'])
        require(binding(inputs['manifest']) == prior['rows'][0]['training_manifest']
                and len(manifest['parents']) == history['valid_parent_count'] == 16 and history['train_only'],
                'Original consumed population differs')
        for name,expected in receipt['outputs'].items():
            p = train/name
            require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size) == expected, 'Original fit output differs')
        for filename,key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
            inventory = receipt[key]
            require(inventory == history[key], 'Receipt/history tensor inventory differs')
            with np.load(train/filename,allow_pickle=False) as arrays:
                require(set(arrays.files) == set(inventory), 'Full tensor inventory required')
                for name,spec in inventory.items():
                    value = arrays[name]
                    require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                            and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'], 'Retained tensor bytes/type/shape differ')
                if key == 'initial_inventory':
                    heads = ('head.weight','head.bias') if global_model else ('head.weight','head.bias','modal.2.weight','modal.2.bias')
                    require(all(np.count_nonzero(arrays[n]) == 0 for n in heads), 'Output heads not zero initialized')
        count = sum(math.prod(s['shape']) for s in receipt['final_inventory'].values())
        # Independent width12 global architecture sum: fine, forcing, two local/
        # global blocks, spatial decoder and output head (K1 for this actual fit).
        global_count = 4872+156+48+312+2*(512*12*12+256+33*12*12)+7788+80
        expected_count = global_count if global_model else 8680+4696*k+15*k*k
        require(count == receipt['parameters'] == expected_count, 'Independent architecture count differs')
        old = read(ROOT/prior_by_k[k]['original_receipt']['path'])
        same_initial = receipt['initial_inventory'] == old['initial_inventory']
        if not global_model:
            require(same_initial, 'Compact control initialization differs from corresponding protocol2 model')
        rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
        order = np.asarray([(int(rng.integers(16)),int(rng.integers(121)),1 if u <= config['single_block_updates'] else 2)
                            for u in range(1,updates+1)],dtype='<i8')
        padded = [math.prod(n+1 for n in p['scene']['counts']) for p in manifest['parents']]
        calls = int(order[:,2].sum())
        work = dict(network_forwards=calls,encoder_cells=sum(padded[int(p)]*int(b) for p,_,b in order),predicted_physical_states=calls*k)
        require(hashlib.sha256(order.tobytes()).hexdigest() == history['sample_order_sha256']
                and history['optimizer_work'] == work, 'Independent sample/order/work differs')
        trainer = (project/trainer_name).read_text()
        require(('second=network(predicted[:,-1],' in trainer) if old_feedback else
                ('second=network(next_input(predicted,counts,norm[\'scales\']),' in trainer),
                'Actual second-block feedback route differs')
        require(('feedback_encoding' not in history['curriculum']) if old_feedback else
                history['curriculum']['feedback_encoding'] == 'physical_reconstruction_then_deployed_mean_centering_v2',
                'Original recorded feedback curriculum differs')
        model_name = 'bundle_model_global.py' if global_model else 'bundle_model_v5.py'
        model = (project/model_name).read_text()
        require(f'from {model_name[:-3]} import BundleNet' in trainer
                and 'pressure=state[:,None,0]+delta[:,:,0]' in model, 'Actual model/pressure-persistence route differs')
        if not global_model:
            prior_sources = {Path(s['path']).name:s['sha256'] for s in prior_by_k[k]['sources']}
            require(all(binding(project/n)['sha256'] == prior_sources[n] for n in
                        ['bundle_model_v5.py','adaptive_pool.py','temporal_features_v2.py']), 'Compact architecture/deployed features differ')
        else:
            require('self.global_blocks=nn.ModuleList([GlobalMix(width)for _ in range(2)])' in model
                    and 'self.decode=nn.Sequential(nn.Conv3d(2*width,width,3' in model, 'Actual global/decode architecture differs')
        objective = (project/'temporal_objective.py').read_text()
        require(('log1p' not in objective) if mse else ('torch.log1p(values[name]/1e-6)' in objective), 'Actual robust/MSE objective differs')
        if mse:
            prior_objective = ROOT/next(s['path'] for s in prior_by_k[1]['sources'] if Path(s['path']).name == 'temporal_objective.py')
            require(objective.split('def loss(')[0] == prior_objective.read_text().split('def loss(')[0],
                    'MSE control changes field-error definitions')
        card = cards[item['fit_item_id']]
        require(not card.get('method_review_status'), 'Do not replace a native published review')
        architecture = ('A width12 encoder, two learned global cosine-channel mixing blocks, a3×3decoder and one joint state head. This changes capacity and spatial architecture together; there is no analytical pressure inverse.' if global_model else
            'The same width4 compact current-state encoder, pooled/coarse branch, cheap joint horizon heads and63non-DC learned pressure modes per output as the later protocol2 models. Future prescribed forces enter heads/pooling, and previous pressure is part of the current state.')
        purpose = ('Measure fitting/restoration cost and the original two-block mechanism with only16updates, rather than train a competitive K4model.' if pilot else
            'Test greater shared/global capacity at K1 under the existing logarithmic objective.' if global_model else
            'Test replacing logarithmic error compression with relative MSE while retaining the exact compact K1model and corrected deployed feedback.' if mse else
            'Original fully trained K1direct-generator control before deployment-equivalent feedback encoding was introduced.')
        lesson = ('Keep this16-update smoke/cost fit distinct from10,000-update scientific candidates. Its K4shape and successful receipt do not imply trained accuracy.' if pilot else
            'Capacity and global representation change together, so comparisons cannot assign a gain solely to global mixing. Zero output heads preserve the initial persistence function but not the smaller model’s hidden weights or training cost.' if global_model else
            'The MSE and logarithmic loss scalars have different scales. An objective around0.015versus15is not an accuracy improvement by itself; use the common field/equation benchmark.' if mse else
            'Training’s second block consumes a raw predicted normalized tensor, while deployment reconstructs native physical fields then repacks/mean-centers pressure. Preserve this historical discrepancy when reproducing the fit; the later protocol2 control explicitly changes that boundary.')
        row = dict(fit_item_id=item['fit_item_id'],note_path=card['card_path'],generator_identifier=config['identifier'],horizons=k,
            original_receipt=binding(receipt_path),history=binding(train/'history.json'),sources=sources,
            original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            training_inputs=bound_inputs,training_manifest=binding(inputs['manifest']),retained_parameter_count=count,
            actual_train_trajectories=16,actual_parents_sampled=int(np.unique(order[:,0]).size),available_train_states=2064,
            optimizer_updates=updates,optimizer_network_forwards=calls,actual_optimizer_work=work,
            physical_output_dt_s=1/60,block_physical_advance_s=k/60,raw_tensor_second_block_feedback=old_feedback,
            same_seeded_initialization_as_later_joint_fit=same_initial,
            fresh_common_full_generator_benchmark_complete=False,
            method=f'A {count:,}-parameter F32full-state generator producing K={k}complete pressure/velocity/tracer states per pass. '+architecture+' Pressure is previous normalized pressure plus a predicted correction, gauged at the first cell; velocities/tracer are increments from the original block state. Unpacking upcasts pressure toF64but internal prediction isF32. There is no pressure solver or sequential full physical integrator inside the network.',
            purpose=purpose,
            experiment=f'Original16train-only scenes/2064states at8/16cells per metre and fixed1/60s, with four shapes including32cubed and rectangular domains. All original data/runtime slots match the previously independently decoded corpus. Fit uses{updates}CUDA Adam updates, batch1, private model/sample seeds and the recorded schedule. First{config["single_block_updates"]}updates supervise one block, later updates supervise two, for{calls}network calls/{work["predicted_physical_states"]}physical training predictions. '+('The second call directly consumes predicted[:,-1]without deployed physical unpack/repack.' if old_feedback else 'The second call uses physical reconstruction and deployed mean-centered re-encoding of its own endpoint.')+' All closure/typed recipe/output bytes, full initial/final tensors and PCG64sample/work counts verify.',
            outcome=f'Original fit completes its declared{updates}updates; final logged composite objective {history["history"][-1]["loss"]:.17g}. Parameter count is independently rederived. '+('Exact initial tensors match the corresponding later protocol2joint fit. ' if same_initial else 'The larger global model has a different parameter inventory. ')+'No new common GPU accuracy, full-block latency or own-history drift is measured by this source review. Consumed-training diagnostics are not an independent holdout.',
            lesson=lesson,
            reproduction='Preserve exact captured model, objective, trainer, config, typed recipe/runtime, original16scene F32archive and normalization, private seeds/order and the historical feedback branch. Reproducing an old fit must not silently repair its raw-tensor training feedback. The original publication/restore recipes execute CPU neural probes; qualify a GPU-only reproduction before rerunning. New-Rheon regeneration, clean retraining and600-output drift remain unqualified.')
        raw = (PROJECT/card['card_path']).read_bytes()
        original_receipt = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original_receipt) == revised.count(original_receipt) == 1 and revised.endswith(suffix), 'Original receipt/config suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/card['card_path']))
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/s['path']) == s for s in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural review invariant failed')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=4,
        prior_corpus_and_joint_review=binding(prior_path),original_note_bindings=notes,availability=binding(availability_path),
        new_model_forwards=0,native_mutations=0,new_reference_generation=False,clean_reproduction_qualified=False,
        producer=binding(Path(__file__)),helpers=[binding(Path(__file__).with_name(n)) for n in
            ['prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Four early feedback/MSE/global/cost controls reviewed; complete original tensors/closures/order verified.')


if __name__ == '__main__':
    run()
