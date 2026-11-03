"""Review the four original protocol2 joint temporal generators, without Torch."""
from collections import Counter
import hashlib
import io
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
OUTPUT = DOC/'reports/captured-joint-temporal-generator-method-reviews-01.json'


def verify_archive(manifest, inputs):
    shards = [inputs[f'part_{index:02d}'] for index in range(len(manifest['shards']))]
    require(sum(p.stat().st_size for p in shards) == manifest['archive_bytes'] <= 245*1024**2,
            'Original bounded trajectory archive size differs')
    raw = b''.join(p.read_bytes() for p in shards)
    require(hashlib.sha256(raw).hexdigest() == manifest['archive_sha256'], 'Concatenated archive identity differs')
    verified, total_states = 0, 0
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        expected = {f['key'] for parent in manifest['parents'] for f in parent['fields'].values()}
        require(set(archive.files) == expected, 'Trajectory archive field inventory differs')
        for parent in manifest['parents']:
            total_states += parent['states']
            require(parent['split'] == 'train' and parent['states'] == parent['scene']['steps']+1 == 129,
                    'Full original training trajectory scope differs')
            receipt_path = ROOT/parent['receipt_source']
            require(binding(receipt_path)['sha256'] == parent['receipt_sha256'], 'Original physical capture receipt differs')
            capture = read(receipt_path)
            require(capture['split'] == 'train' and capture['scene'] == parent['scene']
                    and capture['status'] == 'completed' and capture['initial_plus_consecutive_states']
                    and len(capture['states']) == 129
                    and all(s['step'] == i and math.isclose(s['time_s'],i/60,rel_tol=1e-14,abs_tol=1e-14)
                            for i,s in enumerate(capture['states'])), 'Original consecutive 1/60s reference timeline differs')
            for description in parent['fields'].values():
                encoded = archive[description['key']]
                require(encoded.dtype == np.uint8 and encoded.ndim == 3 and encoded.shape[0] == 4,
                        'Lossless float32 byte-plane contract differs')
                bits = encoded.transpose(1,2,0).copy().reshape(encoded.shape[1],encoded.shape[2]*4).view('<u4')
                decoded = np.bitwise_xor.accumulate(bits,axis=0).view('<f4')
                require(list(decoded.shape) == description['shape'] and np.isfinite(decoded).all()
                        and hashlib.sha256(decoded.tobytes()).hexdigest() == description['full_float32_sha256'],
                        'Decoded full trajectory differs from original recorded F32 identity')
                verified += 1
    return dict(archive_bytes=len(raw),decoded_fields=verified,available_states=total_states,
        all_decoded_float32_field_hashes_verified=True,original_physical_capture_receipts_verified=16,
        original_pressure_to_float32_quantization=manifest['worst_field_quantization_relative_l2'],
        native_pressure_retained_precision='F64 according to original capture contracts; this archive contains F32 fitting labels',
        independent_pressure_quantization_remeasurement=False)


def run():
    require(not OUTPUT.exists(), 'Preserve original joint-generator source review')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    availability = read(availability_path)
    selected = [r for r in availability['rows'] if r['original_configuration'].get('schema') == 'rheon-temporal-fit-v2'
                and r['original_configuration'].get('identifier') in ('b1','b2','b3','b4')]
    require(len(selected) == 4, 'Four exact protocol2 original joint generators required')
    selected.sort(key=lambda row:row['declared_horizons'])
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    index = read(index_path)
    cards = {f['item_id']:f for f in index['fits']}
    rows, notes, populations, source_sets, orders = [], [], [], [], []
    archive_facts = None
    for item in selected:
        fid = item['fit_item_id']
        project = ROOT/item['project_path']
        receipt_path = ROOT/item['receipt']['path']
        require(binding(receipt_path)['sha256'] == item['receipt']['sha256'], 'Original fit receipt changed')
        receipt, config = read(receipt_path), read(project/'config.json')
        train = receipt_path.parent
        history = read(train/'history.json')
        require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
                and receipt['optimizer_updates'] == config['updates'] == 10000
                and receipt['horizons'] == config['horizons'] and config['width'] == 4
                and config['feedback_protocol'] == 2 and config['single_block_updates'] == 2000
                and config['sample_step_range'] == [0,120] and config['seed'] == 2026101921
                and config['sampling_seed'] == 2026101922 and config['learning_rate'] == .001
                and config['schedule'] == [dict(update=6001,learning_rate=.0003),dict(update=9001,learning_rate=.0001)],
                'Executed temporal training budget/curriculum differs')
        require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Fit configuration differs')
        sources = [binding(project/name) for name in sorted(receipt['closure']['sources'])]
        require(all(s['sha256'] == receipt['closure']['sources'][Path(s['path']).name] for s in sources), 'Executed source closure differs')
        manifest_project, recipe = read(project/'project.json'), read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/'temporal_train_v2.py').relative_to(ROOT)), 'Actual trainer selection differs')
        for spec in [*manifest_project['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Original typed recipe source differs')
        inputs = {v['id']:ROOT/v['path'] for v in manifest_project['data']}
        bound_inputs = []
        for slot, expected in receipt['closure']['inputs'].items():
            path = inputs[slot]
            observed = binding(path)
            require(dict(sha256=observed['sha256'],bytes=path.stat().st_size) == expected, 'Original trajectory/runtime input differs')
            bound_inputs.append(dict(observed,slot=slot,bytes=path.stat().st_size))
        for name, expected in receipt['outputs'].items():
            path = train/name
            require(dict(sha256=binding(path)['sha256'],bytes=path.stat().st_size) == expected, 'Original fit output differs')
        manifest = read(inputs['manifest'])
        norm = read(inputs['normalization'])
        require(manifest['schema'] == 'rheon-temporal-training-view-v1' and manifest['status'] == 'verified'
                and not manifest['development_or_reserved_used'] and not norm['development_or_reserved_used']
                and manifest['normalization_sha256'] == binding(inputs['normalization'])['sha256']
                and manifest['normalization_sha256'] == receipt['closure']['normalization_sha256'], 'Train-only normalization/view identity differs')
        population = [(p['identifier'],p['scene'],p['states']) for p in manifest['parents']]
        require(len(population) == history['valid_parent_count'] == 16 and history['train_only'], 'Physical fitting population differs')
        if archive_facts is None:
            archive_facts = verify_archive(manifest,inputs)
        else:
            require(population == populations[0] and manifest == read(ROOT/rows[0]['training_manifest']['path']), 'Joint generators use different physical views')
        k = config['horizons']
        rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
        samples = np.asarray([(int(rng.integers(16)),int(rng.integers(121)),1 if u <= 2000 else 2)
                              for u in range(1,10001)],dtype='<i8')
        require(hashlib.sha256(samples.tobytes()).hexdigest() == history['sample_order_sha256'], 'Independent temporal sample order differs')
        padded = [math.prod(n+1 for n in p['scene']['counts']) for p in manifest['parents']]
        enc_cells = sum(padded[int(p)]*int(blocks) for p,_,blocks in samples)
        require(history['optimizer_work'] == dict(network_forwards=18000,encoder_cells=enc_cells,predicted_physical_states=18000*k)
                and history['curriculum'] == dict(single_block_updates=2000,later_blocks=2,second_block_uses_own_predicted_history=True,
                                                 feedback_encoding='physical_reconstruction_then_deployed_mean_centering_v2'),
                'Actual optimizer network work/own-history curriculum differs')
        inventories = {}
        for filename, key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
            saved = receipt[key]
            require(saved == history[key], 'Receipt/history tensor inventories differ')
            with np.load(train/filename,allow_pickle=False) as arrays:
                require(set(arrays.files) == set(saved), 'Full tensor inventory differs')
                for name, spec in saved.items():
                    value = arrays[name]
                    require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                            and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'], 'Original tensor bytes/type/shape differ')
                if key == 'initial_inventory':
                    require(all(np.count_nonzero(arrays[n]) == 0 for n in ('head.weight','head.bias','modal.2.weight','modal.2.bias')),
                            'Joint future heads were not zero initialized')
            inventories[key] = saved
        count = sum(math.prod(t['shape']) for t in inventories['final_inventory'].values())
        require(count == receipt['parameters'] == 8680+4696*k+15*k*k, 'Independent joint architecture parameter count differs')
        trainer = (project/'temporal_train_v2.py').read_text()
        model = (project/'bundle_model_v5.py').read_text()
        features = (project/'temporal_features_v2.py').read_text()
        require('from bundle_model_v5 import BundleNet' in trainer
                and "fine=self.fine(torch.cat((state,features[:,:10]),dim=1))" in model
                and 'self.head=nn.Conv3d(width+3*horizons,5*horizons,1)' in model
                and 'from temporal_features_v2 import packed_state, unpack_outputs' in (project/'temporal_feedback.py').read_text()
                and '/60).reshape(-1,1,1,1)' in features and "log(16/scene['cells_per_metre'])" in features,
                'Executed compact encoder, timing or deployed feedback route differs')
        card = cards[fid]
        require(not card.get('method_review_status'), 'Do not replace an existing reviewed generator')
        densities = Counter(p['scene']['cells_per_metre'] for p in manifest['parents'])
        counts = sorted({tuple(p['scene']['counts']) for p in manifest['parents']})
        last = history['history'][-1]
        row = dict(fit_item_id=fid,note_path=card['card_path'],generator_identifier=config['identifier'],horizons=k,
            original_receipt=binding(receipt_path),history=binding(train/'history.json'),sources=sources,
            original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            training_inputs=bound_inputs,training_manifest=binding(inputs['manifest']),retained_parameter_count=count,
            actual_train_trajectories=16,available_train_states=2064,actual_train_cells_per_metre=dict(densities),actual_counts_xyz=counts,
            physical_output_dt_s=1/60,block_physical_advance_s=k/60,internal_sequential_simulation_advances=0,
            optimizer_network_forwards=18000,optimizer_joint_outputs=18000*k,optimizer_encoder_cells=enc_cells,
            sample_order_independently_rederived=True,fresh_common_full_generator_benchmark_complete=False,
            method=f'A {count:,}-parameter F32 joint full-state generator for K={k}. A shared width4 encoder consumes the padded current five-field state, normalized coordinates, log grid counts/spacing and current known force. One3x3x3 convolution plus1x1 layers produce fine/coarse shared features; inexpensive joint1x1heads also receive all K prescribed force increments. A pooled32hidden-unit branch predicts63nonconstant cosine pressure coefficients per horizon. It emits pressure, three staggered velocities and tracer jointly; pressure is cell-zero gauged, velocity walls are masked, and physical fields are cropped/scaled. Pressure is computed in F32 then cast/gauged in F64 on unpacking; this is not F64 internal pressure prediction. No pressure solver or sequential physical advance is called inside this network.',
            purpose=f'Test amortizing one current-state encoding over {k} simultaneous physical outputs at1/60s spacing, compared with the other protocol2 K1/2/3/4 joint-head generators.',
            experiment=f'10,000CUDA Adam updates, batch1, model seed2026101921/sample seed2026101922. Rate0.001 drops to0.0003 at6001 and0.0001 at9001. Sixteen train-only scenes provide129consecutive states each: eight at8cells/metre and eight at16, with16cubed,9x11x11,15x6x8 and32cubed domains, two seeds per grid/density. Each scene spans128/60seconds of retained native simulation. Updates1–2000supervise one K-output block; later updates add a second block from the first block own reconstructed endpoint, yielding18,000network calls and{18000*k:,}predicted states during fitting. Source state indices are sampled0..120. The loss averages field/equation log-relative-square terms plus divergence, mass, energy and tracer-bound penalties. Known forcing varies in space/time, but its integration and features are fixed at1/60s; no independent dt channel is present. Fitting pressure labels are quantized toF32, with recorded worst conversion relative L2 6.225179570786273e-8; original native evaluation fields remain separate.',
            outcome=f'Original training completed at10,000updates; final logged composite loss {last["loss"]:.17g} with{last["blocks"]}supervised blocks. Original initial/final tensors and CPU probe are retained and byte-verified. No fresh common complete-generator accuracy, latency or own-history drift result is established by this source review. The training loss is not pressure relative L2, and an original CPU probe is not a current GPU benchmark.',
            lesson=f'This is a genuine joint-head K={k} forecast: it shares the main encoder rather than calling a single-step network K times. One inference pass spans{k}/60seconds, while the training curriculum sometimes calls two such blocks for own-history supervision. It does not inherit the near-exact F64 pressure-component anchor: its pressure arithmetic and fitting labels are F32, its outputs include full fluid state, and its input/control interface differs. Fixed1/60s training cannot establish arbitrary timestep, high-density or long-rollout accuracy. Fresh common complete-generator measurements remain required.',
            reproduction='Use the exact typed train recipe, config, eleven trajectory shards, normalization/runtime manifests and complete receipt-bound Python source. Preserve lossless F32 transport, original native reference receipts, zero-initialized heads, private sample order, physical reconstruction between supervised blocks and all loss terms. Historical training and restore recipes execute CPU neural probes, so qualify a GPU-only reproducer before rerunning. This audit performs no neural forward, fit or scene generation; clean retraining and new-Rheon regeneration remain unqualified.')
        original = (PROJECT/card['card_path']).read_bytes()
        receipt_raw = verified_review_receipt(row)
        revised, suffix = revised_card(original,row)
        require(original.count(receipt_raw) == revised.count(receipt_raw) == 1 and revised.endswith(suffix), 'Original card receipt/configuration suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/card['card_path']))
        populations.append(population)
        source_sets.append({Path(s['path']).name:s['sha256'] for s in sources})
        orders.append(samples)
    require(all(s == source_sets[0] for s in source_sets[1:]) and all(np.array_equal(s,orders[0]) for s in orders[1:]),
            'K-sweep changes source implementation or physical sample/curriculum order')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes), 'Current notes/index changed during review')
    require('torch' not in sys.modules, 'Source review imported a neural runtime')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=4,
        new_model_forwards=0,native_mutations=0,new_reference_generation=False,clean_reproduction_qualified=False,
        original_note_bindings=notes,training_archive_audit=archive_facts,
        overview_addition='\n\n## Original protocol2 joint full-state generators\n\nThe K1/2/3/4 compact temporal generators share one spatial encoder and emit all future fields through joint heads. Their later two-block fitting curriculum is separate from inference K and uses reconstructed own history. Training uses16scenes at8/16cells per metre and fixed1/60s, including32cubed; labels/compute areF32. These methods need their own common complete-generator benchmark, not pressure-request timing or near-exact F64 component scores.\n',
        producer=binding(Path(__file__)),helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1'/name) for name in
            ['prepare_method_review_revisions.py','review_water_learning_sequence.py']],availability=binding(availability_path))
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Four joint full-state generators reviewed; original F32 archive/tensors and sample work verified; no neural execution.')


if __name__ == '__main__':
    run()
