"""Review the two fixed-cadence scaling experiments without neural execution."""
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
OUTPUT = DOC/'reports/captured-step-scaled-temporal-generator-method-reviews-01.json'


def force_component(scene, component, velocity_scale):
    """Independent NumPy evaluation at actual faces and the original 128 times."""
    state, mask = scene['seed'], (1 << 64)-1
    uniforms = []
    for _ in range(4):
        state = (state+0x9e3779b97f4a7c15)&mask
        z = state
        z = ((z^(z>>30))*0xbf58476d1ce4e5b9)&mask
        z = ((z^(z>>27))*0x94d049bb133111eb)&mask
        uniforms.append(((z^(z>>31))>>11)/(1 << 53))
    direction = [2*v-1 for v in uniforms[:3]]
    length = math.sqrt(sum(v*v for v in direction))
    direction = [v/length for v in direction]
    phase = uniforms[3]*math.tau
    counts = scene['counts']
    extent = [n/scene['cells_per_metre'] for n in counts]
    positions = []
    for axis,n in enumerate(counts):
        values = (np.arange(n+1 if axis == component else n,dtype=np.float64)
                  +(0 if axis == component else .5))/scene['cells_per_metre']
        shape = [1,1,1,1]
        shape[3-axis] = len(values)
        positions.append(values.reshape(shape))
    omega = math.tau*scene['frequency']
    times = (np.arange(128,dtype=np.float64)/60).reshape(-1,1,1,1)
    center = [extent[a]*(scene['center'][a]+.05*np.sin(omega*times+phase+a)) for a in range(3)]
    offsets = [(positions[a]-center[a])/(scene['radius']*min(extent)) for a in range(3)]
    squared = sum(v*v for v in offsets)
    envelope = np.where(squared < 16,np.exp(-.5*squared)*(1-squared/16)**2,0.)
    a,b = (component+1)%3,(component+2)%3
    rotation = direction[a]*offsets[b]-direction[b]*offsets[a]
    wave = sum(np.sin(math.tau*positions[a]/extent[a]+omega*times*(a+1)+phase+component)
               for a in range(3))/3
    value = (scene['acceleration']*(.65+.35*np.sin(omega*times+phase))*envelope
             *(direction[component]+scene['swirl']*rotation+scene['fluctuation']*wave)/60)
    value = value.astype('<f4')/np.float32(velocity_scale)
    boundary = [slice(None)]*4
    boundary[3-component] = [0,-1]
    value[tuple(boundary)] = 0
    return value


def verify_statistics(manifest, inputs, norm, stats):
    """Decode the exact training bytes and independently remeasure fixed scales."""
    raw = b''.join(inputs[f'part_{i:02d}'].read_bytes() for i in range(len(manifest['shards'])))
    require(len(raw) == manifest['archive_bytes'] and hashlib.sha256(raw).hexdigest() == manifest['archive_sha256'],
            'Original complete training archive required')
    totals = {str(k):{n:dict(squared=0.,elements=0) for n in ('velocity','tracer')} for k in (1,2,3,4)}
    force = dict(squared=0.,elements=0)
    decoded_fields, parents = 0, []
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        require(set(archive.files) == {s['key'] for p in manifest['parents'] for s in p['fields'].values()},
                'Complete field archive inventory required')
        for parent in manifest['parents']:
            require(parent['split'] == 'train' and parent['states'] == 129, 'Original train-only scope required')
            local = {str(k):{n:dict(squared=0.,elements=0) for n in ('velocity','tracer')} for k in (1,2,3,4)}
            for name,spec in parent['fields'].items():
                encoded = archive[spec['key']]
                require(encoded.dtype == np.uint8 and encoded.shape[0] == 4, 'Four original byte planes required')
                delta = encoded.transpose(1,2,0).copy().reshape(encoded.shape[1],-1).view('<u4')
                values = np.bitwise_xor.accumulate(delta,axis=0).view('<f4')
                require(list(values.shape) == spec['shape'] and np.isfinite(values).all()
                        and hashlib.sha256(values.tobytes()).hexdigest() == spec['full_float32_sha256'],
                        'Original decoded training field changed')
                decoded_fields += 1
                if name == 'pressure':
                    continue
                group = 'velocity' if name in ('vx','vy','vz') else 'tracer'
                normalized = (values/np.float32(norm['scales'][group])).astype(np.float64)
                for k in (1,2,3,4):
                    difference = normalized[k:]-normalized[:-k]
                    total = local[str(k)][group]
                    total['squared'] += float(np.square(difference).sum())
                    total['elements'] += difference.size
            for k,groups in local.items():
                for name,spec in groups.items():
                    for key,value in spec.items():
                        totals[k][name][key] += value
            parents.append(dict(identifier=parent['identifier'],increments={k:{n:math.sqrt(v['squared']/v['elements'])
                for n,v in groups.items()} for k,groups in local.items()}))
            for component in range(3):
                value = force_component(parent['scene'],component,norm['scales']['velocity']).astype(np.float64)
                force['squared'] += float(np.square(value).sum())
                force['elements'] += value.size
    for k,groups in totals.items():
        for name,spec in groups.items():
            recorded = stats['aggregate_sums'][k][name]
            require(recorded['elements'] == spec['elements'] and
                    math.isclose(recorded['squared'],spec['squared'],rel_tol=1e-12,abs_tol=1e-12),
                    'Independent aggregate increment statistics differ')
            rms = math.sqrt(spec['squared']/spec['elements'])
            require(math.isclose(rms,stats['normalized_increment_rms'][k][name],rel_tol=1e-12),
                    'Recorded fixed normalized increment scale differs')
            require(math.isclose(rms*norm['scales'][name],stats['physical_increment_rms'][k][name],rel_tol=1e-12),
                    'Physical increment units differ')
    for actual,recorded in zip(parents,stats['parents'],strict=True):
        require(actual['identifier'] == recorded['identifier'] and all(
            math.isclose(value,recorded['increments'][k][name],rel_tol=1e-12)
            for k,groups in actual['increments'].items() for name,value in groups.items()),
            'Independent per-parent increment statistics differ')
    require(force['elements'] == stats['known_force_aggregate']['elements'] and
            math.isclose(force['squared'],stats['known_force_aggregate']['squared'],rel_tol=2e-7),
            'Independent known-force aggregate differs')
    force_rms = math.sqrt(force['squared']/force['elements'])
    require(math.isclose(force_rms,stats['normalized_prescribed_force_rms'],rel_tol=1e-7),
            'Independent fixed force scale differs')
    return dict(decoded_fields=decoded_fields,training_parents=16,available_states=2064,
        increment_aggregate_relative_tolerance=1e-12,force_aggregate_relative_tolerance=2e-7,
        independent_force_rms=force_rms,recorded_force_rms=stats['normalized_prescribed_force_rms'],
        all_horizon_increment_scales_verified=True,all_parent_increment_scales_verified=True,
        method='NumPy byte-plane XOR decode; native unpadded field extents; original F32 state normalization then F64 differences/reduction. Independent NumPy force formula at actual native faces; no Torch or neural forward.')


def run():
    require(not OUTPUT.exists(), 'Preserve original scaling-family review')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    selected = [r for r in read(availability_path)['rows']
                if r['original_configuration'].get('schema') == 'rheon-temporal-fit-step-scaled-v1']
    require(len(selected) == 2, 'Both fixed-cadence scaling fits required')
    prior_path = DOC/'reports/captured-sharing-temporal-generator-method-reviews-01.json'
    prior = read(prior_path)
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {f['item_id']:f for f in read(index_path)['fits']}
    rows, notes, statistics_check, bound_inputs = [], [], None, None
    for item in sorted(selected,key=lambda r:r['original_configuration']['architecture']):
        project = ROOT/item['project_path']
        receipt_path = ROOT/item['receipt']['path']
        require(binding(receipt_path)['sha256'] == item['receipt']['sha256'], 'Original fit receipt changed')
        receipt, config = read(receipt_path), read(project/'config.json')
        train = receipt_path.parent
        history = read(train/'history.json')
        architecture = config['architecture']
        base = next(r for r in prior['rows'] if r['architecture'] == architecture and r['horizons'] == 2)
        require(config['horizons'] == 2 and config['width'] == 12 and config['feedback_protocol'] == 2
                and config['updates'] == receipt['optimizer_updates'] == 10000
                and config['single_block_updates'] == 2000 and config['seed'] == 2026101921
                and config['sampling_seed'] == 2026101922 and config['sample_step_range'] == [0,120]
                and config['learning_rate'] == .001 and config['schedule'] == [dict(update=6001,learning_rate=.0003),dict(update=9001,learning_rate=.0001)]
                and config['physical_context']['dt_s'] == 1/60 and receipt['status'] == 'complete' and receipt['mode'] == 'train',
                'Original controlled training budget/cadence differs')
        require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Fit config changed')
        sources = [binding(project/n) for n in sorted(receipt['closure']['sources'])]
        require(all(s['sha256'] == receipt['closure']['sources'][Path(s['path']).name] for s in sources),
                'Original executed source closure changed')
        document, recipe = read(project/'project.json'), read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/'temporal_train_step_scaled.py').relative_to(ROOT)), 'Actual trainer differs')
        for spec in [*document['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Original typed recipe source differs')
        inputs = {v['id']:ROOT/v['path'] for v in document['data']}
        require(receipt['closure']['inputs'] == {i['slot']:{k:i[k] for k in ('sha256','bytes')} for i in base['training_inputs']},
                'Scaling fit changed the shared original corpus/runtime inputs')
        actual_inputs = []
        for slot,expected in receipt['closure']['inputs'].items():
            spec = binding(inputs[slot])
            require(dict(sha256=spec['sha256'],bytes=inputs[slot].stat().st_size) == expected, 'Original fitting input changed')
            actual_inputs.append(dict(spec,slot=slot,bytes=expected['bytes']))
        if bound_inputs is None:
            bound_inputs = {r['slot']:r['sha256'] for r in actual_inputs}
        else:
            require(bound_inputs == {r['slot']:r['sha256'] for r in actual_inputs}, 'Paired corpus identity differs')
        stats_path = project/'step-scales.json'
        require(binding(stats_path)['sha256'] == config['step_scale_statistics_sha256'], 'Fixed statistics changed')
        stats, norm, manifest = read(stats_path), read(inputs['normalization']), read(inputs['manifest'])
        require(stats['status'] == 'verified_train_only_statistics' and not stats['development_or_reserved_used']
                and stats['training_parent_count'] == 16 and not manifest['development_or_reserved_used']
                and not norm['development_or_reserved_used']
                and stats['fitting_view_sha256'] == binding(inputs['manifest'])['sha256']
                and stats['original_normalization_sha256'] == config['original_normalization_sha256'] == binding(inputs['normalization'])['sha256'],
                'Statistics were not bound to the unchanged original training view')
        statistics_source = ROOT/'experiments/full-ai-next-priorities-v1/prepare_temporal_step_scales.py'
        require(binding(statistics_source)['sha256'] == stats['source_sha256'], 'Original statistics producer changed')
        if statistics_check is None:
            statistics_check = verify_statistics(manifest,inputs,norm,stats)
        else:
            require(binding(stats_path)['sha256'] == rows[0]['fixed_statistics']['sha256'], 'Paired fixed scales differ')
        for name,expected in receipt['outputs'].items():
            p = train/name
            require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size) == expected, 'Original fit output changed')
        old_receipt = read(ROOT/base['original_receipt']['path'])
        require(binding(ROOT/base['original_receipt']['path']) == base['original_receipt'], 'Prior same-architecture receipt changed')
        require(receipt['initial_inventory'] == old_receipt['initial_inventory'], 'Scaling experiment changed seeded parameter initialization')
        for filename,key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
            inventory = receipt[key]
            require(inventory == history[key], 'Receipt/history tensor inventory differs')
            with np.load(train/filename,allow_pickle=False) as arrays:
                require(set(arrays.files) == set(inventory), 'Complete tensor inventory required')
                for name,spec in inventory.items():
                    value = arrays[name]
                    require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                            and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'], 'Retained parameter bytes differ')
        count = sum(math.prod(s['shape']) for s in receipt['final_inventory'].values())
        require(count == receipt['parameters'] == base['retained_parameter_count'], 'Scaling must not add learned parameters')
        rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
        order = np.asarray([(int(rng.integers(16)),int(rng.integers(121)),1 if u <= 2000 else 2) for u in range(1,10001)],dtype='<i8')
        require(hashlib.sha256(order.tobytes()).hexdigest() == history['sample_order_sha256']
                and history['optimizer_work'] == base['actual_optimizer_work'] and history['train_only']
                and history['valid_parent_count'] == 16, 'Original sample/order/work differs')
        model = (project/'bundle_model_step_scaled.py').read_text()
        trainer = (project/'temporal_train_step_scaled.py').read_text()
        require('features[:,7:]/self.forcing_scale' in model
                and "raw[:,:,index]*gains['velocity']" in model and "raw[:,:,4]*gains['tracer']" in model
                and "fine=self.fine(torch.cat((state[:,1:],features[:,:10]),dim=1))" in model
                and "forcing_scale=stats['normalized_prescribed_force_rms'],increment_scales=stats['normalized_increment_rms']" in trainer
                and 'from bundle_model_step_scaled import BundleNet' in trainer,
                'Actual train-only scaling constructor/reconstruction differs')
        card = cards[item['fit_item_id']]
        require(not card.get('method_review_status'), 'Do not replace published review')
        last = history['history'][-1]
        row = dict(fit_item_id=item['fit_item_id'],note_path=card['card_path'],generator_identifier=config['identifier'],
            architecture=architecture,horizons=2,original_receipt=binding(receipt_path),history=binding(train/'history.json'),
            sources=sources,original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            training_inputs=actual_inputs,training_manifest=binding(inputs['manifest']),fixed_statistics=binding(stats_path),
            statistics_source=binding(statistics_source),same_architecture_base_fit=base['fit_item_id'],
            exact_initial_parameters_equal_to_base=True,retained_parameter_count=count,
            actual_train_trajectories=16,available_train_states=2064,physical_output_dt_s=1/60,block_physical_advance_s=2/60,
            optimizer_network_forwards=18000,actual_optimizer_work=history['optimizer_work'],
            fresh_common_full_generator_benchmark_complete=False,
            method=f'A {count:,}-parameter F32 width12 K2 {architecture} generator with the same learned spatial/global encoder and heads as its original unscaled counterpart. Previous pressure is excluded. The prescribed-force channels are divided by fixed train-only RMS {stats["normalized_prescribed_force_rms"]:.17g}. Raw velocity/tracer increments are multiplied by their fixed horizon-specific normalized training RMS before addition to the original block state. Pressure remains a fresh gauged output; physical input/output normalization is unchanged. The endpoint branch predicts its last state and jointly decodes intermediate states; the reuse branch advances retained features using prior predictions. No numerical pressure solve or future query label is used.',
            purpose='Test whether matching force and increment magnitudes improves trainability of the existing compute-sharing architecture without adding parameters or changing simulated cadence. The name step-scaled refers to increment units, not variable dt or longer-time training.',
            experiment='Original16train-only scenes/2064states at8/16cells per metre, four shapes including32cubed and rectangular domains, fixed1/60s outputs and K2blocks spanning2/60s. Fixed RMS values are computed once over all training-parent increments at horizons1–4 and prescribed forces. No development/reserved data is used. Two fits retain their corresponding unscaled architecture’s exact seeded initial tensors, optimizer schedule, private sample stream,10,000CUDAupdates and18,000encoder calls under the2000one-block/8000two-own-history-block curriculum. Complete source/typed recipe/input/output identities and initial/final tensor inventories verify.',
            outcome=f'The original fit completed10,000updates; last logged composite objective {last["loss"]:.17g}. Independent NumPy decoding checks all80original field hashes and rederives every parent/aggregate increment scale plus known-force RMS. The paired architecture adds zero parameters and preserves exact seeded initialization. Fresh common GPU full-field accuracy, block timing and own-history drift remain unmeasured for this fit. A normalized composite objective is not a pressure relative-error result.',
            lesson='This is a controlled conditioning/optimization experiment rather than proof that larger temporal strides generalize. The K2endpoint or latent-reuse structure can amortize encoder work, but fixed scale factors do not certify a high-accuracy endpoint, remove F32 prediction arithmetic, or recover the near-exact F64 anchors. Preserve the candidate for common benchmarking instead of rejecting it from an obsolete timing target.',
            reproduction='Retain exact captured model/trainer/config/typed recipe/runtime, original16scene F32archive and normalization, the separately source-bound step-scales.json, private seeds/order and deployed feedback reconstruction. The fixed scales are constructor inputs outside the NPZ tensor inventory; weights alone are insufficient. Original publication/restore recipes execute CPU neural probes and require GPU-only qualification before rerunning. Clean retraining, new-Rheon data generation and600-output drift remain unqualified.')
        raw = (PROJECT/card['card_path']).read_bytes()
        original_receipt = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original_receipt) == revised.count(original_receipt) == 1 and revised.endswith(suffix),
                'Original receipt/config suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/card['card_path']))
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/s['path']) == s for s in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural review invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=2,
        independent_statistics_verification=statistics_check,new_model_forwards=0,native_mutations=0,
        new_reference_generation=False,clean_reproduction_qualified=False,original_note_bindings=notes,
        prior_sharing_review=binding(prior_path),availability=binding(availability_path),producer=binding(Path(__file__)),
        helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1'/n) for n in
            ['prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Two fixed-cadence scaling fits reviewed; full train-only statistics independently verified; no neural execution.')


if __name__ == '__main__':
    run()
