"""Review the no-new-parameters curl representation and its actual guarantees."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_temporal_review import verified_review_receipt
from prepare_method_review_revisions import revised_card
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-curl-temporal-generator-method-review-01.json'


def curl_identity_checks():
    """Independent NumPy edge/face incidence identity; not Torch/model parity."""
    rng = np.random.Generator(np.random.PCG64(20261025))
    observations = []
    for nx,ny,nz in ((3,4,5),(2,3,4),(7,2,3)):
        ax,ay,az = np.zeros((3,nz+1,ny+1,nx+1),dtype=np.float64)
        ax[1:nz,1:ny,:nx] = rng.normal(size=(nz-1,ny-1,nx))
        ay[1:nz,:ny,1:nx] = rng.normal(size=(nz-1,ny,nx-1))
        az[:nz,1:ny,1:nx] = rng.normal(size=(nz,ny-1,nx-1))
        vx = np.diff(az[:nz,:,:nx+1],axis=1)-np.diff(ay[:,:ny,:nx+1],axis=0)
        vy = np.diff(ax[:,:ny+1,:nx],axis=0)-np.diff(az[:nz,:ny+1,:],axis=2)
        vz = np.diff(ay[:nz+1,:ny,:],axis=2)-np.diff(ax[:nz+1,:,:nx],axis=1)
        divergence = np.diff(vx,axis=2)+np.diff(vy,axis=1)+np.diff(vz,axis=0)
        maximum = float(np.abs(divergence).max())
        walls = [vx[:,:,[0,-1]],vy[:,[0,-1],:],vz[[0,-1],:,:]]
        require(maximum < 1e-12 and all(np.count_nonzero(v) == 0 for v in walls),
                'Independent closed-wall discrete curl identity failed')
        observations.append(dict(counts_xyz=[nx,ny,nz],maximum_absolute_F64_divergence=maximum,normal_wall_flux_exact_zero=True))
    return observations


def run():
    require(not OUTPUT.exists(), 'Preserve original curl review')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    item = next(r for r in read(availability_path)['rows'] if r['fit_item_id'] == 'fit-a47365bce83a4ce1')
    prior_path = DOC/'reports/captured-increment-supervision-generator-method-reviews-01.json'
    base = next(r for r in read(prior_path)['rows'] if r['supervision_mode'] == 'increments')
    project = ROOT/item['project_path']
    receipt_path = ROOT/item['receipt']['path']
    require(binding(receipt_path)['sha256'] == item['receipt']['sha256'], 'Original curl receipt changed')
    config, receipt = read(project/'config.json'), read(receipt_path)
    train = receipt_path.parent
    history = read(train/'history.json')
    previous_receipt = read(ROOT/base['original_receipt']['path'])
    require(config['schema'] == 'rheon-temporal-fit-curl-readout-v1' and config['architecture'] == 'endpoint'
            and config['horizons'] == 2 and config['width'] == 12 and config['supervision_mode'] == 'increments'
            and config['updates'] == receipt['optimizer_updates'] == 10000 and config['single_block_updates'] == 2000
            and config['seed'] == 2026101921 and config['sampling_seed'] == 2026101922
            and config['learning_rate'] == .001 and config['schedule'] == [dict(update=6001,learning_rate=.0003),dict(update=9001,learning_rate=.0001)]
            and receipt['status'] == 'complete' and receipt['mode'] == 'train', 'Original matched curl fit differs')
    require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Original config changed')
    sources = [binding(project/n) for n in sorted(receipt['closure']['sources'])]
    require(all(s['sha256'] == receipt['closure']['sources'][str((ROOT/s['path']).relative_to(project))] for s in sources),
            'Actual curl source closure changed')
    document, recipe = read(project/'project.json'), read(project/'train-recipe.json')
    require(recipe['entrypoint'] == str((project/'temporal_train_curl.py').relative_to(ROOT)), 'Actual curl trainer differs')
    for spec in [*document['recipes'],*recipe['sources']]:
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Original typed recipe/source differs')
    expected_inputs = {r['slot']:{k:r[k] for k in ('sha256','bytes')} for r in base['training_inputs']}
    inputs = {r['id']:ROOT/r['path'] for r in document['data']}
    require(receipt['closure']['inputs'] == expected_inputs, 'Curl fit changes original shared corpus/runtime')
    bound_inputs = []
    for slot,expected in expected_inputs.items():
        spec = binding(inputs[slot])
        require(dict(sha256=spec['sha256'],bytes=inputs[slot].stat().st_size) == expected, 'Original corpus input changed')
        bound_inputs.append(dict(spec,slot=slot,bytes=expected['bytes']))
    require(binding(project/'step-scales.json')['sha256'] == config['step_scale_statistics_sha256'] == base['fixed_statistics']['sha256'],
            'Independently audited fixed scales differ')
    for name,expected in receipt['outputs'].items():
        p = train/name
        require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size) == expected, 'Original fit output changed')
    require(receipt['initial_inventory'] == previous_receipt['initial_inventory'], 'Curl representation changes initial learned tensors')
    for filename,key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
        inventory = receipt[key]
        require(inventory == history[key], 'Receipt/history inventory differs')
        with np.load(train/filename,allow_pickle=False) as arrays:
            require(set(arrays.files) == set(inventory), 'Complete tensor inventory required')
            for name,spec in inventory.items():
                value = arrays[name]
                require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                        and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'], 'Original tensor bytes/type/shape differ')
    count = sum(math.prod(s['shape']) for s in receipt['final_inventory'].values())
    require(count == receipt['parameters'] == 163068 and history['optimizer_work'] == base['actual_optimizer_work'],
            'Curl representation changes learned count/training work')
    rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
    order = np.asarray([(int(rng.integers(16)),int(rng.integers(121)),1 if u <= 2000 else 2) for u in range(1,10001)],dtype='<i8')
    require(hashlib.sha256(order.tobytes()).hexdigest() == history['sample_order_sha256'], 'Private sample stream differs')
    base_sources = {Path(s['path']).name:s['sha256'] for s in base['sources']}
    require(all(binding(project/n)['sha256'] == base_sources[n] for n in
        ['bundle_model_step_scaled.py','temporal_features_v2.py','temporal_objective_increment.py','temporal_feedback.py']),
        'Matched network/loss/deployed feature semantics differ')
    model = (project/'bundle_model_curl.py').read_text()
    require('class BundleNet(DirectBundleNet):' in model and 'delta=velocity_curl(raw[:,:,1:4].double(),counts)' in model
            and "gains['velocity'].double()*min(counts)*value" in model
            and 'from bundle_model_curl import BundleNet' in (project/'temporal_train_curl.py').read_text(),
            'Actual potential/curl readout selection differs')
    identity_checks = curl_identity_checks()
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    card = next(r for r in read(index_path)['fits'] if r['item_id'] == item['fit_item_id'])
    require(not card.get('method_review_status'), 'Do not replace published review')
    row = dict(fit_item_id=item['fit_item_id'],note_path=card['card_path'],generator_identifier=config['identifier'],horizons=2,
        original_receipt=binding(receipt_path),history=binding(train/'history.json'),sources=sources,
        source_closure_semantics='project_relative_paths_including_expert_assets',original_configuration=binding(project/'config.json'),
        original_recipe=binding(project/'train-recipe.json'),training_inputs=bound_inputs,training_manifest=binding(inputs['manifest']),
        fixed_statistics=binding(project/'step-scales.json'),retained_parameter_count=count,exact_initial_parameters_equal_to_MSE_control=True,
        actual_train_trajectories=16,available_train_states=2064,physical_output_dt_s=1/60,block_physical_advance_s=2/60,
        optimizer_updates=10000,optimizer_network_forwards=18000,actual_optimizer_work=history['optimizer_work'],
        fresh_common_full_generator_benchmark_complete=False,
        method='A163,068-parameter F32width12K2endpoint-conditioned generator with the same learned tensors, shared global encoder and fixed train-only scales as the increment-MSE control. Its three raw velocity channels represent edge potentials. A fixed masked discrete curl in F64forms staggered velocity increments, scaled by the horizon RMS and minimum grid dimension, then added to current velocity and cast toF32. Pressure/tracer readouts remain direct learned predictions. No new trainable coefficient, pressure solver or post-prediction velocity projection is added.',
        purpose='Represent learned velocity increments so their ideal discrete divergence cancels by construction and wall-normal increments vanish, while retaining the same network capacity and matched objective/data/initialization.',
        experiment='Original16consumed scenes/2064states at8/16cells per metre, four shapes including32cubed/rectangles, fixed1/60s output intervals and2/60s per block.10,000CUDAupdates use the same private seeds/order, learning-rate schedule and2000one-block then8000two-own-history-block curriculum as the increment-MSE endpoint control. Full closure/typed recipe/input/output identities, exact initial tensors, complete final tensors and18,000encoder-call counts verify. Independent NumPy edge/face incidence checks on three rectangular grids confirm the ideal F64curl/divergence identity and zero normal-wall increments; they are not a new neural execution or Torch-output equivalence test.',
        outcome=f'Original training completes10,000updates; final composite objective {history["history"][-1]["loss"]:.17g}. The learned parameter count and initialization match the control exactly. Independent ideal-readout identities pass, but fresh complete-model GPU errors, block timing and600-output drift remain pending.',
        lesson='A divergence-free increment preserves existing divergence rather than removing it; F32addition/rounding can also introduce a residual. The representation does not guarantee pressure-equation consistency, tracer accuracy or low trajectory drift. Pressure/tracer heads remain learned separately. Compare actual full-state fields and divergence, not the mathematical identity alone.',
        reproduction='Preserve exact curl masking/incidence/readout code, endpoint network/config/objective/typed recipe/runtime, original16scene F32archive/normalization and fixed statistics, private seeds/order and deployed feedback. There are no additional learned curl weights. Original final publication/restore runs CPU neural probes; qualify a GPU-only recipe first. Clean retraining, new-Rheon data generation, fresh common benchmarking and600-output drift remain unqualified.')
    raw = (PROJECT/card['card_path']).read_bytes()
    original_receipt = verified_review_receipt(row)
    revised,suffix = revised_card(raw,row)
    require(raw.count(original_receipt) == revised.count(original_receipt) == 1 and revised.endswith(suffix)
            and index_path.read_bytes() == index_raw and 'torch' not in sys.modules, 'Read-only original receipt/no-neural invariant failed')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=[row],source_reviewed_fits=1,
        ideal_F64_readout_identity_checks=identity_checks,original_note_bindings=[binding(PROJECT/card['card_path'])],
        prior_matched_MSE_review=binding(prior_path),availability=binding(availability_path),new_model_forwards=0,
        native_mutations=0,new_reference_generation=False,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in
            ['captured_temporal_review.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Curl representation fit reviewed; unchanged learned tensors and three ideal discrete identities verified.')


if __name__ == '__main__':
    run()
