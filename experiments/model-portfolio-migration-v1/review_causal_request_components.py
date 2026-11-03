"""Review causal request predictors and flux calibration without any forward."""
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
BASE = ROOT/'runs/full-ai-next-priorities-v1'
OUTPUT = DOC/'reports/captured-causal-request-component-method-reviews-01.json'
SPECS = [('fit-a9e199535346c3d9','b1-causal-request-project-01','b1-causal-request-pair-01'),
         ('fit-ccb969376d283836','b1-factored-request-project-02','b1-factored-request-pair-02')]


def verify_archive(manifest, inputs):
    raw = b''.join(inputs[f'part_{i:02d}'].read_bytes() for i in range(len(manifest['shards'])))
    require(len(raw) == manifest['archive_bytes'] <= 250*1024**2
            and hashlib.sha256(raw).hexdigest() == manifest['archive_sha256'], 'Exact complete causal-request archive required')
    fields, states = 0,0
    with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
        require(set(arrays.files) == {v['key'] for p in manifest['parents'] for v in p['fields'].values()},
                'Complete velocity/request field archive required')
        for parent in manifest['parents']:
            require(parent['split'] == 'train' and parent['states'] == 129
                    and set(parent['fields']) == {'vx','vy','vz','q'}, 'Exact causal fitting fields required')
            require(binding(ROOT/parent['receipt_source'])['sha256'] == parent['receipt_sha256'], 'Original capture receipt changed')
            states += parent['states']
            for spec in parent['fields'].values():
                encoded = arrays[spec['key']]
                require(encoded.dtype == np.uint8 and encoded.shape[0] == 4, 'Lossless F32 byte-plane contract differs')
                delta = encoded.transpose(1,2,0).copy().reshape(encoded.shape[1],-1).view('<u4')
                decoded = np.bitwise_xor.accumulate(delta,axis=0).view('<f4')
                require(list(decoded.shape) == spec['shape'] and np.isfinite(decoded).all()
                        and hashlib.sha256(decoded.tobytes()).hexdigest() == spec['full_float32_sha256'],
                        'Original full decoded field identity differs')
                fields += 1
    require(fields == 64 and states == 2064, 'All16complete causal fitting trajectories required')
    return dict(decoded_fields=fields,states=states,archive_bytes=len(raw),all_decoded_fields_verified=True)


def run():
    require(not OUTPUT.exists(), 'Preserve original causal-request component reviews')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {r['item_id']:r for r in read(index_path)['fits']}
    rows, notes, archive_check, sample_sha = [], [], None, None
    for sequence,(fid,folder,pair) in enumerate(SPECS):
        factored = sequence == 1
        project,train = BASE/folder,BASE/pair/'train'
        receipt_path = train/'receipt.json'
        receipt,config,history = read(receipt_path),read(project/'config.json'),read(train/'history.json')
        require(receipt['status'] == 'complete' and receipt['mode'] == 'train' and receipt['component_only']
                and not receipt['full_state_temporal_candidate'] and not receipt['candidate_numerical_pressure_solve']
                and receipt['optimizer_updates'] == config['updates'] == (11000 if factored else 10000)
                and config['horizons'] == 1 and config['seed'] == 2026101994 and config['sampling_seed'] == 2026101995
                and config['component_only'] and not config['future_request_input'] and not config['current_pressure_input']
                and config['learning_rate'] == .003 and config['schedule'] == [dict(update=6001,learning_rate=.001),dict(update=9001,learning_rate=.0003)],
                'Actual bounded causal component/budget differs')
        require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Original config changed')
        sources = [binding(project/n) for n in sorted(receipt['closure']['sources'])]
        require(all(s['sha256'] == receipt['closure']['sources'][Path(s['path']).name] for s in sources), 'Captured source closure changed')
        document,recipe = read(project/'project.json'),read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/'pressure_request_train.py').relative_to(ROOT)), 'Actual trainer differs')
        for spec in [*document['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Typed recipe/source changed')
        inputs = {r['id']:ROOT/r['path'] for r in document['data']}
        bound_inputs = []
        for slot,expected in receipt['closure']['inputs'].items():
            spec = binding(inputs[slot])
            require(dict(sha256=spec['sha256'],bytes=inputs[slot].stat().st_size) == expected, 'Original fitting input changed')
            bound_inputs.append(dict(spec,slot=slot,bytes=expected['bytes']))
        manifest,norm = read(inputs['manifest']),read(inputs['normalization'])
        require(manifest['schema'] == 'rheon-causal-pressure-request-training-view-v1' and manifest['status'] == 'verified'
                and not manifest['development_or_reserved_used'] and not manifest['current_pressure_input']
                and not manifest['reference_pressure_input'] and not norm['development_or_reserved_used']
                and manifest['normalization_sha256'] == receipt['closure']['normalization_sha256'] == binding(inputs['normalization'])['sha256']
                and history['valid_parent_count'] == len(manifest['parents']) == 16 and history['train_only']
                and not history['own_history_supported'] and not history['future_request_input'], 'Original causal input/label population differs')
        if archive_check is None:
            archive_check = verify_archive(manifest,inputs)
        else:
            require(binding(inputs['manifest']) == rows[0]['training_manifest'], 'Paired component uses another fitting view')
        for name,expected in receipt['outputs'].items():
            p = train/name
            require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size) == expected, 'Original fit output changed')
        require(cards[fid]['receipt_sha256'] == binding(receipt_path)['sha256'] and not cards[fid].get('method_review_status'),
                'Card is another/already-reviewed fit')
        for filename,key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
            inventory = receipt[key]
            require(inventory == history[key], 'Original receipt/history inventory differs')
            with np.load(train/filename,allow_pickle=False) as arrays:
                require(set(arrays.files) == set(inventory), 'Complete retained parameter inventory required')
                for name,spec in inventory.items():
                    value = arrays[name]
                    require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                            and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'], 'Original tensor bytes/type/shape differ')
        count = (3*27+2*16*3*27+16*80+16*27) if factored else (6*27+2*8*6*27+16*27+6*125)
        require(count == receipt['parameters'] == (4385 if factored else 3936), 'Independent architecture parameter count differs')
        rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
        order = []
        for _ in range(10000):
            parent = int(rng.integers(16))
            bucket = float(rng.random())
            step = 0 if bucket < .25 else int(rng.integers(1,9)) if bucket < .5 else int(rng.integers(9,121))
            order.append((parent,step))
        order = np.asarray(order,dtype='<i8')
        measured = hashlib.sha256(order.tobytes()).hexdigest()
        require(measured == history['sample_order_sha256'] and (sample_sha is None or measured == sample_sha),
                'Private biased-start sampling order differs')
        sample_sha = measured
        padded = [math.prod(n+1 for n in p['scene']['counts']) for p in manifest['parents']]
        work = dict(network_forwards=10000,predicted_request_fields=10000,input_cells=sum(padded[int(p)] for p,_ in order))
        require(history['optimizer_work'] == work, 'Actual request/cell optimizer work differs')
        inherited_flux = None
        if factored:
            linear = history['linear_calibration']
            require(receipt['linear_optimizer_updates'] == linear['updates'] == config['linear_pretrain_updates'] == 1000
                    and receipt['main_optimizer_updates'] == config['main_updates'] == 10000
                    and linear['train_patterns'] == 64 and linear['development_patterns'] == 16
                    and not linear['temporal_scenes_used'] and linear['pressure_solves'] == 0
                    and history['main_initial_inventory']['flux.weight'] == receipt['final_inventory']['flux.weight']
                    and receipt['final_inventory']['flux.weight']['sha256'] == linear['learned_weight_sha256'],
                    'Separate flux calibration/frozen main-fit work differs')
            require(all(history['main_initial_inventory'][n] == s for n,s in receipt['initial_inventory'].items() if n != 'flux.weight'),
                    'Flux-only stage changes unrelated initialization')
            with np.load(BASE/'b2-coupled-project-01/expert-assets/request-weights.npz',allow_pickle=False) as arrays:
                require(hashlib.sha256(arrays['flux.weight'].tobytes()).hexdigest() == linear['learned_weight_sha256'],
                        'Later coupled generator inherits different flux coefficients')
            inherited_flux = binding(BASE/'b2-coupled-project-01/expert-assets/request-weights.npz')
        model = (project/'pressure_request_model.py').read_text()
        data = (project/'pressure_request_data.py').read_text()
        objective = (project/'pressure_request_objective.py').read_text()
        require('return velocity,known,scalars' in data and 'pressure_request_objective' not in model
                and 'teacher_pressure(difference)' in objective and 'scalars' in model
                and '1/60,1000.,.001' in data, 'Actual causal feature/training-only inverse route differs')
        row = dict(fit_item_id=fid,note_path=cards[fid]['card_path'],original_receipt=binding(receipt_path),history=binding(train/'history.json'),
            sources=sources,original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            training_inputs=bound_inputs,training_manifest=binding(inputs['manifest']),retained_parameter_count=count,horizons=1,
            actual_train_trajectories=16,available_train_states=2064,target_time_offset_s=1/60,
            actual_optimizer_work=work,optimizer_updates=config['updates'],private_sample_order_verified=True,
            inherited_frozen_flux_asset=inherited_flux,original_q_label_quantization=manifest['worst_q_quantization_relative_l2'],
            method=(f'A {count:,}-parameter F32 causal one-step pressure-request predictor. The factored branch uses an 81-coefficient learned linear flux on current velocity plus known force, a velocity-only nonlinear branch with width 16, and physical h/dt/rho unit conditioning. The calibrated flux is frozen during the 10,000-update nonlinear fit. Final output is a mean-free future q request, not pressure or a complete simulation state.' if factored else
                f'A {count:,}-parameter F32 causal one-step pressure-request predictor. A learned 3³ linear branch, two width-eight local projections with quadratic products, a learned quadratic readout and a 5³ viscous branch consume current velocity and known force. Per-example h, dt, rho and mu condition physical unit factors. It outputs the mean-free next q request; current pressure and future reference requests are not inputs.'),
            purpose=('Separate a native-calibrated linear request operator from learned velocity-dependent nonlinear terms, testing whether the structured decomposition improves causal request prediction.' if factored else
                'Learn a future pressure request causally from current velocity and prescribed controls, so later pressure experts can be driven without receiving a future reference request.'),
            experiment='Use 16 original consumed fitting trajectories with 129 states each, at 8/16 cells per metre and four shapes including 32³ and rectangles. All 64 archived velocity/request field hashes are independently decoded and checked. Training selects initial step 0 with 25% probability, steps 1–8 with 25%, and steps 9–120 with 50%; full private PCG64 order and 10,000 main root calls verify. Targets are separately stored q at step+1. '+('An additional 1,000 full-batch F64 SGD updates calibrate flux on 64 manufactured patterns with 16 development patterns, then copy the learned coefficients to F32 and freeze them. The main F32 Adam fit has 4,304 trainable coefficients. ' if factored else 'All 3,936 coefficients train with F32 Adam. ')+'Both main fits use rate 0.003 with drops at 6001/9001, gradient cap 1 and final-update selection. The offline objective applies a fixed analytical pressure inverse to predicted-request errors for supervision; it is excluded from inference. All native training timesteps are 1/60 s despite explicit scalar conditioning.',
            outcome=f'Original training completes {config["updates"]} total declared optimizer updates; final logged main objective {history["history"][-1]["loss"]:.17g}. Complete initial/final tensors, source/recipe/input/output identities and sample/work counts verify. '+('The later coupled generator inherits the exact calibrated flux tensor values; it does not inherit the nonlinear request coefficients. ' if factored else '')+'No new request/pressure GPU benchmark or full-state/own-history drift result is measured in this source review.',
            lesson='Explicit dt input does not establish unseen-timestep generalization when training uses one cadence. A request predictor needs a separately identified pressure component and state update before it becomes a simulator. The factored design changes width, architecture, calibration data and precision as well as freezing flux, so a gain cannot be assigned to a single change. The analytical inverse is a training loss tool; distinguish that from candidate inference and disclose it rather than claiming the whole training method has no inverse.',
            reproduction='Preserve exact captured model, objective, data codec, config, typed recipe/runtime, original F32 velocity/q archive and normalization, private biased-start sampling stream and main Adam schedule. The factored fit also needs its linear teacher, 1,000-update F64 calibration stage and the calibrated frozen flux before main fitting; initial-weights and post-calibration checkpoint-000 represent different states. Original final publication/restore probes execute CPU neural inference and need a GPU-only variant before rerunning. Clean retraining, new-Rheon generation, comparable complete compositions and600-output drift remain unqualified.')
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix), 'Original receipt/config suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural invariant failed')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=2,
        full_training_archive_verification=archive_check,original_note_bindings=notes,new_model_forwards=0,
        native_mutations=0,new_reference_generation=False,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in ['prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Two causal request components reviewed; full64-field archive, main sample stream and inherited81flux values verified.')


if __name__ == '__main__':
    run()
