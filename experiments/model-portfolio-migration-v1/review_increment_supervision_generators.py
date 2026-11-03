"""Audit the matched full-state versus fixed-increment MSE supervision fits."""
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
OUTPUT = DOC/'reports/captured-increment-supervision-generator-method-reviews-01.json'


def run():
    require(not OUTPUT.exists(), 'Preserve original matched-supervision review')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    selected = [r for r in read(availability_path)['rows']
                if r['original_configuration'].get('schema') == 'rheon-temporal-fit-increment-supervision-v1']
    require(len(selected) == 2, 'Both matched MSE supervision arms required')
    prior_path = DOC/'reports/captured-step-scaled-temporal-generator-method-reviews-01.json'
    prior = read(prior_path)
    base = next(r for r in prior['rows'] if r['architecture'] == 'endpoint')
    require(prior['independent_statistics_verification']['all_horizon_increment_scales_verified'],
            'Independent fixed-statistics audit required')
    for spec in [prior['producer'],*prior['helpers'],base['fixed_statistics']]:
        require(binding(ROOT/spec['path']) == spec, 'Prior scale-audit evidence changed')
    old_receipt = read(ROOT/base['original_receipt']['path'])
    require(binding(ROOT/base['original_receipt']['path']) == base['original_receipt'], 'Prior base receipt changed')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {r['item_id']:r for r in read(index_path)['fits']}
    rows, notes, source_sets, normalized_configs = [], [], [], []
    for item in sorted(selected,key=lambda r:r['original_configuration']['supervision_mode']):
        project = ROOT/item['project_path']
        receipt_path = ROOT/item['receipt']['path']
        require(binding(receipt_path)['sha256'] == item['receipt']['sha256'], 'Original fit receipt changed')
        config, receipt = read(project/'config.json'), read(receipt_path)
        train = receipt_path.parent
        history = read(train/'history.json')
        mode = config['supervision_mode']
        require(mode in ('full_state','increments') and config['architecture'] == 'endpoint'
                and config['horizons'] == 2 and config['width'] == 12 and config['feedback_protocol'] == 2
                and config['updates'] == receipt['optimizer_updates'] == 10000
                and config['single_block_updates'] == 2000 and config['sample_step_range'] == [0,120]
                and config['seed'] == 2026101921 and config['sampling_seed'] == 2026101922
                and config['learning_rate'] == .001 and config['schedule'] == [dict(update=6001,learning_rate=.0003),dict(update=9001,learning_rate=.0001)]
                and config['physical_context']['dt_s'] == 1/60 and receipt['mode'] == 'train' and receipt['status'] == 'complete',
                'Original matched architecture/budget/cadence differs')
        require(binding(project/'config.json')['sha256'] == receipt['closure']['config_sha256'], 'Config changed')
        sources = [binding(project/n) for n in sorted(receipt['closure']['sources'])]
        require(all(s['sha256'] == receipt['closure']['sources'][Path(s['path']).name] for s in sources),
                'Executed source closure changed')
        source_sets.append({Path(s['path']).name:s['sha256'] for s in sources})
        normalized_configs.append({k:v for k,v in config.items() if k not in ('supervision_mode','identifier','loss')})
        manifest_project, recipe = read(project/'project.json'), read(project/'train-recipe.json')
        require(recipe['entrypoint'] == str((project/'temporal_train_increment_supervision.py').relative_to(ROOT)),
                'Actual matched trainer differs')
        for spec in [*manifest_project['recipes'],*recipe['sources']]:
            require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'Typed recipe input changed')
        inputs = {r['id']:ROOT/r['path'] for r in manifest_project['data']}
        require(receipt['closure']['inputs'] == old_receipt['closure']['inputs'], 'Matched fitting corpus differs')
        bound_inputs = []
        for slot,expected in receipt['closure']['inputs'].items():
            spec = binding(inputs[slot])
            require(dict(sha256=spec['sha256'],bytes=inputs[slot].stat().st_size) == expected, 'Fitting input changed')
            bound_inputs.append(dict(spec,slot=slot,bytes=expected['bytes']))
        statistics_path = project/'step-scales.json'
        require(binding(statistics_path)['sha256'] == config['step_scale_statistics_sha256'] == base['fixed_statistics']['sha256']
                and config['original_normalization_sha256'] == binding(inputs['normalization'])['sha256'],
                'Previously independently audited fixed statistics/normalization changed')
        for name,expected in receipt['outputs'].items():
            p = train/name
            require(dict(sha256=binding(p)['sha256'],bytes=p.stat().st_size) == expected, 'Original fit output changed')
        require(receipt['initial_inventory'] == old_receipt['initial_inventory'], 'Matched arm changed seeded initialization')
        for filename,key in [('initial-weights.npz','initial_inventory'),('weights.npz','final_inventory')]:
            inventory = receipt[key]
            require(inventory == history[key], 'Receipt/history tensor inventory differs')
            with np.load(train/filename,allow_pickle=False) as arrays:
                require(set(arrays.files) == set(inventory), 'Complete retained tensor inventory required')
                for name,spec in inventory.items():
                    value = arrays[name]
                    require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                            and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'], 'Parameter bytes/type/shape changed')
        count = sum(math.prod(s['shape']) for s in receipt['final_inventory'].values())
        require(count == receipt['parameters'] == base['retained_parameter_count'] == 163068,
                'Matched arms changed parameter count')
        rng = np.random.Generator(np.random.PCG64(config['sampling_seed']))
        order = np.asarray([(int(rng.integers(16)),int(rng.integers(121)),1 if u <= 2000 else 2) for u in range(1,10001)],dtype='<i8')
        require(hashlib.sha256(order.tobytes()).hexdigest() == history['sample_order_sha256']
                and history['optimizer_work'] == base['actual_optimizer_work'] and history['valid_parent_count'] == 16
                and history['train_only'], 'Matched sample/order/work differs')
        previous_sources = {Path(s['path']).name:s['sha256'] for s in base['sources']}
        require(all(binding(project/n)['sha256'] == previous_sources[n] for n in
                ['bundle_model_step_scaled.py','temporal_features_v2.py','temporal_feedback.py','temporal_objective.py','learned_global.py']),
                'Scaling architecture, feature/feedback or shared physical metrics changed')
        objective = (project/'temporal_objective_increment.py').read_text()
        trainer = (project/'temporal_train_increment_supervision.py').read_text()
        require("from temporal_objective_increment import loss as supervised_loss" in trainer
                and "return partial(supervised_loss,increment_scales=network.increment_scales,mode=mode)" in trainer
                and "values['velocity']=velocity_error/squared_scales('velocity')" in objective
                and "values['tracer']=tracer_error/squared_scales('tracer')" in objective
                and "difference=(predicted[:,:,component]-target[:,:,component])" in objective
                and 'log1p' not in objective,
                'Actual MSE objective/unchanged full-state-error numerator differs')
        card = cards[item['fit_item_id']]
        require(not card.get('method_review_status'), 'Do not replace a native published review')
        row = dict(fit_item_id=item['fit_item_id'],note_path=card['card_path'],generator_identifier=config['identifier'],
            supervision_mode=mode,architecture='endpoint',horizons=2,retained_parameter_count=count,
            original_receipt=binding(receipt_path),history=binding(train/'history.json'),sources=sources,
            original_configuration=binding(project/'config.json'),original_recipe=binding(project/'train-recipe.json'),
            training_inputs=bound_inputs,training_manifest=binding(inputs['manifest']),fixed_statistics=binding(statistics_path),
            exact_seeded_initialization_matches_scaled_endpoint=True,paired_fitting_inputs_order_and_budget_equal=True,
            actual_train_trajectories=16,available_train_states=2064,physical_output_dt_s=1/60,block_physical_advance_s=2/60,
            optimizer_network_forwards=18000,actual_optimizer_work=history['optimizer_work'],
            fresh_common_full_generator_benchmark_complete=False,
            method='A163,068-parameter F32width12K2endpoint-conditioned generator with one shared spatial/global encoder, fresh predicted endpoint and joint intermediate decoder. It retains the audited fixed train-only force/increment scaling, excludes previous pressure, and predicts all five fields without a numerical pressure solver. Its model implementation, initial tensors and deployed feedback match the earlier scaled endpoint; only the fitted objective changes. '+('Velocity/tracer use the existing full-state relative squared errors.' if mode == 'full_state' else 'Velocity/tracer use mean squared full prediction error divided by fixed aggregate training increment RMS squared at each horizon. The numerator remains predicted full state minus target full state, including accumulated state error; it does not replace reference history or supervise only a detached local delta.'),
            purpose='Use a matched MSE pair to isolate whether supervising velocity/tracer in typical-step units improves predictive increments when full-state field magnitude can hide small changes. A separate full-state MSE arm controls the removal of the earlier logarithmic robust objective.',
            experiment='Both original fits use the same16train-only scenes/2064states,8/16cells per metre, four shapes including32cubed/rectangles, fixed1/60s outputs, and2/60s per block. Original source/typed recipe/input/output closures, all initial/final tensors, exact seeded initialization and PCG64sample stream verify. Both train10,000CUDAupdates with identical learning-rate schedule and2000one-block then8000two-own-history-block curriculum,18,000encoder calls. Both replace prior log1p-scaled pressure/equation/field/physical terms with MSE terms; between these two arms only velocity/tracer denominators differ. Pressure/equation and divergence/mass/energy/bounds coefficients remain common to the pair.',
            outcome=f'Original training completed10,000updates; final recorded composite MSE objective {history["history"][-1]["loss"]:.17g}. The two arms preserve exact untrained parameters and source code but fit different final tensors under their selected denominators. Fresh common full-field GPU scores, block latency and own-history drift remain pending. Loss values across these modes or the earlier logarithmic objective are not directly comparable performance metrics.',
            lesson='Comparing the two MSE arms isolates the choice of field-error scale. Comparing either arm directly with the earlier step-scaled fit changes the robust objective as well, so an observed gain cannot be assigned solely to increment supervision. Fixed increment units do not supply variable timestep conditioning or longer temporal training, nor guarantee accurate intermediate states.',
            reproduction='Use the exact captured config/typed recipe/runtime/model and MSE objective source, the original16scene F32archive/normalization, fixed step-scales.json and selected full_state/increments mode. Retain common private seeds/order, schedule and deployed own-history reconstruction. Original training publication/restore uses CPU neural probes; qualify a GPU-only recipe before retraining. New-Rheon regeneration, clean retraining and600-output drift remain unqualified.')
        raw = (PROJECT/card['card_path']).read_bytes()
        original_receipt = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original_receipt) == revised.count(original_receipt) == 1 and revised.endswith(suffix),
                'Original receipt/config suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/card['card_path']))
    require(source_sets[0] == source_sets[1] and normalized_configs[0] == normalized_configs[1],
            'The paired fit changes more than selected supervision mode/identifier/loss description')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/s['path']) == s for s in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural review invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=2,
        new_model_forwards=0,native_mutations=0,new_reference_generation=False,clean_reproduction_qualified=False,
        original_note_bindings=notes,prior_statistics_review=binding(prior_path),availability=binding(availability_path),
        producer=binding(Path(__file__)),helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1'/n) for n in
            ['prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Two matched MSE supervision fits reviewed; fixed history/error semantics and exact control identities verified.')


if __name__ == '__main__':
    run()
