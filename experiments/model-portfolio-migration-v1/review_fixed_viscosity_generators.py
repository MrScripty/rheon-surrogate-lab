"""Review nine seeded/horizon fits with frozen learned viscosity and displacement."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_temporal_review import audit, verified_review_receipt
from prepare_method_review_revisions import revised_card
from review_coupled_temporal_generators import verify_experts
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-fixed-viscosity-generator-method-reviews-01.json'


def run():
    require(not OUTPUT.exists(), 'Preserve original fixed-viscosity family review')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    selected = [r for r in read(availability_path)['rows']
                if r['original_configuration'].get('schema') == 'rheon-fixed-neural-viscosity-fit-v1']
    require(len(selected) == 9, 'All nine original available follow-up fits required')
    selected.sort(key=lambda r:(r['original_configuration']['seed'],r['declared_horizons']))
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {r['item_id']:r for r in read(index_path)['fits']}
    rows, notes, seeded_initial, source_sets = [], [], {}, []
    for item in selected:
        c = audit(item,'temporal_train_fixed_viscosity.py')
        project, config, receipt, history = c['project'],c['config'],c['receipt'],c['history']
        k, seed, updates, single = config['horizons'],config['seed'],config['updates'],config['single_block_updates']
        budgets = {1:(20000,4000,16001),2:(10000,2000,8001),3:(6667,1334,5334),4:(5000,1000,4001)}
        require((updates,single,config['schedule'][0]['update']) == budgets[k]
                and config['learning_rate'] == .001 and len(config['schedule']) == 1
                and config['schedule'][0]['learning_rate'] == .0003
                and config['width'] == 12 and config['architecture'] == 'displacement'
                and config['supervision_mode'] == 'increments' and config['displacement_auxiliary_weight'] == 1.
                and seed in (2026102045,2026102046,2026102047)
                and config['sampling_seed'] == 2026101922+seed-2026102045
                and config['viscosity_coefficients_frozen_from_start']
                and not config['pretrained_nonlinear_request_weights_used'], 'Registered fixed-viscosity budget/representation differs')
        frozen,core = verify_experts(c)
        assets = project/'expert-assets'
        for file,meta,name,shape in [
            ('point-weights.npz','point-initialization.json','displacement.point_weight',(12,3,3,3,3)),
            ('viscosity-weights.npz','viscosity-initialization.json','viscous_head.weight',(3,3,3,3,3))]:
            metadata = read(assets/meta)
            require(metadata['native_qualified_float32'] and metadata['weights_sha256'] == binding(assets/file)['sha256'],
                    'Original native-qualified inherited coefficient identity differs')
            with np.load(assets/file,allow_pickle=False) as arrays:
                require(arrays.files == ['projection.weight'], 'Exact inherited projection asset required')
                value = arrays['projection.weight']
                require(value.shape == shape and value.dtype.str == '<f4', 'Inherited point/viscosity tensor extent differs')
                frozen[name] = dict(shape=list(shape),dtype='<f4',sha256=hashlib.sha256(value.tobytes()).hexdigest())
        # Fluid encoder, retained latent update, local displacement features,
        # context modulation and second-order curvature head. No K-shaped heads.
        trainable = (12*10*27+12)+(12*12+12)+(15*12+12+12*12+12)+(24*3*27)+(36*12+36)+(12*36)
        require(receipt['frozen_parameter_inventory'] == frozen and trainable == receipt['trainable_parameters'] == 6600
                and c['facts']['frozen_parameters'] == 93856 and c['facts']['retained_parameter_count'] == 100456,
                'Independent fixed-viscosity architecture/frozen count differs')
        if seed in seeded_initial:
            require(receipt['initial_inventory'] == seeded_initial[seed], 'Same-seed initial tensors change across K')
        else:
            seeded_initial[seed] = receipt['initial_inventory']
        calls = int(c['order'][:,2].sum())
        padded = [math.prod(n+1 for n in p['scene']['counts']) for p in c['manifest']['parents']]
        cells = sum(padded[int(p)]*int(b) for p,_,b in c['order'])
        work = dict(network_forwards=calls,encoder_cells=cells,predicted_physical_states=calls*k,
            encoder_calls=calls,latent_updates=calls*k,pressure_readout_calls=calls*k,flux_readout_calls=calls*k,
            momentum_readout_calls=calls*k,tracer_readout_calls=calls*k,internal_history_updates=calls*(k-1),
            velocity_feature_calls=calls*k,displacement_readout_calls=calls*k,viscous_readouts=calls*k,local_update_cells=cells*k)
        require(history['optimizer_work'] == work and calls*k == 36000, 'Equal physical-prediction budget/work differs')
        model = (project/'bundle_model_displacement.py').read_text()
        fixed = (project/'bundle_model_fixed_viscosity.py').read_text()
        objective = (project/'temporal_displacement_objective.py').read_text()
        trainer = (project/'temporal_train_fixed_viscosity.py').read_text()
        require('from bundle_model_fixed_viscosity import BundleNet' in trainer
                and 'self.viscous_head.weight.requires_grad_(False)' in fixed
                and 'advected=convex_field_readout(current[:,1:5],shift)' in model
                and 'physical_pressure=self.pressure_core(q)' in model
                and 'training_displacement_teacher' not in model
                and 'labels,_=displacement_labels(physical,counts,alpha.detach().double()/self.network.scales[\'velocity\'])' in objective,
                'Actual frozen viscosity/learned displacement/training-only teacher route differs')
        source_sets.append({str((ROOT/s['path']).relative_to(project)):s['sha256'] for s in c['facts']['sources']})
        card = cards[item['fit_item_id']]
        require(not card.get('method_review_status'), 'Do not replace a published source review')
        row = dict(c['facts'],fit_item_id=item['fit_item_id'],note_path=card['card_path'],generator_identifier=config['identifier'],
            horizons=k,model_seed=seed,actual_optimizer_work=work,optimizer_network_forwards=calls,
            equal_physical_training_prediction_budget=36000,same_seed_initialization_equal_across_available_K=True,
            constructor_performs_CPU_neural_unit_responses_before_to_device=True,GPU_only_construction_qualified=False,
            fresh_common_full_generator_benchmark_complete=False,
            method=f'A100,456-parameter recurrent full-state generator for K={k}:6,600trainable coefficients and93,856frozen spatial-pressure/flux/point/viscosity coefficients. One width12fluid encoder is shared across a pass. Each output updates retained features, predicts four carrier displacements from a fused frozen-point/trainable-local projection with an alpha-squared curvature correction, and reconstructs velocity/tracer through fixed convex trilinear gathers. Frozen learned viscosity then acts on forced velocity, followed by learned flux, the inherited pressure map and tied momentum. Koutputs still require Kpressure/flux/momentum readouts and Ktransport reconstructions; there is no cheap endpoint-only pressure shortcut.',
            purpose=f'Test the fixed learned243-coefficient viscosity calibration within a displacement-based full-state generator, at K={k}and model seed{seed}. Keep the trainable architecture independent of K and balance the number of predicted physical training states across horizons.',
            experiment=f'Original16consumed scenes/2064states,8/16cells per metre and four shapes including32cubed/rectangles, fixed1/60s outputs with K={k}advancing{k}/60s. The fit runs{updates}CUDA Adam updates, first{single}one-block then two own-history blocks, for{calls}encoder calls and36,000physical predictions. Rate0.001drops to0.0003at update{config["schedule"][0]["update"]}. Private sampling seed{config["sampling_seed"]}is shared within this seed group; both model and sampling seeds differ between repetitions. Same-seed initial tensors match exactly across available K. A training-only native-qualified midpoint backtrace supplies detached displacement labels from the actual entering predicted state; target error is scaled by alpha squared. Labels are consumed after prediction and are absent from the inference module. All original receipts/recipes/inputs/outputs, nested assets, tensors and work counters verify.',
            outcome=f'Original training completed{updates}updates; last logged composite objective {history["history"][-1]["loss"]:.17g}. Frozen pressure/flux/point/viscosity tensors remain byte-identical. Independent counts reproduce6,600trainable plus93,856frozen coefficients and36,000physical training predictions. No fresh comparable GPU field accuracy, full-block latency, drift or scientific promotion is established by this review.',
            lesson='Equal predicted-state budgets do not mean equal optimizer-update counts or encoder calls. K1/2/3/4use20,000/10,000/6,667/5,000updates. Repetitions change model initialization and sampling order together, so they do not isolate initialization sensitivity alone. The main encoder is shared, while costly pressure/transport work repeats for every output. Keep fixed readout interpolation, precision conversions, constructor matrix work and training-only numerical teachers explicit rather than treating every Koutput architecture as equivalent compute sharing.',
            reproduction='Retain exact captured config/typed recipe/model/objective/runtime and private seeds/order, the original16scene F32archive/normalization and step-scales.json, all expert assets including frozen point/viscosity calibrations, learned pressure gains/bounds and parent provenance. Preserve the K-specific update/curriculum/rate budget and actual-own-history displacement labels. The old constructor and final publication/restore probes execute CPU neural work; qualify a GPU-only variant before executing them. Derived learned pressure matrices are nonpersistent buffers. Clean retraining, new-Rheon regeneration, fresh common benchmarks and600-output drift remain unqualified.')
        raw = (PROJECT/card['card_path']).read_bytes()
        original_receipt = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original_receipt) == revised.count(original_receipt) == 1 and revised.endswith(suffix), 'Original receipt/config suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/card['card_path']))
    require(all(s == source_sets[0] for s in source_sets[1:]), 'Seed/horizon family changes implementation or expert assets')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/s['path']) == s for s in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural review invariant failed')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=9,
        original_note_bindings=notes,availability=binding(availability_path),new_model_forwards=0,native_mutations=0,
        new_reference_generation=False,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in
            ['captured_temporal_review.py','review_coupled_temporal_generators.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Nine frozen-viscosity fits reviewed;36,000physical predictions each, all nested assets/frozen tensors verified.')


if __name__ == '__main__':
    run()
