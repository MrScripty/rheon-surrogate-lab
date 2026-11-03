"""Review pressure-expert, lead/gauge/tracer decoder and displacement prototypes."""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_temporal_review import audit, verified_review_receipt
from prepare_method_review_revisions import revised_card
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-expert-readout-generator-method-reviews-01.json'
IDS = ['fit-1031d32821fb88cf','fit-0a92490241d21911','fit-273889e7a0201f42']


def run():
    require(not OUTPUT.exists(), 'Preserve original expert/readout review')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    available = {r['fit_item_id']:r for r in read(availability_path)['rows']}
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {r['item_id']:r for r in read(index_path)['fits']}
    rows, notes, initial_expert = [], [], None
    for fid in IDS:
        displacement = fid == IDS[2]
        decoder = fid == IDS[1]
        trainer_name = 'temporal_train_displacement.py' if displacement else ('temporal_train_decoder.py' if decoder else 'temporal_train_pressure_expert.py')
        c = audit(available[fid],trainer_name)
        project, config, receipt, history = c['project'],c['config'],c['receipt'],c['history']
        require(config['horizons'] == 2 and config['updates'] == 10000 and config['single_block_updates'] == 2000
                and config['sampling_seed'] == 2026101922 and config['learning_rate'] == .001
                and config['supervision_mode'] == 'increments' and config['feedback_protocol'] == 2
                and config['seed'] == (2026102045 if displacement else 2026101921)
                and config['width'] == (12 if displacement else 16)
                and config['architecture'] == ('displacement' if displacement else 'endpoint')
                and config['schedule'] == ([dict(update=8001,learning_rate=.0003)] if displacement else
                    [dict(update=6001,learning_rate=.0003),dict(update=9001,learning_rate=.0001)]), 'Original readout budget/architecture differs')
        assets = project/'expert-assets'
        core = read(assets/'pressure/configuration.json')
        require(core['spatial_parameters'] == 92560 and core['parent_learned_parameters'] == 128938
                and core['steps'] == 2 and core['parent_entry']['id'] == 'a-tail8', 'Original learned pressure expert differs')
        frozen = {}
        with np.load(assets/'pressure/fine-weights.npz',allow_pickle=False) as arrays:
            for name in arrays.files:
                value = arrays[name]
                prefix = 'pressure_core.source.fine.' if displacement else 'pressure_core.fine.'
                frozen[prefix+name] = dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(value.tobytes()).hexdigest())
        with np.load(assets/'request-weights.npz',allow_pickle=False) as arrays:
            value = arrays['flux.weight']
            name = 'flux.weight' if displacement else 'request.flux.weight'
            frozen[name] = dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(value.tobytes()).hexdigest())
            if not displacement:
                require(all(receipt['initial_inventory']['request.'+n] == dict(shape=list(arrays[n].shape),dtype=arrays[n].dtype.str,
                    sha256=hashlib.sha256(arrays[n].tobytes()).hexdigest()) for n in arrays.files), 'Causal request initialization differs')
        if displacement:
            with np.load(assets/'point-weights.npz',allow_pickle=False) as arrays:
                value = arrays['projection.weight']
                require(value.shape == (12,3,3,3,3) and value.dtype.str == '<f4', 'Original frozen point projection differs')
                frozen['displacement.point_weight'] = dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(value.tobytes()).hexdigest())
            trainable = (12*10*27+12)+(12*12+12)+(15*12+12+12*12+12)+(24*3*27)+(36*12+36)+(12*36)+(3*3*27)
        else:
            common = 2*(16*3*27)+16*80+16*27+(16*16*27+16)+(16*16+16)+16*27+3*27+3*16+(16*26+16)+16+3*16
            tracer = ((16*28+16)+(27*16+27)+(16*29+16)+(54*16+54)) if decoder else ((8*29*27+8)+8+(8*19+8)+8)
            trainable = common+tracer
        require(receipt['frozen_parameter_inventory'] == frozen and receipt['trainable_parameters'] == trainable
                == (6843 if displacement else (14882 if decoder else 19009)), 'Independent trainable/frozen architecture count differs')
        if initial_expert is None:
            initial_expert = receipt['initial_inventory']
        elif decoder:
            common_keys = [n for n in initial_expert if not n.startswith(('tracer.','tracer_head.','middle_tracer.','middle_t.'))]
            require(all(receipt['initial_inventory'].get(n) == initial_expert[n] for n in common_keys),
                    'Decoder changes seeded common fluid/request/pressure tensors')
        calls = int(c['order'][:,2].sum())
        padded = [math.prod(n+1 for n in p['scene']['counts']) for p in c['manifest']['parents']]
        cells = sum(padded[int(p)]*int(b) for p,_,b in c['order'])
        if displacement:
            work = dict(network_forwards=calls,encoder_cells=cells,predicted_physical_states=2*calls,encoder_calls=calls,
                latent_updates=2*calls,pressure_readout_calls=2*calls,flux_readout_calls=2*calls,momentum_readout_calls=2*calls,
                tracer_readout_calls=2*calls,internal_history_updates=calls,velocity_feature_calls=2*calls,
                displacement_readout_calls=2*calls,viscous_readouts=2*calls,local_update_cells=2*cells)
        else:
            work = dict(network_forwards=calls,encoder_cells=cells,predicted_physical_states=2*calls,encoder_calls=calls,
                latent_updates=0,endpoint_predictions=calls,intermediate_states=calls)
        require(calls == 18000 and history['optimizer_work'] == work
                and receipt['pressure_expert_calls_per_block'] == (2 if displacement else 1), 'Actual root/internal/pressure work differs')
        module_name = 'bundle_model_displacement.py' if displacement else ('bundle_model_decoder.py' if decoder else 'bundle_model_pressure_expert.py')
        source = (project/module_name).read_text()
        require('physical_pressure=self.pressure_core(q)' in source if displacement else
                ('pressure=self.pressure_core(q.double()*self.q_scale)' in source), 'Actual learned pressure route differs')
        if decoder:
            require('horizon*dt/(rho*h)' in source and 'weight=weight-weight.mean' in source
                    and 'weights=logits.double().softmax(dim=1)' in source, 'Combined lead/gauge/convex-tracer changes differ')
        elif displacement:
            require('advected=convex_field_readout(current[:,1:5],shift)' in source
                    and 'training_displacement_teacher' not in source
                    and config['displacement_auxiliary_weight'] == 1., 'Actual learned displacement/teacher separation differs')
            # This parent learns viscosity; later fixed-viscosity fits retain the
            # same tensor shape but freeze the calibrated asset instead.
            require('viscous_head.weight' not in frozen and receipt['initial_inventory']['viscous_head.weight']['shape'] == [3,3,3,3,3],
                    'Original displacement viscosity must remain trainable')
        else:
            require('q=self.request(state[:,1:4],force[:,anchor-1],scalars)' in source
                    and 'self.tracer=nn.Sequential' in source, 'Actual learned request/additive tracer route differs')
        card = cards[fid]
        require(not card.get('method_review_status'), 'Do not replace a published review')
        method = ('A learned-displacement full-state composition with100,456parameters:6,843trainable and93,613frozen. A shared width12encoder precedes K2recurrent neural displacement/transport, learned trainable viscosity, frozen flux, learned pressure and tied momentum. Point coefficients are frozen; a local quadratic/curvature correction predicts shifts used by a fixed convex trilinear carrier readout. Both outputs run pressure separately; pressure is downcast into the F32packed state.' if displacement else
            f'A width16K2endpoint-conditioned generator with{receipt["parameters"]:,}parameters,{trainable:,}trainable and92,641frozen. A causal neural request plus learned correction feeds one frozen a-tail8pressure expert per block. The predicted endpoint and shared features condition cheaper intermediate pressure/velocity/tracer heads. Previous pressure and tracer do not feed the fluid encoder. Pressure-expert accuracy is for its supplied predicted request; it does not certify that the request or future state is correct. '+('This decoder multiplies pressure-to-velocity units by the physical lead, centers pressure, constrains pressure-convolution DCresponse to zero and replaces additive tracer updates with fluid-conditioned convex donor attention. It changes tracer parameters and initial tracer behavior together.' if decoder else 'Velocity uses a learned pressure convolution plus direct known-force accumulation and a learned increment. Tracer uses additive learned updates with multiplicative fluid/tracer features. Pressure-to-velocity scaling uses single dtin every horizon.'))
        row = dict(c['facts'],fit_item_id=fid,note_path=card['card_path'],generator_identifier=config['identifier'],horizons=2,
            actual_optimizer_work=work,optimizer_network_forwards=calls,pressure_expert_calls_per_block=2 if displacement else 1,
            constructor_performs_CPU_neural_unit_responses_before_to_device=True,GPU_only_construction_qualified=False,
            fresh_common_full_generator_benchmark_complete=False,method=method,
            purpose=('Introduce supervised neural displacement readouts and a shared carrier reconstruction while retaining learned pressure/flux/momentum components; provide the trainable-viscosity predecessor to the later frozen-viscosity family.' if displacement else
                'Test the combined lead-time, pressure-gauge and bounded passive-tracer representation against the previous pressure-expert endpoint architecture.' if decoder else
                'Reuse the more accurate frozen pressure component once per multistep block while learning causal future requests and smaller intermediate-state readouts.'),
            experiment='Original16consumed training scenes/2064states at8/16cells per metre and four shapes including32cubed/rectangles, fixed1/60s output cadence and2/60s block advancement.10,000CUDA Adam updates use batch1, private seeds/order and2000one-block then8000two-own-history-block curriculum. All original nested assets/source/typed recipe/input/output identities, complete initial/final/frozen tensors and work counters verify. '+('The displacement objective adds detached midpoint-backtrace labels from actual entering predicted states in alpha-squared units. The teacher is training-only. Its later learning-rate drop is8001; the endpoint fits drop at6001and9001.' if displacement else
                    'The endpoint fits use the fixed-increment MSE objective. The decoder preserves all common seeded fluid/request/pressure tensors but changes its tracer architecture/function, so it is a bundled representation comparison.'),
            outcome=f'Original training completes10,000updates; final composite objective {history["history"][-1]["loss"]:.17g}. Independent architecture sums reproduce all parameter counts and frozen coefficients remain exact. No fresh common GPU full-field error, complete block latency, independent holdout or600-output drift result is measured in this review.',
            lesson=('Learned displacement plus fixed interpolation can preserve a local carrier range, but that is not an end-to-end accuracy or drift guarantee. Training’s numerical teacher does not enter deployed inference. Freezing viscosity in later fits changes the trainable coefficient set and calibration, not this parent’s total parameter count.' if displacement else
                'The decoder combines multiple representation changes; its outcome cannot identify one cause without ablations. Convex tracer donors give local range bounds, not automatic global mass conservation. F64softmax/pressure work still passes through F32state interchange.' if decoder else
                'A high-accuracy pressure component can coexist with inaccurate full-state generation when the predicted request or intermediate decoder is wrong. Benchmark every field/lead against the reference, and distinguish one expert call per block from Kfully recurrent expert calls.'),
            reproduction='Retain exact captured config/typed recipe/runtime/objective and all expert assets, including learned pressure gains/provenance, causal request initialization, fixed statistics and original16scene F32corpus/normalization. Displacement additionally needs frozen point calibration and training-only own-history teacher semantics. Original constructors generate neural unit-response matrices on CPU before .to(device), and publication/restore executes CPU neural probes. Qualify an explicit CUDA-only setup/execution variant before running these recipes. Preserve precision/rounding differences visibly; clean retraining, new-Rheon generation and600-output drift remain unqualified.')
        raw = (PROJECT/card['card_path']).read_bytes()
        original_receipt = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original_receipt) == revised.count(original_receipt) == 1 and revised.endswith(suffix), 'Original receipt/config suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/card['card_path']))
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/s['path']) == s for s in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural review invariant failed')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=3,
        original_note_bindings=notes,availability=binding(availability_path),new_model_forwards=0,native_mutations=0,
        new_reference_generation=False,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in
            ['captured_temporal_review.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Three expert/decoder/displacement fits reviewed; true per-block pressure calls and precision boundaries retained.')


if __name__ == '__main__':
    run()
