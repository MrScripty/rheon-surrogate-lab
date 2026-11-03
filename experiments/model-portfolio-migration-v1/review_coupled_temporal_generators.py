"""Review the coupled K2 generator and matched provisional-label continuations."""
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
OUTPUT = DOC/'reports/captured-coupled-temporal-generator-method-reviews-01.json'
IDS = {'fit-950df0d22165f059','fit-1b254e19c19f21cc','fit-a80ef2fb0247b545'}


def verify_experts(context):
    """Tie all deployed frozen tensors to the original supplied learned assets."""
    project, receipt = context['project'],context['receipt']
    assets = project/'expert-assets'
    core = read(assets/'pressure/configuration.json')
    require(core['spatial_parameters'] == 92560 and core['parent_learned_parameters'] == 128938
            and core['steps'] == 2 and core['gains_are_cached_learned_head_outputs']
            and core['parent_entry']['id'] == 'a-tail8', 'Actual learned pressure parent differs')
    for path,sha in core['parent_entry']['identities'].items():
        require(binding(ROOT/path)['sha256'] == sha, 'Retained pressure-parent identity changed')
    with np.load(assets/'pressure/fine-weights.npz',allow_pickle=False) as arrays:
        require(sum(v.size for v in arrays.values()) == 92560, 'Learned spatial asset count differs')
        expected = {}
        for name in arrays.files:
            value = arrays[name]
            expected['pressure_core.source.fine.'+name] = dict(shape=list(value.shape),dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes()).hexdigest())
    with np.load(assets/'request-weights.npz',allow_pickle=False) as arrays:
        value = arrays['flux.weight']
        require(value.shape == (1,3,3,3,3) and value.dtype.str == '<f4', 'Managed81coefficient flux required')
        expected['flux.weight'] = dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(value.tobytes()).hexdigest())
    require(all(receipt['frozen_parameter_inventory'].get(n) == s for n,s in expected.items()),
            'Frozen deployed neural coefficients differ from original assets')
    return expected,core


def run():
    require(not OUTPUT.exists(), 'Preserve original coupled-family review')
    availability_path = DOC/'reports/full-generator-review-input-availability-01.json'
    selected = [r for r in read(availability_path)['rows'] if r['fit_item_id'] in IDS]
    require(len(selected) == 3, 'Coupled parent and both provisional arms required')
    selected.sort(key=lambda r:0 if r['fit_item_id'] == 'fit-950df0d22165f059' else 1)
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {r['item_id']:r for r in read(index_path)['fits']}
    rows, notes, parent, paired = [], [], None, []
    for item in selected:
        continuation = item['original_configuration']['schema'] == 'rheon-temporal-fit-provisional-v1'
        c = audit(item,'temporal_train_provisional.py' if continuation else 'temporal_train_coupled.py')
        project, config, receipt, history = c['project'],c['config'],c['receipt'],c['history']
        require(config['horizons'] == 2 and config['width'] == 12 and config['architecture'] == 'coupled'
                and config['supervision_mode'] == 'increments' and config['feedback_protocol'] == 2
                and config['sampling_seed'] == 2026101922 and config['sample_step_range'] == [0,120]
                and config['physical_context']['dt_s'] == 1/60, 'Actual coupled family differs')
        require(config['updates'] == (4000 if continuation else 10000)
                and config['single_block_updates'] == (0 if continuation else 2000)
                and config['seed'] == (2026102021 if continuation else 2026101921)
                and config['learning_rate'] == (.0003 if continuation else .001)
                and config['schedule'] == ([dict(update=3001,learning_rate=.0001)] if continuation else
                    [dict(update=6001,learning_rate=.0003),dict(update=9001,learning_rate=.0001)]),
                'Original coupled/continuation training budget differs')
        # Independent count from the actual width12 fluid/local/transport/viscous/
        # convex-tracer architecture, plus inherited spatial92560 and flux81.
        trainable = (12*10*27+12)+(12*12+12)+2*(12*3*27)+(39*12+12+12*12+12)
        trainable += (36*12+36)+(3*36)+(3*3*27)+(16*43+16)+(27*16+27)
        expected_frozen,core = verify_experts(c)
        require(receipt['frozen_parameter_inventory'] == expected_frozen
                and c['facts']['frozen_parameters'] == 92641 and c['facts']['trainable_parameters'] == trainable == 7970
                and c['facts']['retained_parameter_count'] == 100611, 'Independent coupled parameter count differs')
        calls = int(c['order'][:,2].sum())
        padded = [math.prod(n+1 for n in p['scene']['counts']) for p in c['manifest']['parents']]
        cells = sum(padded[int(p)]*int(b) for p,_,b in c['order'])
        expected_work = dict(network_forwards=calls,encoder_cells=cells,predicted_physical_states=2*calls,
            encoder_calls=calls,latent_updates=2*calls,pressure_readout_calls=2*calls,flux_readout_calls=2*calls,
            momentum_readout_calls=2*calls,tracer_readout_calls=2*calls,internal_history_updates=calls,
            velocity_feature_calls=4*calls,viscous_readouts=2*calls,local_update_cells=2*cells)
        require(history['optimizer_work'] == expected_work, 'Independent coupled inference-work accounting differs')
        model = (project/'bundle_model_coupled.py').read_text()
        require('for step in range(self.horizons):' in model
                and 'physical_pressure=self.pressure_core(q)' in model
                and 'kernel=-learned_flux_weight' in (project/'tied_momentum_readout.py').read_text()
                and 'with torch.no_grad():' in (project/'full_learned_pressure_map.py').read_text()
                and 'matrix[:,start:start+batch]=self.linear_graph' in (project/'full_learned_pressure_map.py').read_text(),
                'Actual sequential neural flux/pressure/momentum/unit-map route differs')
        if continuation:
            require(parent is not None and receipt['initial_inventory'] == parent['receipt']['final_inventory']
                    == config['continuation_inventory']
                    and binding(project/'continuation-weights.npz')['sha256'] == config['continuation_weights_sha256']
                    == binding(parent['train']/'weights.npz')['sha256'], 'Continuation does not start from exact completed parent')
            source = (project/'temporal_provisional_objective.py').read_text()
            require(config['auxiliary_provisional_weight'] in (0.,1.)
                    and 'return base+self.weight*auxiliary' in source
                    and 'with torch.no_grad():truth=provisional_labels' in source
                    and 'return v-contribution' in source, 'Actual label-only provisional auxiliary differs')
            paired.append(dict(config={k:v for k,v in config.items() if k not in ('identifier','auxiliary_provisional_weight')},
                sources={Path(s['path']).name:s['sha256'] for s in c['facts']['sources']},order_sha256=history['sample_order_sha256']))
        else:
            parent = c
        card = cards[item['fit_item_id']]
        require(not card.get('method_review_status'), 'Do not replace a native published review')
        row = dict(c['facts'],fit_item_id=item['fit_item_id'],note_path=card['card_path'],generator_identifier=config['identifier'],horizons=2,
            actual_optimizer_work=expected_work,optimizer_network_forwards=calls,
            exact_parent_checkpoint_continuation=continuation,auxiliary_provisional_weight=config.get('auxiliary_provisional_weight'),
            constructor_performs_CPU_neural_unit_responses_before_to_device=True,GPU_only_construction_qualified=False,
            fresh_common_full_generator_benchmark_complete=False,
            method='A100,611-parameter coupled full-state composition with7,970trainable and92,641frozen coefficients. One width12fluid encoder is shared across a K2block; each physical step still invokes local transport/viscosity, neural flux, the frozen learned pressure map, tied neural momentum and convex tracer readout. Flux uses81inherited learned coefficients; momentum uses their negative spatial transpose. Pressure inherits the a-tail8spatial network and cached learned modal gains. Small registered grids use F64unit-response compositions;32cubed uses the original frozen component. Output packing passes pressure through F32before F64unpack. These are two actual recurrent advances, not one endpoint plus a free intermediate state.',
            purpose=('Continue the same completed coupled checkpoint with a matched4,000-update fresh Adam experiment, selecting provisional-velocity auxiliary weight'+str(config['auxiliary_provisional_weight'])+'. The extra target is reference velocity minus the learned pressure-momentum contribution, consumed only by the objective.' if continuation else
                'Couple learned provisional velocity, pressure request and momentum readouts through shared pretrained flux coefficients while sharing the fluid encoder across physical steps.'),
            experiment=f'Original16consumed fitting scenes/2064states at8/16cells per metre, four shapes including32cubed, fixed1/60s outputs and2/60s block advancement. Fit performs{config["updates"]}CUDAupdates/{calls}root calls/{2*calls}step readouts. All original source/recipe/input/output identities, expert assets, initial/final/frozen tensors and private sample/work counts verify. '+('Both continuation arms start from the exact10,000-update parent’s final tensors, use identical sources/sample order and fresh Adam; only auxiliary weight differs. Both compute provisional labels even when the selected weight is0.' if continuation else
                'The first2000updates supervise one block and later8000supervise a second reconstructed own-history block. Inherited flux provenance records1000updates within its11000-update request parent; pressure provenance remains separate from this fit’s updates.'),
            outcome=f'Original fit completed; final recorded composite objective {history["history"][-1]["loss"]:.17g}. All frozen pressure/flux coefficients remain byte-identical to their inherited assets. '+('Exact continuation-weight equality proves the checkpoint relationship, without recovering or modifying historical Trial ancestry. ' if continuation else '')+'Fresh common GPU full-state accuracy/timing and600-output drift are pending; loss values that include an auxiliary term are not a fair accuracy ranking.',
            lesson='Sharing the encoder saves some work, but K2still runs the pressure readout twice. Numerical interpolation and physical unit scalings remain explicit parts of the learned composition. Learned-map construction derives coefficients from neural unit queries rather than a numerical inverse, yet the original constructor executes those queries on CPU before .to(device); do not run it unchanged under GPU-only inference policy. Constructor/setup time and matrix memory must be reported separately from warmed complete-block latency.',
            reproduction='Preserve exact captured config/model/objective/typed recipe/runtime, original16scene F32corpus/normalization, step-scales.json and every expert asset including learned gains/bounds/provenance. Continuations also require exact continuation-weights.npz and fresh Adam, not restored optimizer moments. The full learned pressure matrices are nonpersistent derived buffers and must be reconstructed. Qualify CUDA-only constructor unit responses and publication probes before executing old recipes; retain arithmetic differences as an execution variant. Clean retraining, new-Rheon data generation and600-output drift remain unqualified.')
        raw = (PROJECT/card['card_path']).read_bytes()
        original_receipt = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original_receipt) == revised.count(original_receipt) == 1 and revised.endswith(suffix), 'Original receipt/config suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT/card['card_path']))
    require(paired[0] == paired[1], 'Provisional arms differ outside the registered auxiliary weight/identifier')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/s['path']) == s for s in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural review invariant failed')
    value = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=3,
        original_note_bindings=notes,availability=binding(availability_path),new_model_forwards=0,native_mutations=0,
        new_reference_generation=False,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in
            ['captured_temporal_review.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(value)+b'\n')
    print('Coupled parent and both provisional continuations reviewed; exact ancestry bytes/frozen experts verified.')


if __name__ == '__main__':
    run()
