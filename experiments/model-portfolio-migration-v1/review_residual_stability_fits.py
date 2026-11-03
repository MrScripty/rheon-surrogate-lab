"""Review C16--C25 operator learning without running a neural model or solver."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from captured_pressure_review import load_fit
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_water_learning_sequence import binding, canonical, population_identity, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-residual-stability-method-reviews-01.json'


def check_feedback(facts):
    config, history = facts['config'], facts['history']
    declared = config.get('analytical_feedback')
    saved = history.get('analytical_feedback')
    if not declared or declared['weight'] == 0:
        require(saved is None, 'Disabled manufactured feedback unexpectedly executed')
        return dict(actual_manufactured_patterns=0,actual_manufactured_examples=0)
    require(declared == dict(patterns=32,seed=2026101507,weight=.1)
            and saved['shape_zyx'] == [16,16,16] and len(saved['samples']) == 128
            and saved['additional_examples'] == config['training']['updates'] == 5000,
            'Actual manufactured bank/budget differs')
    # Reproduce descriptor draws and sampling order without generating fields or
    # importing Torch. The original fitted correction amplitude remains bound
    # to its retained history, rather than reinterpreting reduction roundoff.
    rng = np.random.Generator(np.random.PCG64(declared['seed']))
    fixed = [(0,0,1),(0,1,0),(1,0,0),(0,1,2),(3,2,1),(4,4,4),(0,0,15),(15,15,15)]
    amplitude = -saved['samples'][0]['amplitude_pa']/2
    descriptions = []
    for index in range(32):
        if index < 8:
            components = [(fixed[index],1.)]
        else:
            components = []
            for _ in range(1 if index%2 else 3):
                frequencies = tuple(int(rng.integers(16)) for _ in range(3))
                if not any(frequencies):
                    frequencies = (0,0,1)
                components.append((frequencies,float(rng.uniform(.5,1.5))))
        for multiplier in (-2.,-.5,.5,2.):
            descriptions.append(dict(components_zyx=components,amplitude_pa=amplitude*multiplier))
    order = rng.integers(128,size=5000,dtype=np.int64)
    require(canonical(descriptions) == canonical(saved['samples'])
            and hashlib.sha256(canonical(descriptions)).hexdigest() == saved['bank_description_sha256']
            and hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() == saved['sampling_order_sha256']
            and saved['loss_weights'] == dict(pressure=.25,gradient=.25,equation=.5,amplification=1.),
            'Manufactured frequencies/amplitudes/order/objective differ')
    return dict(actual_manufactured_patterns=32,actual_manufactured_examples=128,
        additional_manufactured_optimizer_proposals=5000,manufactured_description_and_order_rederived=True,
        original_correction_amplitude_pa=amplitude)


def run():
    require(not OUTPUT.exists(), 'Preserve original source-review report')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    index = json.loads(index_raw)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {e['id']:e for e in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score = read(DOC/'reports/pressure-common-score-01.json')['models']
    timing_path = DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    timing = read(timing_path)
    require(timing['status'] == 'all127_named_pressure_models_two_pass_GPU_timings_and_exact_accuracy_join_verified',
            'Qualified common-boundary timing required')
    purposes = {
        'C16':'Replace seven feature channels with a physically scaled pressure-equation residual, retaining nonlinear SiLU corrections.',
        'C17':'Make the correction network linear by removing SiLU and all convolution biases, matching the linear sealed pressure operator.',
        'C18':'Train four shared-weight pressure corrections per request, recomputing the equation residual between corrections.',
        'C19':'Reweight training toward the required pressure correction, its spatial gradients, equation error and worst residual.',
        'C20':'Provide a 5,000-update single-correction control for the recurrent model, with the same shorter learning-rate schedule.',
        'C21':'Add manufactured Neumann pressure modes and mixtures, including a penalty for residual amplification, to four-correction training.',
        'C22':'Scale learned corrections by the training correction RMS rather than previous-pressure RMS, without manufactured feedback.',
        'C23':'Combine correction-RMS output scaling with manufactured-mode feedback during four-correction training.',
        'C24':'Test whether manufactured-mode training can retain accuracy with one correction per pressure request.',
        'C25':'Test a single-correction model with correction-RMS scaling, manufactured feedback and a ten-times-lower learning-rate schedule.'}
    lessons = {
        'C16':'Removing irrelevant conditioning simplifies the pressure map but does not by itself resolve common worst errors. This is still fitting one physical frame; broad validation does not enlarge the optimizer scene population.',
        'C17':'The actual linear transform has 19 weight tensors and no biases or SiLU, despite inherited activation=silu metadata. Common worst pressure error improves versus C16, while worst equation error changes little. Preserve executed structure separately from inherited labels.',
        'C18':'Four learned corrections improve the common equation error relative to the matched 5,000-update single-correction control C20. They cost more work within one pressure request; they are not four simulation steps. Prepared-field strategy also differs, so do not attribute all differences to recurrence alone.',
        'C19':'The physical correction/gradient objective changes the meaning of logged training and validation loss. Its common pressure/equation errors trade off against C17 rather than establishing an overall gain. Pressure-equation diagnostics do not establish velocity/tracer drift.',
        'C20':'This is the short-budget control for C18, not merely a renamed C17 checkpoint. Its first 5,000 physical sample selections match the longer fit, but its learning-rate changes occur at different updates.',
        'C21':'Synthetic operator examples materially improve common errors despite using only one CFD frame. The 128 manufactured fields are 32 patterns at four signed amplitudes on one 16-cubed grid; they are not 128 new simulations or evidence of large-grid physical training.',
        'C22':'Output scale changes improve common pressure/equation errors over the prior four-correction baseline without adding scenes. The configuration declares 32 feedback patterns but weight zero; the saved history confirms no manufactured bank executed.',
        'C23':'The combination gives much smaller common errors than the earlier four-correction single-frame baseline. This motivates operator-diverse rehearsal, while retaining the limits of one CFD frame, one manufactured shape and consumed evaluation cases. A larger logged training objective includes extra terms and is not an accuracy regression.',
        'C24':'Single-correction deployment reduces internal work but loses common accuracy relative to C21. Every call still returns one pressure proposal; this result says nothing about multi-frame simulation generation.',
        'C25':'The single-correction and lower-learning-rate changes do not retain C23 accuracy. More than one factor changes, so this is not an isolated causal test of learning rate or correction count.'}
    rows, notes, inventories, populations, orders = [], [], {}, {}, {}
    for number in range(16,26):
        label = f'C{number:02}'
        facts = load_fit(label,work=work,material=material,index=index,score=score)
        meta, config = facts['metadata'], facts['config']
        require((meta['actual_train_trajectories'],meta['actual_train_frames'],meta['actual_validation_trajectories'],meta['actual_validation_frames'])
                == (1,1,64,128), 'Actual physical training/validation population differs')
        train = [p for p in facts['split']['parents'] if p['entry']['split'] == 'train'][0]
        require(train['counts_xyz'] == [16,16,16] and train['cell_size_m'] == .125 and train['frames'][0]['index'] == 15
                and all(p['entry']['dt_s'] == 1/60 for p in facts['split']['parents']), 'Physical scene/density/cadence differs')
        corrections = config['correction']['steps']
        nonlinear = config['correction']['activation'] == 'silu'
        require(config['normalization'] == 'none' and config['conditioning'] == 'physical-residual'
                and corrections == (4 if number in (18,21,22,23) else 1)
                and nonlinear == (number == 16), 'Actual pressure policy differs')
        require(meta['actual_parameter_count'] == (370433 if nonlinear else 370016)
                and meta['initialized_tensor_count'] == (38 if nonlinear else 19), 'Actual tensor inventory differs')
        if not nonlinear:
            require(all(name.endswith('.weight') for name in facts['inventory']), 'Linear model retained trainable biases')
        project = facts['project']
        physical = (project/'physical_residual.py').read_text()
        require("q=residual*(rho/h).reshape(-1,1,1,1,1)" in physical
                and "module.register_parameter('bias',None)" in physical
                and 'if isinstance(layer,nn.SiLU):block.layers[index]=nn.Identity()' in physical,
                'Actual physical-input/linearity transform differs')
        require(config['training']['seed'] == 20261015 and config['training']['batch_size'] == 1
                and config['training']['sampling_seed'] == 2026102501, 'Captured optimizer setup differs')
        feedback = check_feedback(facts)
        correction_scale = facts['normalization'].get('output_scale') == 'correction_pressure'
        require(correction_scale == (number in (22,23,25)), 'Executed output scaling differs')
        if number == 19:
            require(config['objective'] == 'correction-gradient-divergence'
                    and "return .25*correction_error+.25*gradient_error/gradient_norm.clamp_min(floor)+.4*equation+.1*maximum" in physical,
                    'Physical loss weights differ')
        inventories[label], populations[label], orders[label] = facts['inventory'], population_identity(facts['split']['parents']), facts['sample_order']
        count, budget = meta['actual_parameter_count'], config['training']['updates']
        method = (f'A {count:,}-parameter multiscale 3D CNN learns a pressure correction from one channel: '
            'q=(rho/h)*equation residual, divided by its train-only RMS. Grid shape supplies the sealed cubic Neumann geometry; '
            'density and spacing act through the physical rescaling. Timestep is validated but is not a separate neural channel. '
            'The F32 core has encoder widths 8/16/24/32, bottleneck 48 and decoder widths 32/24/16/8, with no GroupNorm. ')
        method += ('SiLU and convolution biases remain.' if nonlinear else 'SiLU and every convolution bias are removed; inherited activation=silu metadata does not describe those executed layers.')
        method += (f' The configured policy allows {corrections} correction(s) per pressure request, recomputing the F64 equation residual between passes. '
            'The F32 correction is scaled, cell-zero gauged and added to supplied pressure with F64 arithmetic. '
            'This is not a simulation advance and does not accept independent obstacle/free-surface masks.')
        training = (f'{budget:,} whole-network Adam updates, batch 1, initial rate {config["training"]["learning_rate"]:g}, '
            f'schedule {config["training"]["schedule"]}, seed 20261015 and clipping at norm 1.0. '
            'One physical training frame at native index 15, 16x16x16 cells and 8 cells/metre; '
            '64 validation trajectories provide 128 frames at 8/16 cells per metre. Physical timestep is 1/60 s. ')
        training += ('The loss weights correction error 0.25, correction-gradient error 0.25, equation error 0.4 and worst residual 0.1. '
                     if number == 19 else 'The physical-frame loss equally weights normalized pressure-field and equation errors. ')
        training += ('Output uses the train correction RMS (2.1793165157463443 Pa). ' if correction_scale else
                     'Output uses the train previous-pressure RMS (26.41425532430595 Pa). ')
        training += ('Additional feedback uses 32 manufactured cosine patterns/mixtures at four signed amplitudes, giving 128 fields and 5,000 sampled extra proposals at weight 0.1. '
            'Its pressure/gradient/equation weights are 0.25/0.25/0.5 plus a residual-amplification penalty. These are operator examples, not CFD scenes. '
            if feedback['actual_manufactured_examples'] else 'No manufactured feedback executes. ')
        h, common, perf = facts['history'], score[label]['worst'], timing['models'][label]['summary']
        outcome = (f'Last logged training objective {h["history"][-1]["loss"]:.17g}; retained final validation objective {h["validation"]["final"]:.17g}. '
            f'Common 128-request worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}; '
            f'equation relative L2 {common["common_equation_relative_l2"]:.17g}. '
            f'Complete GPU pressure-call timing: median of case medians {perf["median_case_median_ms"]:.17g} ms, '
            f'worst case median {perf["worst_case_median_ms"]:.17g} ms. The boundary includes native CPU arrays/packing/transfers/output/synchronization; '
            'setup and scoring are excluded. This is not per-simulation-step latency or an own-history drift result.')
        row = dict(meta,method=method,purpose=purposes[label],experiment=training,outcome=outcome,lesson=lessons[label],
            reproduction='Use the original typed train recipe/config/runtime, full source witness, bound input slots, preprocessing and seeded initial inventory. '
                'Manufactured feedback is regenerated by the captured analytical.py from the training frame shape/scalars/correction amplitude and its private RNG seed. '
                'Historical training ends with CPU deployment/parity forwards; a clean GPU-only reproducer must first be qualified under the current policy. '
                'Keep old-Rheon data separate from new-solver captures. No clean retraining, optimizer resume or new-Rheon regeneration is qualified by this review.',
            retained_timing=binding(timing_path),configured_pressure_corrections=corrections,
            actual_train_cells_per_metre=[8],actual_validation_cells_per_metre=[8,16],**feedback)
        original = verified_review_receipt(row)
        note = PROJECT/meta['note_path']
        raw = note.read_bytes()
        revised, suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix), 'Original receipt/configuration suffix changed')
        notes.append(binding(note))
        rows.append(row)
    seed_weights = {k:v for k,v in inventories['C16'].items() if k.endswith('.weight')}
    require(all({k:v for k,v in value.items() if k.endswith('.weight')} == seed_weights for value in inventories.values()),
            'Claimed common surviving seeded weight tensors differ')
    require(all(pop == populations['C16'] for pop in populations.values())
            and np.array_equal(orders['C18'],orders['C20'])
            and all(np.array_equal(order,orders['C16'][:len(order)]) for order in orders.values()),
            'Claimed common physical population or sample-order prefixes differ')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes), 'Native authored notes/index changed during review')
    require('torch' not in sys.modules, 'Retrospective review imported a neural runtime')
    report = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        shared_physical_populations_and_surviving_seed_weights_exact=True,physical_sample_order_prefixes_exact=True,
        source_reviewed_fits=10,original_note_bindings=notes,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        clean_reproduction_qualified=False,
        overview_addition='\n\n## Physical-residual and manufactured-operator experiments\n\nC16–C25 still fit one physical simulation frame. The family removes extraneous neural conditioning, distinguishes actual linear transforms from inherited SiLU labels, tests one versus four pressure corrections, and adds manufactured Neumann modes/output scaling. C23 improves common pressure error substantially without adding CFD scenes. Manufactured fields broaden operator directions; they do not establish scene, large-grid or timestep diversity. Correction passes are internal pressure work, not generated simulation frames. Common GPU timings now have an explicit CPU-input/output boundary.\n',
        producer=binding(Path(__file__)),helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1/captured_pressure_review.py'),
            binding(ROOT/'experiments/model-portfolio-migration-v1/review_water_learning_sequence.py')])
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Ten residual/stability fits source-reviewed; no neural execution or native mutation.',flush=True)


if __name__ == '__main__':
    run()
