"""Review C26--C36 real-scene diversity and manufactured operator coverage.

Reads original training closures and independently rederives private RNG draws.
Does not import Torch, execute a network, simulate fluid or edit native notes.
"""
import hashlib
import math
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
OUTPUT = DOC/'reports/captured-multigrid-recurrence-method-reviews-01.json'


def check_bank(facts):
    declared = facts['config']['analytical_feedback']
    saved = facts['history']['analytical_feedback']
    grids = declared['grids']
    require(declared['patterns'] == 32 and declared['seed'] == 2026101507
            and declared['weight'] == .1 and declared['sampling'] == 'uniform-balanced-grid'
            and len(grids) in (5,8) and len(saved['grids']) == len(grids),
            'Explicit manufactured grids and sampling policy differ')
    rng = np.random.Generator(np.random.PCG64(declared['seed']))
    grid_order = np.concatenate([rng.permutation(len(grids))
        for _ in range(math.ceil(5000/len(grids)))])[:5000]
    within = rng.integers(128,size=5000,dtype=np.int64)
    order = (grid_order*128+within).astype('<i8')
    require(hashlib.sha256(order.tobytes()).hexdigest() == saved['sampling_order_sha256']
            and np.bincount(grid_order,minlength=len(grids)).tolist() == saved['grid_exposures']
            and saved['additional_examples'] == 5000,
            'Balanced grid exposure or complete feedback sample order differs')
    amplitude = -saved['grids'][0]['bank']['samples'][0]['amplitude_pa']/2
    for grid, evidence in zip(grids,saved['grids'],strict=True):
        shape = tuple(reversed(grid['counts_xyz']))
        bank = evidence['bank']
        require(evidence['grid'] == grid and tuple(bank['shape_zyx']) == shape,
                'Manufactured cell counts were confused with physical grid density')
        rng = np.random.Generator(np.random.PCG64(declared['seed']))
        high = tuple(n-1 for n in shape)
        fixed = [(0,0,1),(0,1,0),(1,0,0),(0,1,2),(3,2,1),(4,4,4),(0,0,high[2]),high]
        fixed = [tuple(min(k,n-1) for k,n in zip(f,shape)) for f in fixed]
        descriptions = []
        for index in range(32):
            if index < len(fixed):
                components = [(fixed[index],1.)]
            else:
                components = []
                for _ in range(1 if index%2 else 3):
                    frequencies = tuple(int(rng.integers(n)) for n in shape)
                    if not any(frequencies):
                        frequencies = (0,0,1)
                    components.append((frequencies,float(rng.uniform(.5,1.5))))
            descriptions.extend(dict(components_zyx=components,amplitude_pa=amplitude*factor)
                                for factor in (-2.,-.5,.5,2.))
        bank_order = rng.integers(128,size=1,dtype=np.int64).astype('<i8')
        require(canonical(descriptions) == canonical(bank['samples'])
                and hashlib.sha256(canonical(descriptions)).hexdigest() == bank['bank_description_sha256']
                and hashlib.sha256(bank_order.tobytes()).hexdigest() == bank['sampling_order_sha256'],
                'Manufactured frequencies, signed amplitudes or per-bank draw differ')
    projected = 128*(32*sum(math.prod(g['counts_xyz']) for g in grids)+24*len(grids))
    require(hashlib.sha256(canonical(saved['grids'])).hexdigest() == saved['bank_description_sha256']
            and saved['resident_tensor_bytes'] == projected <= declared['cache_max_bytes'] == 805306368
            and saved['loss_weights'] == dict(pressure=.25,gradient=.25,equation=.5,amplification=1.),
            'Manufactured bank identity, resident budget or objective differs')
    if len([p for p in facts['split']['parents'] if p['entry']['split'] == 'train']) > 1:
        require(declared['amplitude_reference'] == saved['amplitude_reference'] == 'train-correction-cell-rms'
                and saved['amplitude_rms_pa'] == amplitude,
                'Cohort feedback must use recorded train-only cell-weighted correction RMS')
    else:
        require('amplitude_reference' not in declared and 'amplitude_rms_pa' not in saved,
                'Single-frame bank unexpectedly overrides amplitude')
    return dict(manufactured_grid_count=len(grids),manufactured_examples=128*len(grids),
        manufactured_grid_exposures=saved['grid_exposures'],manufactured_resident_tensor_bytes=projected,
        manufactured_description_and_sample_order_rederived=True,original_feedback_amplitude_pa=amplitude,
        manufactured_grids=grids)


def run():
    require(not OUTPUT.exists(),'Preserve original review report')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    index = read(index_path)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {e['id']:e for e in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score = read(DOC/'reports/pressure-common-score-01.json')['models']
    timing_path = DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    timing = read(timing_path)
    require(timing['status'] == 'all127_named_pressure_models_two_pass_GPU_timings_and_exact_accuracy_join_verified',
            'Qualified complete-call timing required')
    purposes = {
        'C26':'Expand manufactured Neumann feedback to five domain shapes, including 24/32/40-cubed grids, while retaining one physical training frame and one correction.',
        'C27':'Test two shared-weight pressure corrections with the same single physical frame and five-grid manufactured bank as C26.',
        'C28':'Replace one-frame fitting with a 16-trajectory physical cohort and use its cell-weighted correction RMS for two-correction manufactured feedback.',
        'C29':'Repeat the C28 two-correction policy with different model-initialization and physical-frame sampling seeds.',
        'C30':'Add a third initialization/sampling repeat of the C28 two-correction policy.',
        'C31':'Measure a single-correction control on the same 16-trajectory cohort and five-grid manufactured bank as C28.',
        'C32':'Measure a four-correction control on the same 16-trajectory cohort and five-grid manufactured bank as C28.',
        'C33':'Add three irregular rectangular manufactured grids at 8 cells/metre to the two-correction cohort policy.',
        'C34':'Repeat the eight-grid rectangular-feedback two-correction policy with a second initialization/sampling seed pair.',
        'C35':'Repeat the eight-grid rectangular-feedback two-correction policy with a third initialization/sampling seed pair.',
        'C36':'Test a single-correction control with the eight-grid rectangular-feedback cohort policy.'}
    lessons = {
        'C26':'Manufactured 32/40-cubed operator fields are real additional training work, but are not large-grid CFD scenes. A one-correction fit still has substantial common error; larger synthetic shapes alone do not establish accuracy at larger simulation domains.',
        'C27':'Two corrections improve both common worst errors over C26 with matching initialization, physical samples and manufactured draws. These corrections consume work inside a pressure call and produce no additional simulation frames.',
        'C28':'Broader physical fitting and train-cohort amplitude calibration improve both common errors over C27. Both factors change, so the effect cannot be assigned to scene count alone. The 16 physical trajectories remain small grids; the larger shapes are manufactured operator fields.',
        'C29':'A second seed pair gives different common worst errors on the same physical/manufactured populations. Repeating architecture labels is not repeating the fitted state; initialization and physical sample order both change.',
        'C30':'The third repeat confirms meaningful seed variation, especially in worst pressure error. Three fits do not establish an uncertainty distribution or unseen-scene qualification.',
        'C31':'The single-correction cohort control is faster work in principle but materially less accurate than C28 on both common worst metrics. Use measured complete GPU latency for the actual speed tradeoff.',
        'C32':'Four corrections improve both common worst errors relative to the matched one/two-correction cohort controls. More internal pressure work is not more temporal advancement; complete-step drift remains unmeasured by these pressure cases.',
        'C33':'The rectangular bank improves common worst equation error over C28 but worsens common worst pressure error. Five-grid exposures fall from 1,000 to 625 each when eight grids share the same 5,000 proposals; this is not an equal-exposure pure geometry comparison.',
        'C34':'The second rectangular-bank seed pair improves both common worst errors relative to C33. Its range overlaps the five-grid repeat results, so a single best seed does not prove the bank universally superior.',
        'C35':'A third rectangular-bank repeat retains the same physical cohort and feedback draw policy but changes initialized weights and physical draws. Compare all repeats and both error metrics instead of promoting one training loss.',
        'C36':'The single-correction rectangular-bank control remains much less accurate than C33 on both common worst metrics. It improves over the corresponding five-grid single-correction control, while changing the allocation of manufactured exposures.'}
    rows, notes, states, populations, orders = [], [], {}, {}, {}
    for number in range(26,37):
        label = f'C{number:02}'
        facts = load_fit(label,work=work,material=material,index=index,score=score)
        meta, config, history = facts['metadata'], facts['config'], facts['history']
        train = [p for p in facts['split']['parents'] if p['entry']['split'] == 'train']
        cohort = number >= 28
        require((meta['actual_train_trajectories'],meta['actual_train_frames'],meta['actual_validation_trajectories'],meta['actual_validation_frames'])
                == ((16,32,64,128) if cohort else (1,1,64,128)), 'Actual physical populations differ')
        densities = sorted({1/p['cell_size_m'] for p in train})
        require(densities == ([8.,16.] if cohort else [8.])
                and max(max(p['counts_xyz']) for p in train) == 16
                and all(p['entry']['dt_s'] == 1/60 for p in facts['split']['parents']),
                'Physical training density/domain/timestep differs')
        if cohort:
            require(all(len(p['frames']) == 2 for p in train),'Cohort physical frame count differs')
        else:
            require(train[0]['counts_xyz'] == [16,16,16] and train[0]['frames'][0]['index'] == 15,
                    'Original one-frame physical training differs')
        corrections = config['correction']['steps']
        require(corrections == (1 if number in (26,31,36) else 4 if number == 32 else 2)
                and config['conditioning'] == 'physical-residual' and config['normalization'] == 'none'
                and config['correction']['activation'] == 'linear' and config['output_scale'] == 'previous-pressure-rms'
                and meta['actual_parameter_count'] == 370016 and meta['initialized_tensor_count'] == 19
                and all(name.endswith('.weight') for name in facts['inventory']), 'Executed linear pressure policy differs')
        require(config['training']['updates'] == 5000 and config['training']['batch_size'] == 1
                and config['training']['learning_rate'] == .001
                and config['training']['schedule'] == [dict(learning_rate=.0001,update=3000),dict(learning_rate=.00001,update=4500)],
                'Optimizer update budget/schedule differs')
        physical = (facts['project']/'physical_residual.py').read_text()
        analytical = (facts['project']/'analytical.py').read_text()
        trainer = (facts['project']/'train.py').read_text()
        require("q=residual*(rho/h).reshape(-1,1,1,1,1)" in physical
                and "module.register_parameter('bias',None)" in physical
                and 'if isinstance(layer,nn.SiLU):block.layers[index]=nn.Identity()' in physical
                and "eigenvalue=math.fsum(2-2*math.cos(math.pi*k/n)" in analytical
                and 'grid_order=np.concatenate([rng.permutation' in analytical
                and '(analytical_weight*extra).backward()' in trainer,
                'Source no longer implements the reviewed physical/operator computation')
        bank = check_bank(facts)
        require(bank['manufactured_grid_count'] == (8 if number >= 33 else 5),'Manufactured family grid count differs')
        common, perf = score[label]['worst'], timing['models'][label]['summary']
        method = ('A one-channel, 370,016-parameter linear convolutional pressure-correction network with 19 weight tensors. '
            'The inherited cubic-groupnorm-v3 label does not imply executed GroupNorm, SiLU or biases: these are removed. '
            'It forms q=(rho/h)*native pressure-equation residual, divides by the train-only q RMS and applies F32 convolutions. '
            f'The configured {corrections} shared-weight correction(s) recompute the F64 finite-volume residual between passes; '
            'outputs use previous-pressure RMS scaling and cell-zero gauge before F64 addition. '
            'Physical dt is validated but is not an independent CNN channel. The operator is sealed uniform cubic-voxel Neumann pressure; '
            'there are no independent obstacle/free-surface masks, velocity outputs or temporal advancement.')
        experiment = (f'5,000 whole-network Adam updates, batch 1, learning rate 0.001 then 0.0001 at update 3,000 and 0.00001 at 4,500, '
            f'model seed {config["training"]["seed"]}, physical sampling seed {config["training"]["sampling_seed"]}, gradient clipping 1.0. '
            f'Physical fitting uses {meta["actual_train_trajectories"]} trajectories / {meta["actual_train_frames"]} frames, '
            f'{"8 and 16" if cohort else "8"} cells/metre, at most 16 cells on each axis, dt=1/60 s; '
            'validation uses 64 trajectories / 128 frames at 8/16 cells/metre. '
            f'Manufactured feedback adds {bank["manufactured_examples"]} pressure-operator fields across {bank["manufactured_grid_count"]} grids '
            '(16/24/32/40-cubed and 24x16x12 at 16 cells/metre'
            + ('; plus 17x13x9, 33x17x9 and 9x25x13 at 8 cells/metre' if number >= 33 else '')
            + f'). Each grid has 32 cosine patterns/mixtures at four signed amplitudes, using recorded amplitude RMS {bank["original_feedback_amplitude_pa"]:.17g} Pa. '
            'Five thousand balanced-grid feedback proposals add weight 0.1 with pressure/gradient/equation weights 0.25/0.25/0.5 and residual-amplification penalty. '
            + ('Amplitude comes from all physical train cells; normalization remains physical-train-only. ' if cohort else 'Amplitude comes from the single physical train correction. ')
            + 'Manufactured examples are not additional physical simulation trajectories.')
        outcome = (f'Last logged training objective {history["history"][-1]["loss"]:.17g}; retained final validation objective {history["validation"]["final"]:.17g}. '
            f'Common 128-request worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}; '
            f'equation relative L2 {common["common_equation_relative_l2"]:.17g}. '
            f'Complete GPU pressure-call timing: median of case medians {perf["median_case_median_ms"]:.17g} ms; '
            f'worst case median {perf["worst_case_median_ms"]:.17g} ms. '
            'Timing includes native CPU inputs, packing/transfers, learned pressure work, independent CPU outputs and synchronization. '
            'It excludes setup/scoring, has no simulation-step denominator, and does not establish own-history drift.')
        row = dict(meta,method=method,purpose=purposes[label],experiment=experiment,outcome=outcome,lesson=lessons[label],
            reproduction='Use the captured typed recipe, config/runtime, exact source witness, input slots, train-only preprocessing, physical split/order and seeded initialization. '
                'Rebuild the manufactured bank from its explicit grids, density, scalar physics, train-only correction amplitude and private seed; retain the balanced exposures. '
                'The original trainer performs CPU deployment/parity forwards after GPU training, so do not execute it unchanged under the current GPU-only policy. '
                'Clean GPU-only retraining, optimizer resume and new-Rheon data regeneration remain unqualified; old/new solver epochs stay separate.',
            retained_timing=binding(timing_path),configured_pressure_corrections=corrections,
            actual_train_cells_per_metre=densities,actual_validation_cells_per_metre=[8,16],**bank)
        original = verified_review_receipt(row)
        note = PROJECT/meta['note_path'];raw = note.read_bytes();revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix),'Original complete receipt/configuration suffix changed')
        notes.append(binding(note));rows.append(row)
        states[label],populations[label],orders[label] = facts['inventory'],population_identity(facts['split']['parents']),facts['sample_order']
    for group in [('C26','C27'),('C28','C31','C32','C33','C36')]:
        require(all(states[k] == states[group[0]] and np.array_equal(orders[k],orders[group[0]])
                    for k in group),'Claimed matched initialization/physical selection differs')
    require(all(populations[f'C{n:02}'] == populations['C28'] for n in range(28,37)),
            'Physical cohort changed across repeat/control comparisons')
    require(all(states[k] != states['C28'] and not np.array_equal(orders[k],orders['C28'])
                for k in ('C29','C30','C34','C35')),'Repeat initialization/sample-order changes were not verified')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes),
            'Native notes/index changed during source review')
    require('torch' not in sys.modules,'Source review imported a neural runtime')
    report = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        source_reviewed_fits=11,new_model_forwards=0,native_mutations=0,new_reference_generation=False,
        matched_physical_populations_initial_tensors_and_sample_orders_verified=True,
        manufactured_descriptors_grid_exposures_and_sample_orders_rederived=True,original_note_bindings=notes,
        clean_reproduction_qualified=False,
        overview_addition='\n\n## Multigrid operator feedback and physical-cohort repeats\n\nC26–C27 add manufactured 32/40-cubed pressure fields but retain one physical training frame. C28–C36 use 16 physical trajectories / 32 frames at 8/16 cells per metre, with at most 16 cells per axis. Their larger and irregular shapes are manufactured operator fields, not large-domain CFD captures. One/two/four pressure corrections and initialization/sampling repeats expose accuracy/work tradeoffs. The eight-grid bank redistributes a fixed feedback budget, so rectangular-bank changes are not pure equal-exposure geometry tests. These pressure-only observations neither generate multiple physical frames nor establish rollout stability.\n',
        producer=binding(Path(__file__)),helpers=[binding(ROOT/'experiments/model-portfolio-migration-v1/captured_pressure_review.py'),
            binding(ROOT/'experiments/model-portfolio-migration-v1/review_water_learning_sequence.py')])
    OUTPUT.write_bytes(canonical(report)+b'\n')
    print('Eleven multigrid/recurrence fits source-reviewed; no neural execution or native mutation.',flush=True)


if __name__ == '__main__':
    run()
