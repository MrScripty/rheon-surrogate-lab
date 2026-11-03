"""Explain manufactured-capacity fits using source, tensor and sampler custody.

No captured module is imported. Physical data calibrates some fits but does not
enter their optimizer; those populations are reported separately.
"""
from collections import Counter
import copy
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_native_pressure_review import load_fit
from captured_replay_pressure_review import check_tensors
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_pressure_qualification_fits import audit_auxiliary
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-pressure-capacity-method-reviews-01.json'
SPECS = [('C62',2000),('C63',1500),('C64',4800),('C65',600),('C66',1600),('C67',1600),('C69',200)]


def audit_fundamental(fine,stats,updates,*,aggregate):
    spec = fine['fitting_bank']
    cases = [dict(**g,mode_zyx=m) for g in spec['grids'] for m in spec['modes_zyx']]
    require(cases == stats['cases'] and hashlib.sha256(canonical(cases)).hexdigest() == stats['description_sha256'],
            'Fundamental fitting descriptions differ')
    count = len(cases)
    examples = updates*count if aggregate else updates
    require(stats['optimizer_examples'] == examples and examples % count == 0,
            'Fundamental optimizer example budget differs')
    rng = np.random.Generator(np.random.PCG64(spec['sampling_seed']))
    order = np.concatenate([rng.permutation(count) for _ in range(examples//count)]).astype('<i8')
    exposures = np.bincount(order,minlength=count).tolist()
    require(hashlib.sha256(order.tobytes()).hexdigest() == stats['sampling_order_sha256'],
            'Fundamental private draw order differs')
    require(exposures == stats['expected_exposures'] == stats['optimizer_exposures'],
            'Fundamental optimizer exposures differ')
    if aggregate:
        require(stats['optimizer_updates'] == updates and stats['aggregation'] == 'mean-complete-balanced-cycle',
                'Fundamental complete-cycle aggregation differs')
    cells = sum(math.prod(cases[int(i)]['counts_xyz']) for i in order)
    residency = len(spec['modes_zyx'])*sum(32*math.prod(g['counts_xyz'])+24 for g in spec['grids'])
    require(stats['resident_tensor_bytes'] == residency <= spec['cache_max_bytes'],
            'Fundamental exact tensor residency differs')
    require(len(stats['trace']) == 20,'Fundamental complete recorded diagnostic trace required')
    for n,row in enumerate(stats['trace'],1):
        require(row['update'] == n*10*(1 if aggregate else count)
                and row['exposures_per_case'] == n*10
                and [(m['case'],m['passes']) for m in row['measurements']]
                    == [(i,k) for i in range(count) for k in (1,2,4,8)]
                and all(math.isfinite(m['relative_equation_residual']) and m['relative_equation_residual'] >= 0 for m in row['measurements']),
                'Fundamental retained diagnostic trace structure differs')
    return dict(manufactured_cases=count,optimizer_examples=examples,optimizer_updates=updates,
        aggregation='mean-complete-balanced-cycle' if aggregate else 'one-example-per-update',
        exposures_per_case=exposures,sampling_order_sha256=stats['sampling_order_sha256'],
        private_order_independently_rederived=True,student_network_forwards=examples*4,
        student_forward_cells=cells*4,retained_diagnostic_points=20,diagnostics_freshly_recomputed=False)


def audit_curriculum(config,history):
    spec,saved = config['supplemental_feedback'],history['supplemental_feedback']
    require(spec['patterns'] == 32 and saved['source_bank_schema'] == 'rheon-frequency-curriculum-v1'
            and saved['curriculum'] == dict(seed=spec['seed'],fundamental_patterns=12,
                other_low_patterns=12,high_patterns=8,max_low_mode=3), 'Supplemental static frequency specification differs')
    updates = config['training']['updates']
    grids = spec['grids']
    rng = np.random.Generator(np.random.PCG64(spec['seed']))
    grid_order = np.concatenate([rng.permutation(len(grids)) for _ in range(math.ceil(updates/len(grids)))])[:updates]
    order = (128*grid_order+rng.integers(128,size=updates,dtype=np.int64)).astype('<i8')
    require(saved['additional_examples'] == updates and saved['grid_exposures'] == np.bincount(grid_order,minlength=len(grids)).tolist()
            and hashlib.sha256(order.tobytes()).hexdigest() == saved['sampling_order_sha256'],
            'Supplemental static frequency draw order differs')
    categories = []
    amplitude = saved['amplitude_rms_pa']
    for grid,bank in zip(grids,saved['grids'],strict=True):
        z,y,x = (n-1 for n in reversed(grid['counts_xyz']))
        modes = [('fundamental',m) for m in [(0,0,1),(0,1,0),(1,0,0)]*4]
        modes += [('other_low',m) for m in [(2,0,0),(0,2,0),(0,0,2),(3,0,0),(0,3,0),(0,0,3),
                  (1,1,0),(1,0,1),(0,1,1),(1,1,1),(1,2,3),(3,2,1)]]
        modes += [('high',m) for m in [(4,0,0),(0,4,0),(0,0,4),(4,4,4),(z,0,0),(0,y,0),(0,0,x),(z,y,x)]]
        expected = [dict(components_zyx=[(m,1.)],amplitude_pa=amplitude*a,category=c)
                    for c,m in modes for a in (-2.,-.5,.5,2.)]
        require(bank['grid'] == grid and canonical(bank['samples']) == canonical(expected),
                'Supplemental static frequency descriptors differ')
        categories.extend(c for c,_ in modes for _ in range(4))
    exposure = dict(Counter(categories[int(i)] for i in order))
    require(saved['actual_mode_category_exposures'] == exposure,
            'Supplemental static frequency category exposures differ')
    require(hashlib.sha256(canonical(saved['grids'])).hexdigest() == saved['bank_description_sha256'],
            'Supplemental static frequency bank identity differs')
    return dict(manufactured_fields=len(categories),grids=len(grids),proposals=updates,
        mode_category_exposures=exposure,descriptors_and_sampling_independently_rederived=True,
        spatial_frequencies=True,temporal_frequency_or_timestep_curriculum=False,
        repeated_fundamental_patterns=True)


def select_original_training_receipt(work):
    """C62's preliminary copy is the same fit; choose its complete train closure."""
    selected = copy.deepcopy(work)
    fit = next(f for f in selected['fit_cards'] if f['item_id'] == 'fit-4f5ce39a5ba8aace')
    original = [ROOT/p for p in fit['receipt_paths']]
    require(len(original) == 2 and all(binding(p)['sha256'] == fit['receipt_sha256'] for p in original),
            'C62 duplicate receipt identity differs')
    complete = [p for p in original if (p.parent/'history.json').is_file() and (p.parent/'checkpoint-000.npz').is_file()]
    require(len(complete) == 1,'One complete original C62 training closure required')
    fit['receipt_paths'] = [str(complete[0].relative_to(ROOT))]
    return selected,[binding(p) for p in original]


def run():
    require(not OUTPUT.exists(),'Preserve original capacity reviews')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw,index = index_path.read_bytes(),read(index_path)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    selected_work,receipt_copies = select_original_training_receipt(work)
    material = {r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score_path,timing_path = DOC/'reports/pressure-common-score-01.json',DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    scores,timing = read(score_path)['models'],read(timing_path)['models']
    purposes = {
        'C62':'Test fitting capacity on one axis fundamental on one rectangular grid, calibrated using original train-only scales.',
        'C63':'Rebalance only the supplemental spatial-frequency bank toward axis fundamentals while retaining the five physical/manufactured/direction streams.',
        'C64':'Fit the complete 24-case combination of eight rectangular grid/density pairs and three axis fundamentals.',
        'C65':'Isolate three axis fundamentals on the single 13x15x16 grid.',
        'C66':'Isolate variation across eight rectangular grid/density pairs while fitting only the x-axis fundamental.',
        'C67':'Delay the learning-rate drops in the matched C66 eight-shape fitting test.',
        'C69':'Accumulate and average one complete eight-shape cycle before each optimizer step, comparing equal example exposure with C66.'}
    rows,notes,facts = [],[],{}
    for name,updates in SPECS:
        f = load_fit(name,work=selected_work if name == 'C62' else work,material=material,index=index,score=scores)
        meta,cfg,h = f['metadata'],f['config'],f['history']
        fine = read(f['project']/'fine-tuning.json')
        refs = {r['id']:ROOT/r['path'] for r in read(f['project']/'project.json')['data']}
        require(meta['actual_parameter_count'] == 370016 and meta['initialized_tensor_count'] == 19
                and meta['actual_train_trajectories'] == 16 and meta['actual_train_frames'] == 32
                and meta['actual_validation_trajectories'] == 0 and cfg['training']['updates'] == updates
                and cfg['training']['batch_size'] == 1 and cfg['correction'] == dict(activation='linear',steps=2)
                and fine['training_passes'] == 4 and fine['early_weight'] == .05
                and fine['relative_equation_target'] == 1e-6,
                'Capacity fit manifest/population/default recurrence differs')
        require(binding(refs['initial_weights'])['sha256'] == fine['initial_weights_sha256'] == material['C53']['weights_sha256']
                and f['inventory'] == fine['initial_inventory'], 'Capacity fit C53 warm checkpoint differs')
        initial = check_tensors(f['training']/'checkpoint-000.npz',f['inventory'])
        warm = check_tensors(refs['initial_weights'],fine['initial_inventory'])
        require(all(initial[n].tobytes() == warm[n].tobytes() for n in initial),'Capacity warm parameter values differ')
        check_tensors(f['training']/'weights.npz',h['checkpoints'][f'checkpoint-{updates:03d}.npz']['inventory'])
        stats = f['receipt']['fine_tuning_statistics']
        source = (f['project']/'train.py').read_text()
        calibration = f['normalization']['training_provenance']
        require(calibration['frames'] == 32 and len(calibration['trajectory_ids']) == 16,
                'Original calibration physical population differs')
        pure = name != 'C63'
        if pure:
            require(fine['teacher_weight'] == 0 and all(stats[k] == 0 for k in ['teacher_forwards','teacher_cache_hits','teacher_cache_peak_bytes'])
                    and h['analytical_feedback'] is None and 'current_feedback' not in h
                    and 'analytical_samples=None;analytical_evidence=None' in source
                    and 'current_samples=None;current_evidence=None' in source,
                    'Pure fitting test unexpectedly executes inherited auxiliary banks/teacher')
            if name == 'C62':
                case,saved = fine['fitting_case'],stats['single_case']
                require(case == saved['case'] and saved['optimizer_examples'] == updates
                        and saved['optimizer_network_passes_per_example'] == 4
                        and 'single_case.one_update(' in source and len(saved['trace']) == 20,
                        'Single manufactured fitting stream differs')
                budget = dict(manufactured_cases=1,optimizer_examples=updates,optimizer_updates=updates,
                    student_network_forwards=updates*4,student_forward_cells=updates*4*math.prod(case['counts_xyz']),
                    case=case,amplitude_rms_pa=saved['amplitude_rms_pa'],physical_optimizer_examples=0)
                for i,entry in enumerate(saved['trace'],1):
                    require(entry['update'] == i*100 and [m['passes'] for m in entry['measurements']] == [1,2,4,8],
                            'Single-case retained trace structure differs')
            else:
                require('fit_bank.one_update(' in source and 'fit_bank.bind(cache,config[\'training\'][\'updates\'])' in source,
                        'Actual fundamental optimizer owner differs')
                budget = audit_fundamental(fine,stats['fundamental_bank'],updates,aggregate=name == 'C69')
            require(h['optimizer_work']['network_forwards'] == budget['student_network_forwards']
                    and h['optimizer_work']['forward_cells'] == budget['student_forward_cells'],
                    'Actual pure manufactured neural work differs')
            population = (f'Original 16 physical trajectories/32 frames calibrate fixed normalization and manufactured pressure amplitude, '
                f'but no physical scene frame enters the optimizer. Actual fitting uses {budget["manufactured_cases"]} manufactured fundamental fields '
                f'and {budget["optimizer_examples"]:,} example presentations; all inherited auxiliary banks are disabled. '
                f'There are {updates:,} optimizer updates and {budget["student_network_forwards"]:,} student neural root forwards. ')
        else:
            base_history = {k:v for k,v in h.items() if k != 'supplemental_feedback'}
            auxiliary = audit_auxiliary(cfg,base_history,refs,updates)
            current = audit_auxiliary(dict(direction_feedback=cfg['current_feedback']),dict(direction_feedback=h['current_feedback']),
                dict(direction_manifest=refs['current_manifest']),updates)['direction_feedback']
            curriculum = audit_curriculum(cfg,h)
            require(h['optimizer_work']['network_forwards'] == updates*5*4 and fine['teacher_weight'] == .1
                    and stats['teacher_forwards'] == 15400 and stats['teacher_cache_hits'] == 5575,
                    'Curriculum five-stream/frozen teacher work differs')
            budget = dict(manufactured_curriculum=curriculum,auxiliary_training=auxiliary,current_feedback=current,
                physical_optimizer_examples=updates,student_network_forwards=30000,
                teacher_network_forwards=15400,teacher_cache_hits=5575)
            population = ('Sixteen physical training trajectories/32 frames plus two manufactured and two frozen captured-direction streams remain active. '
                'Only the supplemental bank changes: eight grid/density pairs each have 32 spatial patterns at four amplitudes; '
                'the 12/12/8 pattern allocation repeats each axis fundamental four times. Its 1,500 actual category draws are '
                '579 fundamental, 542 other low and 379 high. This is fixed spatial-frequency weighting, not a time-varying temporal curriculum. '
                'The student records 30,000 root forwards and its frozen eight-correction teacher records 15,400 forwards plus 5,575 cache hits. ')
        common,perf = scores[name]['worst'],timing[name]['summary']
        row = dict(meta,physical_manifest_train_trajectories=16,physical_manifest_train_frames=32,
            actual_train_trajectories=0 if pure else 16,actual_train_frames=0 if pure else 32,
            calibration_trajectories=16,calibration_frames=32,pure_manufactured_optimizer=pure,
            original_fine_tuning_contract=binding(f['project']/'fine-tuning.json'),actual_fitting_budget=budget,
            training_corrections=4,default_deployment_corrections=2,intended_four_correction_variant_freshly_benchmarked=False,
            retained_timing=binding(timing_path),common_score_source=binding(score_path),
            method='A 370,016-parameter linear, bias-free one-channel F32 multiscale pressure CNN warm-started from exact C53 tensors. F64 sealed cubic-grid equation assembly uses q=(rho/h)r and fixed train-only normalization. Shared neural corrections are gauged and accumulated in F64. The fitting objective unrolls four pressure corrections with final logarithmic equation loss and total earlier-stage weight 0.05; the captured default inference still executes two. These passes do not advance physical simulation time.',
            purpose=purposes[name],
            experiment=f'{updates:,} CUDA Adam updates; model seed {cfg["training"]["seed"]}, retained physical-index seed {cfg["training"]["sampling_seed"]}, initial rate {cfg["training"]["learning_rate"]:g} and schedule {cfg["training"]["schedule"]}. '+population+
                ('C69 averages all eight case losses/gradients before clipping and one Adam step; C66/C67 step after each case. ' if name == 'C69' else '')+
                'Physical metadata remains dt=1/60 s and fluid density 1000 kg/m3; grid density means cells per metre, not fluid density. Calibration and consumed diagnostics add no independent optimizer scenes.',
            outcome=f'Retained final fitting loss {h["history"][-1]["loss"]:.17g}. Common 128 consumed pressure requests use default TWO-correction inference: worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}, equation relative L2 {common["common_equation_relative_l2"]:.17g}, median case-median complete GPU pressure call {perf["median_case_median_ms"]:.17g} ms. Host input/output, transfers and synchronization are included; setup/scoring are excluded. Fitting-bank 1/2/4/8-correction traces are retained consumed-case equation diagnostics, not independent pressure or full-step benchmarks. No new model execution, fresh four-correction common benchmark or rollout result is added.',
            lesson=('A physical data manifest does not establish physical optimizer exposure. '+
                ('The one-pattern capacity probe can fit a narrow operator direction without demonstrating scene generalization. ' if name == 'C62' else '')+
                ('The frequency curriculum is a static supplemental spatial distribution with repeated modes; it does not train a range of physical time intervals. ' if name == 'C63' else '')+
                ('C66/C67/C69 share exactly 1,600 sampled examples, 200 exposures per case and 6,400 neural forwards. C69 has fewer optimizer updates because it averages a full cycle; those updates are not eight times less neural work. C67 changes rate-drop timing. ' if name in ('C66','C67','C69') else '')+
                'Report calibration data, actual optimizer population, update aggregation, fitting recurrence and measured inference recurrence separately.'),
            reproduction='Retain exact typed native source/config/runtime/input receipts, C53 warm tensors, fine-tuning contract, normalization, final checkpoint and the actual optimizer owner. Recreate manufactured fundamental fields from explicit grid/mode/density definitions and original train-calibrated amplitude; reproduce private balanced cycles and gradient aggregation. For the curriculum preserve its repeated patterns and supplemental category order. Historical publication/restore uses CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, intended four-correction common evaluation, new-Rheon regeneration and long own-history drift remain unqualified. Preserve original receipts and ancestry.')
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix),'Original capacity facts changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
        facts[name] = f
    for name in ['C67','C69']:
        require(facts[name]['receipt']['fine_tuning_statistics']['fundamental_bank']['sampling_order_sha256']
                == facts['C66']['receipt']['fine_tuning_statistics']['fundamental_bank']['sampling_order_sha256'],
                'Claimed identical eight-case presentation stream differs')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes)
            and 'torch' not in sys.modules,'Read-only capacity review imported a neural runtime or changed native files')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=7,
        original_note_bindings=notes,C62_same_fit_receipt_copies=receipt_copies,new_model_forwards=0,native_mutations=0,
        simulation_arrays_decoded=False,new_reference_generation=False,clean_reproduction_qualified=False,
        producer=binding(Path(__file__)),helpers=[binding(Path(__file__).with_name(n)) for n in ['captured_native_pressure_review.py',
            'captured_replay_pressure_review.py','review_pressure_qualification_fits.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Seven capacity/curriculum fits reviewed; calibration versus optimization, exact balanced exposures and default two-pass scope checked.')


if __name__ == '__main__':
    run()
