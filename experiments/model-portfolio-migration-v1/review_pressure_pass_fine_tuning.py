"""Review C57--C61; distinguish fitting recurrence from default deployment."""
from collections import Counter
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
OUTPUT = DOC/'reports/captured-pressure-pass-fine-tuning-method-reviews-01.json'
SPECS = [('C57','two-pass-control',2,0.),('C58','four-pass-physics',4,0.),
         ('C59','four-pass-teacher',4,.1),('C60','balanced-control',4,.1),('C61','low-mode-tail',4,.1)]


def run():
    require(not OUTPUT.exists(), 'Preserve original fine-tuning source reviews')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw, index = index_path.read_bytes(), read(index_path)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score_path, timing_path = DOC/'reports/pressure-common-score-01.json',DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    scores, timing = read(score_path)['models'],read(timing_path)['models']
    worker = ROOT/'experiments/model-portfolio-migration-v1/pressure_portfolio_gpu_timing.py'
    require('state.model.predict_pressure(state.network, device, state.normalization)' in worker.read_text(),
            'Retained benchmark default pressure entrypoint differs')
    rows, notes, facts = [], [], {}
    for name,arm,passes,teacher_weight in SPECS:
        f = load_fit(name,work=work,material=material,index=index,score=scores)
        meta,cfg,h = f['metadata'],f['config'],f['history']
        fine = read(f['project']/'fine-tuning.json')
        inputs = {r['id']:ROOT/r['path'] for r in read(f['project']/'project.json')['data']}
        require(meta['actual_parameter_count'] == 370016 and meta['initialized_tensor_count'] == 19
                and meta['actual_train_trajectories'] == 16 and meta['actual_train_frames'] == 32
                and meta['actual_validation_trajectories'] == 0
                and cfg['training']['updates'] == 1500 and cfg['training']['batch_size'] == 1
                and cfg['training']['learning_rate'] == 1e-5
                and cfg['training']['schedule'] == [dict(update=1000,learning_rate=1e-6),dict(update=1350,learning_rate=1e-7)]
                and cfg['correction'] == dict(activation='linear',steps=2), 'Matched fine-tuning budget/default inference differs')
        require(fine['arm'] == arm and fine['training_passes'] == passes and fine['teacher_weight'] == teacher_weight
                and fine['teacher_passes'] == 8 and fine['early_weight'] == .05
                and fine['relative_equation_target'] == 1e-6
                and fine['initial_inventory'] == f['inventory']
                and binding(inputs['initial_weights'])['sha256'] == fine['initial_weights_sha256'] == material['C53']['weights_sha256'],
                'Explicit fitting recurrence/frozen teacher/warm checkpoint differs')
        initial = check_tensors(f['training']/'checkpoint-000.npz',f['inventory'])
        warm = check_tensors(inputs['initial_weights'],fine['initial_inventory'])
        require(all(initial[k].tobytes() == warm[k].tobytes() for k in initial), 'Fine-tuning warm parameter values differ')
        check_tensors(f['training']/'weights.npz',h['checkpoints']['checkpoint-1500.npz']['inventory'])
        require(f['receipt']['outputs']['weights.npz']['sha256'] == f['receipt']['outputs']['checkpoint-1500.npz']['sha256'],
                'Selected weights must equal final-update state')
        auxiliary = audit_auxiliary(cfg,h,inputs,1500)
        current = audit_auxiliary(dict(direction_feedback=cfg['current_feedback']),
            dict(direction_feedback=h['current_feedback']),
            dict(direction_manifest=inputs['current_manifest']),1500)['direction_feedback']
        manifest = read(inputs['current_manifest'])
        require(binding(inputs['current_dataset'])['sha256'] == manifest['stages'][cfg['current_feedback']['stage']]['sha256']
                == h['current_feedback']['payload_sha256'] and current['cases'] == 144,
                'Frozen current-feedback stage payload differs')
        stats = f['receipt']['fine_tuning_statistics']
        require(h['optimizer_work']['network_forwards'] == 1500*5*passes
                and stats['teacher_forwards'] % 8 == 0
                and stats['teacher_cache_peak_bytes'] <= fine['teacher_cache_bytes'] == 268435456,
                'Student fitting work or bounded teacher cache differs')
        if teacher_weight:
            require(stats['teacher_forwards'] > 0 and stats['teacher_cache_hits'] + stats['teacher_forwards']//8 == 7500,
                    'Frozen teacher cache miss/hit accounting differs')
        else:
            require(stats == dict(teacher_forwards=0,teacher_cache_hits=0,teacher_cache_peak_bytes=0),
                    'Zero teacher weight performed auxiliary teacher work')
        source = (f['project']/'fine_tuning.py').read_text()
        objective_source = (f['project']/('original_fine_tuning.py' if name in ('C60','C61') else 'fine_tuning.py')).read_text()
        baseline_source = (f['project']/'baseline_model.py').read_text()
        require("'steps': SPEC['training_passes']" in objective_source
                and 'equations[-1]' in objective_source and "SPEC['early_weight']*torch.stack(equations[:-1]).mean(0)" in objective_source
                and 'return physical_residual.predict_pressure(network,sample,normalization,owner,CORRECTION)' in baseline_source,
                'Actual training/default inference recurrence ownership differs')
        low_weight = 0.
        if name in ('C60','C61'):
            low_weight = .1 if name == 'C61' else 0.
            require(fine['low_mode_feedback'] == dict(weight=low_weight,max_mode=3,signal_floor_fraction=.1)
                    and 'final_feedback=final_feedback if SPEC[\'low_mode_feedback\'][\'weight\'] else None' in source,
                    'Final low-mode objective selection differs')
        actual = Counter((p['cell_size_m'],tuple(p['counts_xyz'])) for p in f['split']['parents'])
        common,perf = scores[name]['worst'],timing[name]['summary']
        purpose = {
            'C57':'Continue C53 with a matched two-correction logarithmic-equation fine-tuning control and no teacher loss.',
            'C58':'Increase differentiable fitting recurrence from two to four corrections while retaining the same warm start, physical samples, budget and no teacher loss.',
            'C59':'Add frozen eight-correction C53 teacher feedback to the four-correction fitting objective; deploy no teacher.',
            'C60':'Establish a new-seed balanced manufactured sampling control for the final low-mode objective comparison.',
            'C61':'Add final spatial low-mode supervision at weight 0.1 to the matched C60 four-correction student/eight-correction teacher fitting setup.'}[name]
        row = dict(meta,fine_tuning_contract=binding(f['project']/'fine-tuning.json'),warm_start_weights=binding(inputs['initial_weights']),
            actual_training_corrections=passes,default_deployment_corrections=2,teacher_weight=teacher_weight,
            teacher_work=stats,student_optimizer_network_forwards=h['optimizer_work']['network_forwards'],
            auxiliary_training=auxiliary,current_policy_feedback=current,physical_population_shapes=len(actual),
            retained_timing=binding(timing_path),common_score_source=binding(score_path),default_inference_source=binding(worker),
            intended_four_correction_variant_freshly_benchmarked=False,
            method='A 370,016-parameter linear one-channel F32 multiscale pressure CNN initialized from the exact C53 checkpoint. F64 equation assembly scales the residual by rho/h before train-only normalization; shared learned corrections are gauged and accumulated in F64. Fine-tuning explicitly owns its recurrence separately from the captured default pressure inference function. The teacher is a frozen copy of the warm checkpoint, has no optimizer and is absent from deployed inference. These are internal pressure passes, not simultaneous physical simulation steps.',
            purpose=purpose,
            experiment=f'1,500 CUDA Adam updates at initial rate 1e-5, dropping at updates 1000/1350 to 1e-6/1e-7; batch one. Model seed {cfg["training"]["seed"]}, physical sample seed {cfg["training"]["sampling_seed"]}. Sixteen physical training trajectories/32 frames at 8/16 cells per metre, dt=1/60 s and rho=1000 kg/m3 retain the two manufactured banks and two captured-direction banks. All five streams use {passes} student corrections, totaling {h["optimizer_work"]["network_forwards"]:,} student root forwards. The loss adds log1p(final relative equation squared/1e-12), plus total weight 0.05 over earlier correction stages. Frozen eight-correction teacher pressure has weight {teacher_weight:g}; its cache records {stats["teacher_forwards"]:,} teacher neural forwards and {stats["teacher_cache_hits"]:,} hits. Final low-spatial-mode feedback weight is {low_weight:g}; it does not add a deployed neural pass.',
            outcome=f'Retained final logged fitting loss {h["history"][-1]["loss"]:.17g}. The common 128 consumed requests exercise the captured default TWO-correction deployment: worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}, equation relative L2 {common["common_equation_relative_l2"]:.17g}; median case-median complete GPU pressure call {perf["median_case_median_ms"]:.17g} ms. Its CPU input/output, packing, transfers and synchronization are included; setup/scoring are excluded. Possible host metadata overlap is recorded and its timing effect is unmeasured. Four-correction fitting labels do not establish four-correction benchmark coverage. The native fitting diagnostic is train-only, with one already-consumed wrapper probe. No new inference, full-step or drift claim is made.',
            lesson='Report training recurrence, default inference recurrence and teacher work separately. Four-pass fitting can change the weights while the saved default model still executes two pressure passes; a label alone cannot identify the measured variant. Comparing C57/C58 isolates fitting recurrence under their shared sample order; C58/C59 additionally isolates teacher supervision. C60/C61 share a later sample stream for the final low-mode ablation, so C59/C60 also changes sampling. A teacher cache saves repeated training work without changing inference cost or adding independent simulation scenes.',
            reproduction='Use the exact native evidence recipe, fine-tuning contract, C53 initial tensors, original source/config/runtime and all fixed input banks. Preserve model/physical/manufactured/direction sampling streams, selected final checkpoint, teacher cache statistics, training-only objective and explicitly chosen deployment correction count. Historical publication/restore includes CPU neural queries; qualify a GPU-only equivalent before replay. Retain the original diagnostic and configuration suffix unchanged. Clean training, fresh four-correction common benchmarking, new-Rheon regeneration and 600-output own-history drift remain unqualified.')
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix), 'Original fine-tuning receipts changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
        facts[name] = f
    require(np.array_equal(facts['C57']['sample_order'],facts['C58']['sample_order'])
            and np.array_equal(facts['C58']['sample_order'],facts['C59']['sample_order'])
            and np.array_equal(facts['C60']['sample_order'],facts['C61']['sample_order'])
            and not np.array_equal(facts['C59']['sample_order'],facts['C60']['sample_order'])
            and index_path.read_bytes() == index_raw
            and 'torch' not in sys.modules, 'Matched physical sampling/read-only/no-neural invariant failed')
    require(all(binding(ROOT/n['path']) == n for n in notes), 'Original fine-tuning notes changed')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=5,
        new_model_forwards=0,native_mutations=0,new_reference_generation=False,clean_reproduction_qualified=False,
        original_note_bindings=notes,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in ['captured_native_pressure_review.py',
            'captured_replay_pressure_review.py','review_pressure_qualification_fits.py',
            'prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Five fine-tuning fits reviewed; exact warm tensors, private sample orders, student/teacher work and default two-pass measurements.')


if __name__ == '__main__':
    run()
