"""Review pressure data controls, solver-teacher targets and Adam observation."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_methods_pressure_review import load_fit
from captured_replay_pressure_review import check_tensors
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_pressure_capacity_fits import audit_fundamental
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-methods-data-and-replay-method-reviews-01.json'


def audit_adam(f):
    initial,final = f['initial'],f['final']
    documents,rows = [],[]
    for update in [1,10,100,150,200]:
        path = f['training']/f'adam-trace-{update:03d}.npz'
        with np.load(path,allow_pickle=False) as archive:
            document = json.loads(archive['metadata'].tobytes())
            documents.append(document)
            require(document['updates'] == [1,10,100,150,200] and document['parameter_count'] == 370016,
                    'Adam complete declared observation schedule differs')
            layout = document['layout']
            require({r['name'] for r in layout} == set(initial)
                    and all(r['shape'] == list(initial[r['name']].shape) and r['count'] == initial[r['name']].size for r in layout),
                    'Adam parameter vector layout differs')
            record = next(r for r in document['records'] if r['update'] == update)
            require(record['steps_before'] == [float(update-1)]*19 and record['steps_after'] == [float(update)]*19,
                    'Adam actual optimizer state steps differ')
            vectors = {n:archive[f'u{update:03d}_{n}'] for n in document['vector_names']}
            require(set(archive.files) == {'metadata',*[f'u{update:03d}_{n}' for n in document['vector_names']]}
                    and len(vectors) == 6 and all(v.shape == (370016,) and v.dtype.str == '<f4' and np.isfinite(v).all() for v in vectors.values()),
                    'Adam complete finite original vectors differ')
            if update == 1:
                require(vectors['parameters_before'].tobytes() == np.concatenate([initial[r['name']].ravel() for r in layout]).tobytes()
                        and not vectors['exp_avg_before'].any() and not vectors['exp_avg_sq_before'].any(),
                        'Adam initial parameters/zero moments differ')
            if update == 200:
                require(vectors['parameters_after'].tobytes() == np.concatenate([final[r['name']].ravel() for r in layout]).tobytes(),
                        'Adam final observed/checkpoint parameters differ')
            before,after = vectors['parameters_before'],vectors['parameters_after']
            rows.append(dict(update=update,learning_rate=record['group']['lr'],
                unchanged_parameter_fraction=float(np.mean(before == after)),
                parameter_delta_l2=float(np.linalg.norm(after.astype(np.float64)-before.astype(np.float64))),
                raw_gradient_l2=float(np.linalg.norm(vectors['gradient_raw'].astype(np.float64))),
                clipped_gradient_l2=float(np.linalg.norm(vectors['gradient_clipped'].astype(np.float64))),
                archive=binding(path)))
    require(all(d == documents[0] for d in documents) and documents[0]['diagnostic_network_forwards']
            == f['history']['optimizer_work']['diagnostic_network_forwards'] == 960,
            'Adam repeated metadata/observer work differs')
    return dict(observed_updates=rows,original_vectors_checked=30,diagnostic_neural_forwards=960,
        new_optimizer_or_neural_execution=False,optimizer_steps_numerically_replayed=False)


def run():
    require(not OUTPUT.exists(),'Preserve original methods-data reviews')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw,index = index_path.read_bytes(),read(index_path)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score_path,timing_path = DOC/'reports/pressure-common-score-01.json',DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    scores,timing = read(score_path)['models'],read(timing_path)['models']
    specs = [('teacher-coarse','fit-0c1daefb000fee1c','pressure-methods-v1/teacher-coarse-project-01','equal_tiers'),
             ('v2-small16','fit-80e3e9ce8eb8ab94','pressure-methods-v2/v2-small16-project-02','small_replay'),
             ('adam-replay','fit-d1c66c9301a5b890','pressure-adam-replay-v1/attempt-04/adam-replay-project-01','uniform')]
    specs += [(f'C{n}',material[f'C{n}']['fit_item_id'],material[f'C{n}']['project'].removeprefix('runs/'),'equal_tiers') for n in range(78,83)]
    rows,notes,facts = [],[],{}
    for name,fid,folder,policy in specs:
        f = load_fit(fid,ROOT/'runs'/folder,policy=policy,work=work,index=index)
        meta,cfg,fine,h = f['metadata'],f['config'],f['fine'],f['history']
        require(meta['actual_parameter_count'] == 370016 and meta['initialized_tensor_count'] == 19
                and fine['initial_weights_sha256'] == material['C53']['weights_sha256']
                and cfg['correction'] == dict(activation='linear',steps=2)
                and fine['teacher_weight'] == 0,'Data/replay architecture, warm state or deployed teacher differs')
        adam = name == 'adam-replay'
        if adam:
            budget = audit_fundamental(fine,f['receipt']['fine_tuning_statistics']['fundamental_bank'],200,aggregate=True)
            observation = audit_adam(f)
            parent_project = ROOT/'runs/pressure-balanced-shapes-v1/balanced-shapes-project-01'
            require(all((f['project']/n).read_bytes() == (parent_project/n).read_bytes()
                    for n in ['model.py','fine_tuning.py','original_fine_tuning.py','original_fit_bank.py','gradient_clipping.py']),
                    'Adam replay numerical model/objective/bank/clipping owners differ')
            parent_history = read(ROOT/material['C69']['restore']/'history.json')
            parent_values = check_tensors(ROOT/material['C69']['restore']/'weights.npz',parent_history['checkpoints']['checkpoint-200.npz']['inventory'])
            identical = all(f['final'][k].tobytes() == parent_values[k].tobytes() for k in parent_values)
            require(fine['training_passes'] == 4 and h['optimizer_work']['network_forwards'] == budget['student_network_forwards'] == 6400
                    and h['optimizer_work']['forward_cells'] == budget['student_forward_cells'] == 7424000,
                    'Adam replay actual fitting work differs')
            intervention = dict(pure_manufactured_optimizer=True,physical_calibration_trajectories=16,
                physical_calibration_frames=32,optimizer_trajectories=0,optimizer_physical_frames=0,
                fitting_budget=budget,adam_observation=observation,final_parameters_equal_C69=identical)
            purpose = 'Observe C69-style aggregate training at updates 1/10/100/150/200 without assigning optimizer or RNG ownership to the observer.'
            population = 'Eight manufactured axis fundamentals, 200 whole-cycle Adam updates, 1,600 case presentations and 6,400 optimizer neural forwards; the original 16 physical trajectories/32 frames calibrate scales only. Thirty original F32 vectors retain parameters, moments and raw/clipped gradients. The observer additionally records 960 neural diagnostic forwards, outside optimizer work counters.'
            lesson = ('Fewer optimizer updates do not imply fewer learned corrections. '+
                f'Final parameter values equal C69: {identical}. At the final observed update, rate 1e-8 leaves {observation["observed_updates"][-1]["unchanged_parameter_fraction"]:.6%} of parameters unchanged; the rest still move. '
                'This observation does not establish a universal precision floor or identify the cause of every accuracy plateau. Observer overhead/diagnostics are separate from deployed model speed.')
        else:
            require(fine['training_passes'] == 2 and cfg['training']['updates'] == 1600
                    and cfg['training']['learning_rate'] == 1e-5
                    and h['optimizer_work']['network_forwards'] == 3200
                    and h['optimizer_work']['forward_cells'] == 2*meta['sampled_cells'],
                    'Two-correction physical data fitting work differs')
            require(binding(f['project']/'fixed-normalization.json')['sha256']
                    == binding(f['training']/'normalization.json')['sha256'],'Frozen C53 scales changed')
            method = f['split'].get('method')
            source_evidence = f['split'].get('source_evidence',{})
            require(all(binding(ROOT/p)['sha256'] == sha for p,sha in source_evidence.items()),
                    'Native data-generation source/receipt evidence differs')
            templates = {p.get('source_training_parent',p['entry']['trajectory_id']) for p in f['parents']}
            intervention = dict(native_view_method=method,native_generation_evidence=source_evidence,
                source_training_parent_templates=len(templates),inherited_auxiliary_banks_executed=False)
            if name in ('teacher-coarse','C80','C81'):
                factor = dict(teacher_coarse=1,C80=2,C81=4)[name.replace('-','_')]
                require(method['teacher_substeps'] == method['pressure_and_rhs_divisor'] == factor
                        and len(f['parents']) == 16 and len(templates) == 16
                        and all(p['teacher_substeps'] == factor and p['dt_s'] == 1/60 for p in f['parents']),
                        'Solver-teacher target cadence conversion differs')
                purpose = f'Test solver-generated labels from {factor} reference advances per 1/60-second model interval while keeping the same base-pressure history and original C53 scales.'
                population = (f'Sixteen training parents/32 requests from the matched original scene templates. Reference substeps per model interval: {factor}; '
                    'pressure and RHS are divided by that factor, the initial guess uses the coarse trajectory, and the residual is recomputed for that base. '
                    'These are instantaneous fine-trajectory pressure requests converted to model-dt units, not exact fine-integrated transition targets.')
                lesson = 'A reference solver teacher is different from an eight-pass learned teacher. Finer reference evolution changes the sampled requests; it does not make this pressure-only model a multi-step full-state generator. Preserve target conversion, base history and the explicit limitation when judging timestep accuracy.'
            elif name == 'C82':
                require(len(f['parents']) == 64 and len(templates) == 16
                        and method['cells_per_metre'] == [4,8,16,32], 'Spacing variants/templates differ')
                purpose = 'Test native spacing-varied training without adding an explicit spacing feature to the neural input.'
                population = 'Sixty-four native training trajectories/128 requests reuse 16 original grid/force templates at 4/8/16/32 cells per metre. Counts per grid remain the corresponding original counts, so changing spacing changes physical extents. This is 64 generated variants, not 64 independent template draws.'
                lesson = 'Distinguish grid density, voxel counts and physical extent. The one-channel network receives physically scaled q; scalar spacing is used outside the CNN. This control does not demonstrate a learned explicit density channel or independent geometry diversity.'
            else:
                require(meta['actual_train_trajectories'] == (20 if name == 'C79' else 16),'Small/expanded parent population differs')
                purpose = {'C78':'Control continued fitting on the original small-grid physical requests.',
                    'C79':'Add four native 32/64-cubed training parents at 8/16 cells per metre under fixed update count and equal tier exposure.',
                    'v2-small16':'Establish the 1,600-update small-only replay control for the v2 expansion and retained-exposure comparisons.'}[name]
                population = (f'{meta["actual_train_trajectories"]} training trajectories/{meta["actual_train_frames"]} physical requests; '
                    f'actual tier exposures {meta["tier_exposures"]}, sampled voxel count {meta["sampled_cells"]:,}. '
                    'Inherited manufactured/direction/current-policy banks are disabled. Only recorded physical requests enter fitting.')
                lesson = 'An equal-update comparison does not match sampled voxel work or small-scene exposure. Report the actual tier counts, original versus added parents and fixed scales. The v2 private small stream preserves the relevant sample subsequence when expansion is enabled.'
        if name.startswith('C'):
            common,perf = scores[name]['worst'],timing[name]['summary']
            outcome = (f'Retained common 128-request worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}, '
                f'equation relative L2 {common["common_equation_relative_l2"]:.17g}; median case-median complete GPU pressure call {perf["median_case_median_ms"]:.17g} ms. '
                'This covers CPU native input/output, transfers, two corrections and synchronization; setup/scoring are outside. These are consumed pressure requests, not complete simulation steps or independent holdout evidence.')
        else:
            outcome = 'This unnamed fit has no fresh common GPU inference benchmark under its own retained identity. Original training, field-probe and observer diagnostics do not establish worst-case physical accuracy, deployment speed or drift.'
        row = dict(meta,named_checkpoint=name,method='A 370,016-parameter linear bias-free one-channel F32 multiscale pressure CNN initialized from exact C53 values. Physical residual q=(rho/h)r is normalized using the fixed original scales; F64 equation assembly, residual recomputation, gauging and pressure accumulation surround shared neural corrections. Default deployment executes two pressure corrections and advances no physical simulation state.',
            purpose=purpose,experiment=f'Configured {cfg["training"]["updates"]:,} CUDA Adam updates, initial rate {cfg["training"]["learning_rate"]:g}, schedule {cfg["training"]["schedule"]}, model seed {cfg["training"]["seed"]}, index seed {cfg["training"]["sampling_seed"]}. '+population,
            outcome=f'Retained final fitting loss {h["history"][-1]["loss"]:.17g}. '+outcome,
            lesson=lesson,reproduction='Retain the exact typed evidence recipe, all source/config/runtime and input/output bindings, C53 warm tensors, fixed normalization and original private sampling. Native-view manifests retain generation receipts, time mappings and target-conversion policy; recreate these using the preserved old reference before separately qualifying new Rheon. The Adam arm also needs its aggregate gradient order and observational archives. Historical publication/restore contains CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, full-state drift and new-Rheon regeneration remain unqualified; preserve original receipts and ancestry.',
            actual_intervention=intervention,new_model_forwards=0)
        if adam:
            row.update(actual_train_trajectories=0,actual_train_frames=0,calibration_trajectories=16,calibration_frames=32)
        if name.startswith('C'):
            row.update(retained_accuracy=scores[name]['result'],common_score_source=binding(score_path),retained_timing=binding(timing_path))
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix),'Original methods-data facts changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
        facts[name] = f
    v2_same = all(facts['v2-small16']['final'][k].tobytes() == facts['C78']['final'][k].tobytes() for k in facts['C78']['final'])
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes)
            and 'torch' not in sys.modules,'Read-only methods-data review invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=8,
        original_note_bindings=notes,v2_small16_final_parameters_equal_C78=v2_same,new_model_forwards=0,native_mutations=0,
        simulation_arrays_decoded=False,new_reference_generation=False,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in ['captured_methods_pressure_review.py','captured_replay_pressure_review.py',
            'review_pressure_capacity_fits.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Eight data/replay fits reviewed; full custody, source-bound labels, sampler work and 30 Adam observation vectors checked.')


if __name__ == '__main__':
    run()
