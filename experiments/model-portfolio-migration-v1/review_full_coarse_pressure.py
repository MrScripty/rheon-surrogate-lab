"""Audit C56's completed fit, clipping repair and failed solver qualification.

Reads captured source as bytes, never imports it. All neural weights are checked
with NumPy; no simulation fields, neural forwards or native writes occur.
"""
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_replay_pressure_review import check_tensors
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_current_pressure_feedback import audit_current
from review_pressure_qualification_fits import audit_auxiliary
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-full-coarse-pressure-method-review-01.json'


def run():
    require(not OUTPUT.exists(), 'Preserve original C56 method review')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    index = read(index_path)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    entry = material['C56']
    alias = [a for a in work['named_entries'] if a['label'] == 'C56' and a['item_id'].startswith('cohorts/')]
    require(len(alias) == 1, 'One original C56 alias required')
    fid = alias[0]['fit_item_id']
    fit = next(f for f in work['fit_cards'] if f['item_id'] == fid)
    card = next(c for c in index['fits'] if c['item_id'] == fid)
    receipts = [ROOT/p for p in fit['receipt_paths'] if (ROOT/p).is_file() and binding(ROOT/p)['sha256'] == fit['receipt_sha256']]
    require(len(receipts) == 1 and card['receipt_sha256'] == fit['receipt_sha256']
            and not card.get('method_review_status'), 'Exact unreviewed C56 receipt/card required')
    receipt_path = receipts[0]
    training, project = receipt_path.parent, ROOT/entry['project']
    receipt, history = read(receipt_path), read(training/'history.json')
    config, witness = read(project/'config.json'), read(project/'source-witness.json')
    split, normalization = read(training/'splits.json'), read(training/'normalization.json')
    require(receipt['mode'] == 'train' and receipt['status'] == 'complete'
            and receipt['optimizer_updates'] == config['training']['updates'] == 5000
            and config['training']['batch_size'] == 1
            and config['correction'] == dict(activation='linear',steps=2), 'C56 completed fitting budget differs')
    for name,key in [('config.json','experiment_sha256'),('source-witness.json','source_witness_sha256'),
                     ('runtime-manifest.json','runtime_manifest_sha256')]:
        require(binding(project/name)['sha256'] == receipt['closure'][key], 'C56 original source/config/runtime differs')
    sources = [binding(project/n) for n in sorted(witness['sources'])]
    require(all(s['sha256'] == witness['sources'][Path(s['path']).name] for s in sources), 'C56 complete witnessed sources differ')
    document, recipe = read(project/'project.json'), read(project/'train-recipe.json')
    require(recipe['entrypoint'] == str((project/'full_fit.py').relative_to(ROOT))
            and recipe['arguments'][0] == dict(literal='train')
            and recipe['runtime'] == 'managed_torch_python_v1', 'Actual full-fit wrapper recipe differs')
    for spec in [*document['recipes'],*recipe['sources']]:
        require(binding(ROOT/spec['path'])['sha256'] == spec['sha256'], 'C56 typed source/recipe differs')
    inputs = {r['id']:ROOT/r['path'] for r in document['data']}
    bound_inputs = []
    for slot,expected in receipt['closure']['inputs'].items():
        path = inputs[slot]
        observed = binding(path)
        require(dict(sha256=observed['sha256'],bytes=path.stat().st_size) == expected, 'C56 original input differs')
        bound_inputs.append(dict(observed,slot=slot,bytes=expected['bytes']))
    for name,expected in receipt['outputs'].items():
        path = training/('training-profile.json' if name == 'profile.json' else name)
        require(dict(sha256=binding(path)['sha256'],bytes=path.stat().st_size)
                == {k:expected[k] for k in ('sha256','bytes')}, 'C56 original output/profile differs')
    require(binding(training/'normalization.json')['sha256'] == receipt['closure']['normalization_sha256']
            and history['diagnostic_population'] == receipt['diagnostic_population'] == 'train_only'
            and history['heldout_evaluations'] == receipt['heldout_evaluations'] == 0
            and not receipt['scientific_acceptance_or_model_promotion'], 'C56 normalization/diagnostic scope differs')
    parents = split['parents']
    eligible = [i for i,p in enumerate(parents) if p['entry']['split'] == 'train']
    require(len(eligible) == 16 and sum(len(parents[i]['frames']) for i in eligible) == 32
            and len(parents) == 16 and all(p['entry']['dt_s'] == 1/60 and p['density_kg_m3'] == 1000 for p in parents),
            'C56 physical training population differs')
    rng = np.random.Generator(np.random.PCG64(config['training']['sampling_seed']))
    order = []
    for _ in range(5000):
        p = eligible[int(rng.integers(len(eligible)))]
        order.append([(p,int(rng.integers(len(parents[p]['frames']))))])
    require(hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest()
            == receipt['closure']['sample_order_sha256'], 'C56 private physical draw order differs')
    inventory = history['checkpoints']['checkpoint-000.npz']['inventory']
    require(sum(math.prod(v['shape']) for v in inventory.values()) == 374112 and len(inventory) == 20,
            'C56 complete parameter inventory differs')
    initial = check_tensors(training/'checkpoint-000.npz',inventory)
    final = check_tensors(training/'weights.npz',history['checkpoints']['checkpoint-5000.npz']['inventory'])
    require(initial['spatial_coarse.weight'].shape == (64,64) and not initial['spatial_coarse.weight'].any()
            and final['spatial_coarse.weight'].any(), 'Zero-initialized learned coarse addition did not change')
    parent_history = read(ROOT/material['C53']['restore']/'history.json')
    require({n:v for n,v in inventory.items() if n != 'spatial_coarse.weight'}
            == parent_history['checkpoints']['checkpoint-000.npz']['inventory'], 'C56 base initialization differs from C53')
    reference = read(project/'gradient-reference.json')
    require(reference['format'] == 'pressure-coarse-full-fit-reference-v1'
            and all(binding(project/('baseline_train.py' if n == 'train.py' else n))['sha256'] == s
                    for n,s in reference['original_sources'].items())
            and all(binding(project/n)['sha256'] == s for n,s in reference['repair_sources'].items()),
            'Full-fit clipping reference sources differ')
    baseline = (project/'baseline_train.py').read_bytes()
    old_import = b'from prepared import PreparedSample\n'
    old_clip = b'torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)'
    require(baseline.count(old_import) == baseline.count(old_clip) == 1
            and (project/'train.py').read_bytes() == baseline.replace(old_import,old_import+b'from gradient_clipping import clip_l2_\n')
                .replace(old_clip,b'clip_l2_(network.parameters(), 1.0)'), 'Clipping-only trainer repair differs')
    current = audit_current(dict(project=project,config=config,history=history))
    auxiliary = audit_auxiliary(config,history,inputs,5000)
    old_status_path = project.parent/'work-status.json'
    old_status = read(old_status_path)
    require(old_status['training_completed'] and old_status['optimizer_updates'] == old_status['successful_cuda_checks'] == 5000
            and old_status['pressure_qualification'] == 'failed' and not old_status['candidate_promoted']
            and not old_status['full74_campaign_complete'] and old_status['holdouts_consumed'] == 0
            and history['optimizer_work']['network_forwards'] == 50000, 'Historical completed-fit/failed-qualification distinction differs')
    score_path, timing_path = DOC/'reports/pressure-common-score-01.json', DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    score = read(score_path)['models']['C56']
    perf = read(timing_path)['models']['C56']['summary']
    require(receipt['outputs']['weights.npz']['sha256'] == receipt['outputs']['checkpoint-5000.npz']['sha256']
            == score['weights_sha256'] == entry['weights_sha256'] == alias[0]['declared_weights_sha256'], 'C56 selected final tensor archive differs')
    common = score['worst']
    row = dict(fit_item_id=fid,note_path=card['card_path'],named_checkpoint='C56',original_receipt=binding(receipt_path),
        history=binding(training/'history.json'),sources=sources,source_witness=binding(project/'source-witness.json'),
        original_recipe=binding(project/'train-recipe.json'),original_configuration=binding(project/'config.json'),
        training_population=binding(training/'splits.json'),training_inputs=bound_inputs,actual_parameter_count=374112,
        initialized_tensor_count=20,actual_training_parents=16,actual_training_frames=32,optimizer_updates=5000,
        sample_order_independently_rederived=True,base_initialization_matches_C53=True,initial_coarse_matrix_zero=True,
        clipping_reference=binding(project/'gradient-reference.json'),clipping_only_trainer_repair_verified=True,
        current_policy_feedback=current,auxiliary_training=auxiliary,historical_solver_status=binding(old_status_path),
        common_score_source=binding(score_path),retained_accuracy=score['result'],retained_timing=binding(timing_path),
        new_model_forwards=0,clean_reproduction_qualified=False,
        method='A 374,112-parameter linear one-channel F32 pressure CNN: the 370,016-parameter base plus a learned bias-free 64-by-64 spatial coarse matrix. Fine-cell overlap averages restrict to 4-by-4-by-4 bins, dense learned mixing couples the bins globally, clamped cell-centre interpolation reconstructs the full grid, and the mean-free addition scales by (longest cell count/16)^2. Two shared-weight residual corrections use F64 physical assembly and gauging. The coarse branch is learned inference, not a numerical inverse solver; neither correction represents another physical timestep.',
        purpose='Test whether a small trainable global coarse branch reduces long-wavelength pressure error while preserving the original base network, zero coarse initialization and the matched five fitting streams. The completed full-length fit follows a qualified overflow-only gradient-norm repair; it must be distinguished from capped diagnostic attempts and the later failed simulation qualification.',
        experiment='5,000 CUDA Adam updates, batch one, initial rate 0.001 with drops at 3000/4500 to 0.0001/0.00001. Model seed 20261015 and physical sample seed 2026102501. Sixteen physical trajectories/32 frames at 8/16 cells per metre and dt=1/60 s are supplemented with two manufactured spatial banks and two frozen captured-direction streams; all five total 50,000 neural root calls. The current-policy bank uses the late stage at weight 0.1. All 19 initial base tensors match C53; the extra 4,096 coefficients begin at exact zero. Ordinary finite gradient clipping retains its original path; only overflow in a reduction of finite F32 gradients uses scaled F32 norm arithmetic. This is an explicit optimizer change, not a wider model dtype.',
        outcome=f'Retained final training loss {history["history"][-1]["loss"]:.17g}. Common 128 consumed pressure requests: worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}, equation relative L2 {common["common_equation_relative_l2"]:.17g}, median case-median complete GPU pressure call {perf["median_case_median_ms"]:.17g} ms. The pressure and equation tails are worse than C53 on this population. These are pressure-request timings with host input/output and two corrections, not full simulation-step latency; the original possible host-metadata-overlap flag remains. Historical simulation qualification failed: nine windows passed, three native candidate rejections and one independent certificate failure occurred; the 74-case campaign did not finish and the candidate was not promoted. No new execution or drift measurement is added.',
        lesson='Successful optimization and a learned coarse branch do not establish a usable solver. Preserve completed fit weights, pressure-only errors and failed end-to-end qualification as separate outcomes. The coarse matrix provides learned global coupling rather than an exact conservation or inverse-operator guarantee. Comparisons must disclose the clipping repair and frozen feedback banks; this experiment does not isolate architecture alone from every training change.',
        reproduction='Retain the actual full_fit.py typed entrypoint, its gradient reference and exact clipping/observer sources, all original code/config/runtime/input bindings, 16-parent split, normalization, private draw orders and both checkpoint inventories. Preserve historical failure dispositions and original receipts. The historical evidence publication/restore performs CPU neural queries; qualify a GPU-only equivalent before replay. Clean retraining, fresh native solver qualification, new-Rheon regeneration and long own-history drift remain pending; no Trial ancestry is invented.')
    raw = (PROJECT/row['note_path']).read_bytes()
    original = verified_review_receipt(row)
    revised,suffix = revised_card(raw,row)
    require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix)
            and index_path.read_bytes() == index_raw and 'torch' not in sys.modules,
            'Read-only C56 original receipt/no-neural invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=[row],source_reviewed_fits=1,
        original_note_bindings=[binding(PROJECT/row['note_path'])],new_model_forwards=0,native_mutations=0,
        new_reference_generation=False,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in ['captured_replay_pressure_review.py',
            'review_current_pressure_feedback.py','review_pressure_qualification_fits.py',
            'prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('C56 reviewed: complete source/data/tensors, zero coarse start, clipping repair, five draw streams and failed qualification.')


if __name__ == '__main__':
    run()
