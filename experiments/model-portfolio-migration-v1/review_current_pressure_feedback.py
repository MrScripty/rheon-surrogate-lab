"""Review current-policy captured feedback stages/weights from original facts."""
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
from review_water_learning_sequence import binding,canonical,read,require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-current-pressure-feedback-method-reviews-01.json'


def audit_current(f):
    config, saved = f['config']['current_feedback'], f['history']['current_feedback']
    refs = {r['id']:ROOT/r['path'] for r in read(f['project']/'project.json')['data']}
    manifest = read(refs['current_manifest'])
    require(binding(refs['current_manifest'])['sha256'] == config['manifest_sha256'] == saved['manifest_sha256']
            and saved['stage'] == config['stage'] and saved['weight'] == config['weight']
            and saved['additional_examples'] == 5000 and len(manifest['cases']) == saved['cases'] == 144,
            'Frozen current-policy bank identity/stage/work differs')
    stage = manifest['stages'][config['stage']]
    require(binding(refs['current_dataset'])['sha256'] == stage['sha256'] == saved['payload_sha256'],
            'Selected current-policy stage bytes differ')
    groups = {}
    for i,c in enumerate(manifest['cases']):
        groups.setdefault(c['identity']['group'],[]).append(i)
    names = sorted(groups)
    rng = np.random.Generator(np.random.PCG64(config['seed']))
    group_order = np.concatenate([rng.permutation(len(names)) for _ in range(math.ceil(5000/len(names)))])[:5000]
    order = np.asarray([rng.choice(groups[names[int(i)]]) for i in group_order],dtype='<i8')
    require(names == saved['groups'] and np.bincount(group_order,minlength=len(names)).tolist() == saved['group_exposures']
            and hashlib.sha256(order.tobytes()).hexdigest() == saved['sampling_order_sha256']
            and sum(math.prod(manifest['cases'][int(i)]['counts_xyz']) for i in order) == saved['forward_cells_per_correction'],
            'Independent current-policy case sampling/work differs')
    source = manifest['source']
    require(binding(ROOT/source['artifacts']/'weights.npz')['sha256'] == source['weights_sha256']
            and binding(ROOT/source['project']/'source-witness.json')['sha256'] == source['source_witness_sha256'],
            'Frozen generating-policy checkpoint differs')
    return dict(stage=config['stage'],weight=config['weight'],cases=144,groups=len(names),
        proposals=5000,sampling_order_independently_rederived=True,manifest=binding(refs['current_manifest']),
        payload=binding(refs['current_dataset']),source_checkpoint_sha256=source['weights_sha256'],
        label_arrays_decoded=False,bank_regenerated_during_optimizer_updates=False)


def run():
    require(not OUTPUT.exists(),'Preserve original current-feedback reviews')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    index = read(index_path)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score_path = DOC/'reports/pressure-common-score-01.json'
    timing_path = DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    scores, timing = read(score_path)['models'], read(timing_path)['models']
    parent_project = ROOT/material['C53']['project']
    parent_witness = read(parent_project/'source-witness.json')
    parent_history = read(ROOT/material['C53']['restore']/'history.json')
    rows, notes, facts = [], [], []
    for name,stage,weight in [('C52','early',.1),('C54','late',.03),('C55','late',.01)]:
        f = load_fit(name,work=work,material=material,index=index,score=scores)
        meta,cfg,h = f['metadata'],f['config'],f['history']
        require(meta['actual_parameter_count'] == 370016 and meta['initialized_tensor_count'] == 19
                and meta['actual_train_trajectories'] == 16 and meta['actual_train_frames'] == 32
                and meta['actual_validation_trajectories'] == 0
                and f['inventory'] == parent_history['checkpoints']['checkpoint-000.npz']['inventory'],
                'Matched current-feedback capacity/physical population/initialization differs')
        require(cfg['current_feedback']['stage'] == stage and cfg['current_feedback']['weight'] == weight
                and cfg['correction'] == dict(activation='linear',steps=2)
                and cfg['training']['updates'] == 5000 and cfg['training']['batch_size'] == 1
                and cfg['training']['seed'] == 20261015 and cfg['training']['sampling_seed'] == 2026102501
                and cfg['band_feedback'] == dict(max_mode=3,signal_floor_fraction=.1,stages='first-and-final',weight=.03),
                'Explicit matched stage/weight/budget differs')
        witness = read(f['project']/'source-witness.json')
        differences = {k for k,v in witness['sources'].items() if parent_witness['sources'].get(k) != v}
        require(differences <= {'config.json','custody-validator.py'},'Scientific source changes beyond the declared current-feedback intervention')
        for checkpoint in h['checkpoints'].values():
            path = f['training']/f'checkpoint-{checkpoint["update"]:03d}.npz'
            check_tensors(path,checkpoint['inventory'])
        auxiliary = audit_auxiliary(cfg,h,{r['id']:ROOT/r['path'] for r in read(f['project']/'project.json')['data']},5000)
        current = audit_current(f)
        common,perf = scores[name]['worst'],timing[name]['summary']
        source = (f['project']/'train.py').read_text()
        require('current_sample' in source and 'current_weight' in source
                and h['optimizer_work']['network_forwards'] == 50000,
                'Actual five-stream/two-correction fitting work differs')
        row = dict(meta,current_policy_feedback=current,auxiliary_training=auxiliary,retained_timing=binding(timing_path),
            common_score_source=binding(score_path),scientific_source_differences_from_C53=sorted(differences),
            method='A 370,016-parameter linear one-channel F32 multiscale pressure CNN with 19 bias-free tensors. F64 residual is scaled by rho/h and train-only RMS; two shared-weight internal corrections recompute the F64 equation residual and accumulate cell-zero-gauged pressure. Training combines final physical loss, nonfinal equation contraction, first/final low-spatial-mode feedback and fixed captured residual-direction streams. The extra current-policy bank affects fitting only; it adds no deployed pressure pass and no physical simulation output.',
            purpose=(f'Test {stage} captured directions from the frozen current-policy bank at weight {weight:g}, retaining the matched C53 architecture, initialization and physical/manufactured fitting streams.'),
            experiment=f'5,000 CUDA Adam updates, batch one, initial rate 0.001 with drops to 0.0001 at 3000 and 0.00001 at 4500. Model seed 20261015; physical sample seed 2026102501. Sixteen physical fitting trajectories/32 frames cover small cubic-cell grids at 8/16 cells per metre and dt=1/60 s. Primary and supplemental manufactured banks plus the earlier captured-direction stream remain active. The additional current-policy stream has 144 frozen cases, {current["groups"]} balanced groups, 5,000 proposals and weight {weight:g}, sampled at its {stage} correction stage. All five streams together record 50,000 neural root forwards. The bank is captured before fitting and is not regenerated online from each updated network.',
            outcome=f'Retained final training loss {h["history"][-1]["loss"]:.17g}. On the common 128 consumed pressure requests, worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}, equation relative L2 {common["common_equation_relative_l2"]:.17g}; median of case-median complete GPU pressure calls {perf["median_case_median_ms"]:.17g} ms. Timings include host input/output, validation/packing, transfers, two corrections and synchronization, excluding setup/scoring. These measurements retain the already-recorded host-metadata-overlap flag; its timing effect was not measured. Native history validation is train-only and the wrapper adds one consumed validation probe. No new model execution or drift result is added.',
            lesson='Early/late refer to internal correction directions of the captured source policy, not physical simulation timesteps. A lower auxiliary weight changes optimization, not deployed work. Current-policy rehearsal is a fixed offline bank, not ongoing on-policy trajectory collection. Compare both pressure and equation tails on the same cases and keep inherited fitting labels, consumed probes and generalization claims distinct.',
            reproduction='Retain the exact native evidence recipe, all witnessed sources/config/runtime, fixed physical/primary/supplemental/direction/current input blobs, train-only normalization, initial tensors and all private sample orders. Preserve the selected early/late payload and frozen generating checkpoint. The historical native wrapper/restore performs CPU neural forwards; qualify a GPU-only publication/replay route before rerunning. Clean retraining, new-Rheon regeneration and own-history drift remain unqualified; original receipts/configurations and historical Trial ancestry remain exact.')
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix),'Original current-feedback facts changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
        facts.append(f)
    require(all(np.array_equal(f['sample_order'],facts[0]['sample_order']) for f in facts)
            and index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes)
            and 'torch' not in sys.modules,'Matched order/read-only/no-neural invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,
        source_reviewed_fits=3,original_note_bindings=notes,new_model_forwards=0,native_mutations=0,
        new_reference_generation=False,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in ['captured_native_pressure_review.py',
            'captured_replay_pressure_review.py','review_pressure_qualification_fits.py',
            'prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Three current-policy feedback fits reviewed; frozen bank stage/weight, all sampling and work checked.')


if __name__ == '__main__':
    run()
