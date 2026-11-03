"""Explain C83--C99 using complete original contracts and pressure benchmarks."""
import math
from pathlib import Path
import sys
import time

import numpy as np

from captured_methods_pressure_review import load_fit
from captured_replay_pressure_review import check_tensors
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT/'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC/'reports/captured-methods-architecture-method-reviews-01.json'


def check_base(initial,base,*,spacing):
    for name,value in base.items():
        expected = initial[name]
        if spacing and name == 'encoders.0.layers.0.weight':
            require(expected.shape[1] == 2 and expected[:,:1].tobytes() == value.tobytes()
                    and not expected[:,1:].any(),'Spacing zero-column warm transformation differs')
        else:
            require(expected.tobytes() == value.tobytes() and expected.shape == value.shape,
                    'Original C53 fine warm tensor differs')


def describe(f,number):
    project,initial = f['project'],f['initial']
    text = (project/'physical_residual.py').read_text()
    parameters = f['metadata']['actual_parameter_count']
    rate = 1000 if number in (88,92,93,96,97) else 1
    if rate == 1000:
        optimizer_source = (project/'head_optimizer.py').read_text()
        require(('MULTIPLIER = 1000.' in optimizer_source or 'lr=learning_rate*1000.' in optimizer_source)
                and "group['lr_multiplier']" in (project/'train.py').read_text(),
                'Executed coarse/head rate multiplier differs')
    branch = dict(executed_coarse_head_learning_rate_multiplier=rate)
    if number in (86,87):
        spec = read(project/'spacing-method.json')
        enabled = number == 87
        require(parameters == 370232 and spec['explicit_spacing'] == enabled
                and 'values*torch.log2(h/(1/8))' in text and 'torch.zeros_like(values)' in text,
                'Executed matched spacing-channel policy differs')
        branch.update(kind='explicit_spacing' if enabled else 'zero_spacing_feature',specification=binding(project/'spacing-method.json'))
        method = ('The retained fine CNN gains a second input column: normalized q multiplied by log2(h/(1/8 m))' if enabled
                  else 'The retained fine CNN gains a second input column that is identically zero')
        method += '. Both matched architectures have 370,232 learned coefficients; the new 8-by-1-by-3-by-3-by-3 convolution column begins at exact zero. All original C53 tensor values are retained. Spacing means voxel size, not fluid mass density.'
        purpose = 'Compare explicit spacing conditioning against an identical two-channel zero-feature control on the same native spacing-varied data.'
        lesson = 'Physical q scaling already uses h; this ablation tests an additional learned spacing interaction. Equal parameter count and matched zero initialization are essential. The extra field remains homogeneous in residual amplitude and does not add an independently simulated scene.'
    elif number in (83,84,94,95,98,99):
        spec = read(project/'hybrid-method.json')
        nodes = spec['coarse_nodes_per_axis']
        sparse = number == 84
        require(parameters == 370016 and nodes == (6 if number in (95,98,99) else 4)
                and ('coarse.numerical if METHOD' in text) and 'coarse.pin_neural_correction' in text,
                'Executed numerical stage/node contract differs')
        coarse = (project/'coarse.py').read_text()
        require(f'count = min({nodes},n)' in coarse and 'torch.cholesky_solve' in coarse,
                'Actual Galerkin coarse dimension differs')
        branch.update(kind='sparse_anchors' if sparse else 'numerical_coarse',nodes_per_axis=nodes,
            specification=binding(project/'hybrid-method.json'),learned_parameters=370016)
        if sparse:
            require('coarse.pin_neural_correction(proposed-numerical_base)' in text
                    and 'coefficients = restrict(' in coarse and '/eigenvalues' in coarse,
                    'Exact global anchor source or pinning differs')
            method = 'A hybrid pressure request with at most four anchor cells per axis. It computes a full global cosine transform and all-mode eigenvalue division, evaluates exact pressure at selected anchors, then interpolates and applies the 370,016-parameter learned fine correction while pinning its change to zero at anchors. The anchors are not independent local voxel solves; their global computation must count.'
            purpose = 'Test numerical pressure anchors plus learned infill while preserving exact anchor values.'
            lesson = 'Sparse returned anchors do not mean sparse global work. The full transform/division/selected inverse and infill belong to the solver. Accuracy at anchors does not guarantee fine-grid accuracy or long rollout stability.'
        else:
            method = (f'A hybrid pressure request: an F64 Galerkin solve on at most {nodes} aggregate centres per axis precedes each learned fine correction. '
                'Clamped trilinear interpolation defines P; transpose restriction and the fixed-gauge Cholesky solve use P-transpose L P. The fine CNN retains 370,016 learned coefficients. Restriction, numerical solve, prolongation, residual assembly and neural correction all belong to the configured pressure call.')
            purpose = f'Test a {nodes}-node-per-axis numerical coarse stage during both fitting and deployment, with the retained C53 fine CNN.'
            lesson = 'A parameter count describes only the learned part of this hybrid. Numerical work and geometry preparation require separate accounting. C94/C95 isolate four/six-node stages under matched retained exposure; comparisons to earlier C83 also change data and training budget.'
    elif number in (85,88,92,93,96,97):
        shape = number in (93,96,97)
        require(parameters == (371214 if shape else 370079) and initial['coarse_logits'].shape == (63,)
                and np.all(initial['coarse_logits'] == np.float32(math.log(1/127)))
                and 'cap*torch.sigmoid(' in text and 'remaining.flatten(1).square().sum(1) <=' in text,
                'Actual learned gain initialization/gate differs')
        if shape:
            require(not initial['shape_head.2.weight'].any() and not initial['shape_head.2.bias'].any()
                    and 'network.shape_head(shape_features)' in text and 'nn.Tanh()' in (project/'model.py').read_text(),
                    'Shape-head activation/zero-output initialization differs')
        branch.update(kind='shape_conditioned_coarse' if shape else 'learned_coarse',added_coarse_parameters=63,
            added_shape_parameters=1135 if shape else 0,numerical_inverse_used=False,
            specification=binding(project/'hybrid-method.json'))
        method = ('The retained 370,016-parameter fine CNN gains 63 sigmoid-bounded positive cosine-mode gains, excluding DC. '
            'Modes 0..3 per axis use F64 transform/reconstruction; the cap is max-axis-count squared/2. Initial gain is one at a 16-cubed grid. '
            'Before every fine pass, the coarse proposal is accepted only if its actual fine-equation residual norm does not increase. No analytic inverse eigenvalues supply the learned proposal. ')
        if shape:
            method += 'A further 1,135-parameter 3-to-16-tanh-to-63 MLP adds shape-dependent logits from log(Nz/16), log(Ny/16), log(Nx/16); its final layer starts at zero. '
        purpose = ('Condition learned global gains on grid shape while retaining C53 fine tensors.' if shape else
            'Test a learned global low-mode correction without an analytic coarse inverse.')
        if number == 88:
            purpose = 'Raise the new coarse-gain learning rate by 1000 relative to fixed fine-weight rates, compared with C85.'
        lesson = ('The acceptance gate controls the coarse proposal only; the following fine correction can still worsen the result. '
            f'The captured optimizer really uses coarse/head rate multiplier {rate}; retain that executed policy rather than transferring later metadata discrepancies to this fit. '
            'Shape features describe voxel counts, not an explicit cell-density input. Learned coefficients, inference stages and scene coverage remain separate quantities.')
    else:
        require(number in (89,90,91) and parameters == 370016,'Plain v2 replay control differs')
        branch.update(kind='physical_replay_control',numerical_or_learned_coarse_stage=False)
        method = 'The retained 370,016-parameter linear fine pressure CNN, with no numerical or learned coarse branch. Shared two-correction fitting/deployment retains original C53 scales and tensor values at initialization.'
        purpose = {89:'Extend the small-only physical control from 1,600 to 2,400 updates.',
                   90:'Mix original small requests with expanded-domain data at fixed 1,600 updates.',
                   91:'Retain the full 1,600-example small stream while adding 800 large-grid examples, for 2,400 updates.'}[number]
        lesson = 'Distinguish fixed-update expansion from retained-small-exposure expansion. C90 reduces small exposure to 800 while C91 retains 1,600. Larger voxel work and additional optimizer updates are not free improvements in inference efficiency.'
    if number in (96,97,98,99):
        purpose += f' This is a matched seed-repeat arm, with model seed {f["config"]["training"]["seed"]} and index seed {f["config"]["training"]["sampling_seed"]}.'
        lesson += ' A repeated seed is evidence about sensitivity within this protocol, not an independent physical holdout.'
    return branch,method,purpose,lesson


def run():
    require(not OUTPUT.exists(),'Preserve original architecture reviews')
    index_path = PROJECT/'notes/portfolio-index.json'
    index_raw,index = index_path.read_bytes(),read(index_path)
    work = read(ROOT/'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {r['id']:r for r in read(ROOT/'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score_path,timing_path = DOC/'reports/pressure-common-score-01.json',DOC/'reports/pressure-portfolio-GPU-timing-01.json'
    scores,timing = read(score_path)['models'],read(timing_path)['models']
    base_history = read(ROOT/material['C53']['restore']/'history.json')
    base = check_tensors(ROOT/material['C53']['restore']/'weights.npz',base_history['checkpoints']['checkpoint-5000.npz']['inventory'])
    rows,notes,facts = [],[],{}
    for number in range(83,100):
        name = f'C{number}'
        entry = material[name]
        f = load_fit(entry['fit_item_id'],ROOT/entry['project'],policy='equal_tiers' if number <= 88 else 'small_replay',work=work,index=index)
        meta,cfg,h,fine = f['metadata'],f['config'],f['history'],f['fine']
        check_base(f['initial'],base,spacing=number in (86,87))
        require(cfg['correction'] == dict(activation='linear',steps=2) and fine['training_passes'] == 2
                and fine['teacher_weight'] == 0 and h['optimizer_work']['network_forwards'] == cfg['training']['updates']*2
                and h['optimizer_work']['forward_cells'] == meta['sampled_cells']*2
                and entry['weights_sha256'] == f['receipt']['outputs']['weights.npz']['sha256'] == scores[name]['weights_sha256'],
                'Matched architecture fitting/deployment work or final checkpoint differs')
        require(binding(f['project']/'fixed-normalization.json')['sha256'] == binding(f['training']/'normalization.json')['sha256'],
                'Fixed original C53 normalization changed')
        branch,method,purpose,lesson = describe(f,number)
        common,perf = scores[name]['worst'],timing[name]['summary']
        row = dict(meta,named_checkpoint=name,actual_architectural_intervention=branch,retained_accuracy=scores[name]['result'],
            common_score_source=binding(score_path),retained_timing=binding(timing_path),C53_fine_initial_values_independently_verified=True,
            method=method+' F64 physical residual scaling q=(rho/h)r, gauging and pressure accumulation surround the F32 learned stages. The result is pressure only; two internal corrections do not produce two physical simulation states.',
            purpose=purpose,
            experiment=f'{cfg["training"]["updates"]:,} CUDA Adam updates, batch one, initial fine rate {cfg["training"]["learning_rate"]:g}, schedule {cfg["training"]["schedule"]}, model seed {cfg["training"]["seed"]}, index seed {cfg["training"]["sampling_seed"]}. '
                f'{meta["actual_train_trajectories"]} physical training trajectories/{meta["actual_train_frames"]} requests; grid-density counts {meta["actual_cells_per_metre"]}. '
                f'Actual tier exposure {meta["tier_exposures"]} and sampled voxel count {meta["sampled_cells"]:,}; {h["optimizer_work"]["network_forwards"]:,} fine-network root forwards. '
                f'Executed added coarse/head learning-rate multiplier {branch["executed_coarse_head_learning_rate_multiplier"]}. '
                'All inherited manufactured/direction/current-policy banks are disabled; C53 fine initialization and fixed scales are retained. Physical dt remains 1/60 s and fluid density 1000 kg/m3.',
            outcome=f'Retained final fitting loss {h["history"][-1]["loss"]:.17g}. Common 128 consumed requests: worst mean-free pressure relative L2 {common["pressure_mean_free_relative_l2"]:.17g}, equation relative L2 {common["common_equation_relative_l2"]:.17g}, median case-median complete GPU pressure call {perf["median_case_median_ms"]:.17g} ms. '
                'The captured default call includes its declared numerical/learned stages, CPU input/output, transfers and synchronization. Cached geometry is warmed; setup/first calls/scoring are separate. These are pressure-request observations, not full-step latency, independent holdout evidence or own-history drift. No new model execution is added.',
            lesson=lesson,
            reproduction='Retain the exact typed evidence recipe, witnessed model/preprocessing/coarse sources, method specifications, optimizer group policy, runtime, physical views, fixed scales and transformed C53 initialization. Reproduce the exact private tier/replay stream, including preserved small draws and all large-grid voxel work. Numerical stages require their interpolation, gauge and cached-geometry policy; learned stages require gains, gating and any shape-head initialization. Historical publication/restore has CPU neural calls: qualify GPU-only equivalents before rerunning. Clean fitting, new-Rheon generation and long own-history drift remain unqualified; preserve original receipts and historical ancestry.')
        raw = (PROJECT/row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised,suffix = revised_card(raw,row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix),'Original architecture facts changed')
        rows.append(row)
        notes.append(binding(PROJECT/row['note_path']))
        facts[name] = f
    require(np.array_equal(facts['C86']['order'],facts['C87']['order'])
            and np.array_equal(facts['C94']['order'],facts['C95']['order'])
            and np.array_equal(facts['C96']['order'],facts['C98']['order'])
            and np.array_equal(facts['C97']['order'],facts['C99']['order']),
            'Claimed matched spacing/node/seed-pair physical sample orders differ')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT/n['path']) == n for n in notes)
            and 'torch' not in sys.modules,'Read-only architecture review invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1',created_unix=time.time(),rows=rows,source_reviewed_fits=17,
        original_note_bindings=notes,new_model_forwards=0,native_mutations=0,simulation_arrays_decoded=False,
        new_reference_generation=False,clean_reproduction_qualified=False,producer=binding(Path(__file__)),
        helpers=[binding(Path(__file__).with_name(n)) for n in ['captured_methods_pressure_review.py','captured_replay_pressure_review.py',
            'prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result)+b'\n')
    print('Seventeen methods variants reviewed; complete closure, C53 transformations, sampler exposure, branch/gate and actual optimizer policies verified.')


if __name__ == '__main__':
    run()
