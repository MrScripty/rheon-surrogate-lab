"""Explain C100--C105 and C108/C109 from retained source and measured results."""
import ast
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from captured_replay_pressure_review import load_fit
from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
OUTPUT = DOC / 'reports/captured-replay-pressure-series-method-reviews-01.json'
SPECS = [('C100',371214,26), ('C101',371214,34), ('C102',92560,34), ('C103',92560,34),
         ('C104',209374,34), ('C105',209374,34), ('C108',371214,34), ('C109',371214,34)]


def original_weights(material, name):
    path = ROOT / material[name]['restore'] / 'weights.npz'
    require(binding(path)['sha256'] == material[name]['weights_sha256'], 'Named warm parent bytes differ')
    with np.load(path, allow_pickle=False) as archive:
        return {n:archive[n].copy() for n in archive.files}, binding(path)


def check_slice(parent, child, numerator, denominator):
    """Reconstruct exact tensor selection without importing the captured slicer."""
    require(set(parent) == set(child), 'Channel-sliced checkpoint tensor keys differ')
    old_coarse_widths = (48,32,24,16)
    for name, actual in child.items():
        source = parent[name]
        if source.shape == actual.shape:
            expected = source
        else:
            require(source.ndim == actual.ndim == 5, 'Only dense convolution tensors may shrink')
            outputs = np.arange(actual.shape[0])
            inputs = np.arange(actual.shape[1])
            if name.startswith('decoders.') and name.endswith('.layers.0.weight'):
                stage = int(name.split('.')[1])
                old_coarse = old_coarse_widths[stage]
                coarse = old_coarse * numerator // denominator
                skip = (source.shape[1] - old_coarse) * numerator // denominator
                inputs = np.r_[np.arange(coarse), old_coarse + np.arange(skip)]
            require(source.shape[0] * numerator // denominator == actual.shape[0] or name == 'head.weight',
                    'Declared convolution output width differs')
            expected = source[outputs][:,inputs]
        require(actual.dtype == expected.dtype and actual.shape == expected.shape
                and actual.tobytes() == expected.tobytes(), 'Independent channel slice differs from warm initialization')


def head_multipliers(path):
    if not path.exists():
        return [1.]
    function = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name == 'groups')
    value = next(n.value for n in function.body if isinstance(n,ast.Return))
    def number(node):
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
            return ast.literal_eval(node.left) / ast.literal_eval(node.right)
        return ast.literal_eval(node)
    return [number(next(k.value for k in entry.keywords if k.arg == 'lr_multiplier')) for entry in value.elts]


def run():
    require(not OUTPUT.exists(), 'Preserve original replay-series reviews')
    index_path = PROJECT / 'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    index = read(index_path)
    work = read(ROOT / 'runs/model-portfolio-migration-v1/discovery/model-worklist-05.json')
    material = {m['id']:m for m in read(ROOT / 'runs/model-portfolio-migration-v1/discovery/pressure-material-01.json')['models']}
    score_path = DOC / 'reports/pressure-common-score-01.json'
    timing_path = DOC / 'reports/pressure-portfolio-GPU-timing-01.json'
    scores, timing = read(score_path)['models'], read(timing_path)['models']
    purposes = {
        'C100':'Continue the C93 shape-conditioned full-width pressure proposal with 1,600 preserved small-grid draws and 800 larger-grid draws, testing physical replay rather than inherited manufactured feedback.',
        'C101':'Expand C100-style replay to native captures including 128³ grids and 32 cells per metre, from the exact same C93 weights and fixed update budget.',
        'C102':'Train a genuinely half-width fine CNN sliced from C37, coupled to a numerical six-node-per-axis Galerkin stage, on the expanded physical replay packet.',
        'C103':'Change the half-width numerical-coarse warm start from C37 to C44 while retaining C102 geometry, data, sampling and optimization budget.',
        'C104':'Test fine-correction distillation into a three-quarter-width C101 student, retaining its complete learned coarse/shape branch.',
        'C105':'Provide C104’s matched three-quarter-width student control with zero fine-correction teacher loss. It is a sibling fit, not a continuation of completed C104.',
        'C108':'Continue the full-width C101 checkpoint on the unchanged expanded replay population as the control for the differentiable GCR intervention.',
        'C109':'Train and deploy two flexible GCR iterations using the learned pressure proposal, comparing solver-aware training and inference with the ordinary two-correction C108 control.',
    }
    lessons = {
        'C100':'Replay uses physical requests and equal update count, not equal voxel work. Retained labels for analytical/direction/current feedback are inherited metadata: the actual trainer disables those sample banks.',
        'C101':'Adding native large-grid captures keeps the 1,600 small draws exact but redistributes the 800 larger draws and increases voxel work by about tenfold. Common small-grid pressure worsens versus C100 while equation residual improves; neither measures large-grid generalization.',
        'C102':'A smaller neural parameter count does not imply lower complete GPU latency: numerical coarse solves and field movement remain. The archived common pressure/equation errors are larger than those of the full-width learned-global replay models.',
        'C103':'This is the same reduced architecture and numerical stage as C102 with a different named-checkpoint initialization. Common tails improve modestly over C102, but no seed replication or independent large-grid qualification follows.',
        'C104':'Teacher supervision adds training work without changing deployed student work. This matched three-quarter-width experiment has worse common pressure/equation tails than the zero-teacher control; that does not establish that distillation is generally harmful.',
        'C105':'The control is more accurate on the common diagnostic than C104 and has almost the same deployed timing. Compare it with the half-width C106/C107 pair while retaining parameter counts, physical data and teacher work separately.',
        'C108':'This starts from completed C101, whereas C100/C101 start from C93. Do not infer continuation from cohort numbering. The deployed solver remains two ordinary learned pressure corrections.',
        'C109':'GCR improves both common pressure and equation tails over C108, with higher complete pressure-call GPU latency. It changes the training unroll and deployed solver work together, including residual actions, reductions and double orthogonalization; this is not an architecture-only or loss-only intervention.',
    }
    rows, notes, facts, warm_sources = [], [], {}, {}
    for name, parameters, parents in SPECS:
        f = load_fit(name, work=work, material=material, index=index, score=scores)
        facts[name] = f
        meta, cfg, h, receipt, fine = (f[k] for k in ('metadata','config','history','receipt','fine'))
        require(meta['parameters'] == parameters and meta['actual_training_parents'] == parents
                and h['optimizer_work']['network_forwards'] == 4800
                and h['optimizer_work']['forward_cells'] == 2 * meta['sampled_cells']
                and h['current_feedback']['status'] == 'inherited_auxiliary_banks_disabled', 'Exact replay capacity/work differs')
        expected_rate = .0003 if name in ('C104','C105') else .00001
        require(cfg['training']['learning_rate'] == expected_rate and cfg['training']['batch_size'] == 1
                and cfg['training']['sampling_seed'] == 2026101804, 'Matched replay optimizer budget differs')
        require(receipt['fine_tuning_statistics']['teacher_forwards'] == 0, 'Old eight-pass teacher must remain inactive')
        multipliers = head_multipliers(f['project'] / 'head_optimizer.py')
        expected_multiplier = [1.] if name in ('C102','C103') else [1.,10/3] if name in ('C104','C105') else [1.,1000.]
        require(multipliers == expected_multiplier, 'Executed head learning-rate multiplier differs')
        teacher_work = 0
        if name in ('C102','C103'):
            parent_name = 'C37' if name == 'C102' else 'C44'
            prior, source = original_weights(material, parent_name)
            check_slice(prior, f['initial'], 1, 2)
            warm_sources[name] = dict(parent=parent_name, weights=source, relation='exact_independently_reconstructed_half_channel_slice')
            method_spec = read(f['project'] / 'hybrid-method.json')
            coarse_source = (f['project'] / 'coarse.py').read_text()
            require(method_spec['kind'] == 'numerical-coarse' and method_spec['coarse_nodes_per_axis'] == 6
                    and 'count = min(6,n)' in coarse_source and 'torch.linalg.cholesky(matrix[1:,1:])' in coarse_source
                    and 'torch.cholesky_solve' in coarse_source, 'Actual numerical coarse stage differs')
            method = 'A 92,560-parameter F32 linear bias-free multiscale fine CNN, widths 4/8/12/16 and bottleneck 24, follows a numerical Galerkin coarse solve at each of two internal pressure corrections. Clamped trilinear interpolation at up to six aggregate centers per axis defines P; the F64 coarse system is PᵀLP with a fixed gauge and Cholesky solve. Restriction, solve, prolongation, residual action and neural inference all belong to the pressure call. This is a hybrid pressure request, not a full fluid simulation step.'
        else:
            parent_name = 'C93' if name in ('C100','C101') else 'C101'
            prior, source = original_weights(material, parent_name)
            if name in ('C104','C105'):
                check_slice(prior, f['initial'], 3, 4)
                distill = read(f['project'] / 'distillation.json')
                weight = 1. if name == 'C104' else 0.
                require(distill['width_fraction'] == [3,4] and distill['weight'] == weight
                        and distill['teacher_weights_sha256'] == source['sha256'], 'Three-quarter-width teacher identity differs')
                teacher_work = receipt['fine_tuning_statistics']['distillation_teacher_forwards']
                require(teacher_work == (4800 if weight else 0)
                        and receipt['fine_tuning_statistics']['distillation_teacher_forward_cells'] == (1224395120 if weight else 0),
                        'Actual distillation teacher work differs')
                relation = 'exact_independently_reconstructed_three_quarter_channel_slice'
                fine_description = '208,176 fine CNN coefficients at widths 6/12/18/24, bottleneck 36'
            else:
                require(set(prior) == set(f['initial']) and all(prior[n].tobytes() == f['initial'][n].tobytes() for n in prior),
                        'Full warm start differs from exact named parent')
                relation = 'exact_selected_parent_parameter_bytes'
                fine_description = '370,016 fine CNN coefficients at widths 8/16/24/32, bottleneck 48'
            warm_sources[name] = dict(parent=parent_name, weights=source, relation=relation)
            method = f'A {parameters:,}-parameter F32 pressure proposal has {fine_description}, 63 learned positive low-mode gains and a 1,135-parameter 3→16 tanh→63 grid-shape head. The fine path is linear and bias-free, with pooling, nearest upsampling, skips and global-mean broadcast. Its input is q=(rho/h)r normalized by fixed original C53 scales; dt is validated scalar metadata, not an independent neural channel. At each correction the learned cosine coarse proposal is gated by its actual F64 fine-equation residual, followed by a neural correction accumulated in F64. '
            method += ('Deployment performs two flexible GCR iterations with two reorthogonalization sweeps, image normalization and a recomputed true residual after each update. No analytic coarse inverse supplies the learned proposal.' if name == 'C109' else
                       'Deployment uses two ordinary internal pressure corrections. These are iterations within one request and do not advance velocity or tracer.')
        if name == 'C109':
            code = (f['project'] / 'fine_tuning.py').read_text()
            require('for _ in range(2):' in code and 'residual=initial-owner.full_operator(delta,h,rho)' in code
                    and 'clamp_min(1e-60).sqrt()' in code
                    and receipt['fine_tuning_statistics']['convergence_loss_calls'] == 2400
                    and receipt['fine_tuning_statistics']['convergence_proposals'] == 4800, 'Actual two-GCR training/deployment differs')
        common, perf = scores[name]['worst'], timing[name]['summary']
        require(timing[name]['weights_sha256'] == receipt['outputs']['weights.npz']['sha256'], 'Timing uses another checkpoint')
        experiment = f'2,400 CUDA Adam updates, batch one, clip norm one. Fine rate {expected_rate:g}; executed optimizer multipliers {multipliers}. Every update uses one recorded physical request: 1,600 preserved small-grid samples plus 800 round-robin size-tier samples. There are {parents} train-only parents and {2*parents} selected frames, fixed dt=1/60 s and fluid density 1,000 kg/m³. Actual parent grid resolutions: {meta["actual_parent_densities"]}; sample tier exposures: {meta["tier_exposures"]}. Actual fine-network work is 4,800 root forwards / {h["optimizer_work"]["forward_cells"]:,} input voxels. Inherited manufactured, direction and current-policy banks are disabled despite retained config labels. Final/intermediate physical equation feedback remains, with early log-residual weight 0.05 and target 1e-6. The old eight-pass teacher has weight zero. '
        if name in ('C104','C105'):
            experiment += f'Fine-correction distillation separately records {teacher_work:,} full-teacher forwards, on detached student-generated residual inputs; teacher work is absent from deployed inference. '
        if name == 'C109':
            experiment += 'GCR is differentiated during training and used at publication/deployment. Its log-residual denominator combines target scaling before clamping to avoid the historical overflow derivative. '
        row = dict(meta, warm_start=warm_sources[name], executed_head_multipliers=multipliers,
            fine_network_forwards=4800, fine_network_forward_cells=h['optimizer_work']['forward_cells'],
            distillation_teacher_forwards=teacher_work, common_score_source=binding(score_path), retained_timing=binding(timing_path),
            method=method, purpose=purposes[name], experiment=experiment,
            outcome=f'Final retained weighted training loss {h["history"][-1]["loss"]:.17g}. On the common 128 original consumed requests, worst mean-free pressure relative L2 is {common["pressure_mean_free_relative_l2"]:.17g}, equation relative L2 {common["common_equation_relative_l2"]:.17g}. Retained complete GPU pressure-call median of case medians is {perf["median_case_median_ms"]:.17g} ms, with CPU input/output, validation/packing, transfers, all solver work and synchronization included. Setup and scoring are separate. The native training diagnostic named validation is train-only, and the wrapper probe is already consumed. The common test covers small 8/16-cells-per-metre grids, not the larger training domains. No fresh forward, latency, trajectory or held-out result is added by this review.',
            lesson=lessons[name], reproduction='Use the exact captured native evidence recipe, all witnessed sources/configuration, runtime, fixed normalization, typed input packet, initial parameter inventory and final-update checkpoint. Preserve the private small/large RNG streams and physical exposure budget, disabled auxiliary banks and executed optimizer groups. Retain channel-slice mappings or exact warm-parent bytes, and distinguish teacher work from student work. Original CPU publication/restore performs neural inference and must be replaced by a qualified GPU-only route before replay. Native recipe rebuilding, clean retraining, new-Rheon regeneration and own-history drift remain unqualified; historical Trial ancestry is unchanged.')
        raw = (PROJECT / row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised, suffix = revised_card(raw, row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix), 'Original retained replay facts changed')
        rows.append(row)
        notes.append(binding(PROJECT / row['note_path']))
    require(np.array_equal(facts['C100']['small_order'],facts['C101']['small_order'])
            and all(np.array_equal(facts[n]['order'],facts['C101']['order']) for n in ['C102','C103','C104','C105','C108','C109']),
            'Matched replay physical sample order differs')
    for a,b in [('C104','C105'),('C108','C109')]:
        require(all(facts[a]['initial'][n].tobytes() == facts[b]['initial'][n].tobytes() for n in facts[a]['initial'])
                and facts[a]['receipt']['closure']['inputs'] == facts[b]['receipt']['closure']['inputs'],
                'Matched comparison initial weights or input packet differs')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT / n['path']) == n for n in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural review invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1', created_unix=time.time(), rows=rows,
        source_reviewed_fits=8, original_note_bindings=notes, independently_verified_warm_starts=warm_sources,
        matched_small_stream_C100_C101=True, matched_full_stream_C101_through_C109=True,
        new_model_forwards=0, native_mutations=0, new_reference_generation=False, clean_reproduction_qualified=False,
        producer=binding(Path(__file__)), helpers=[binding(Path(__file__).with_name(n)) for n in
            ['captured_replay_pressure_review.py','prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result) + b'\n')
    print('Eight pressure replay/student/GCR fits reviewed; exact channel slices, sampling and original outcomes verified.')


if __name__ == '__main__':
    run()
