"""Review two guided temporal fits in place, without importing captured code.

The full archive is hashed; only its manifest and framing are inspected. Native
field arrays are not decoded. Selection arithmetic is rederived from retained
scores, not presented as newly executed model accuracy.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from prepare_method_review_revisions import revised_card, verified_review_receipt
from review_water_learning_sequence import binding, canonical, read, require

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/model-portfolio-migration-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
ASSETS = PROJECT / 'artifacts/sha256'
BASE = ROOT / 'runs/full-ai-data-diversity-v1'
OUTPUT = DOC / 'reports/captured-guided-temporal-component-method-reviews-01.json'
FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer', 'rhs')
SPECS = [
    ('fit-10d3a512b41ab2a6', 'shared', 1724, .001, 'candidate-4'),
    ('fit-d18b434d52b73791', 'pressure', 1732, .1, 'probe-3'),
]


def asset(expected):
    path = ASSETS / expected['sha256']
    observed = binding(path)
    require(observed['sha256'] == expected['sha256']
            and path.stat().st_size == expected['bytes'], 'Captured input bytes differ')
    return path, observed


def inspect_bank(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Unique captured bank members required')
        bank = json.loads(archive.read('manifest.json'))
        require(bank['fields'] == list(FIELDS) and bank['reset_interval'] == 16
                and len(bank['masters']) == 24, 'Exact captured full24 bank required')
        groups, windows, members = set(), [], {'manifest.json'}
        state_count = 0
        for identifier, case in bank['masters'].items():
            require(case['split'] == 'train' and case['physical_scenario_group_id'] not in groups,
                    'Training masters must be independent physical groups')
            groups.add(case['physical_scenario_group_id'])
            states = {s['fine_step']: s for s in case['states']}
            require(len(states) == len(case['states']), 'Unique physical state indices required')
            state_count += len(states)
            starts = sorted({w['start_fine_step'] for w in case['windows']})
            expected = {(s, r, k) for s in starts for r in (30, 60, 120) for k in (1, 2, 3, 4)}
            actual = {(w['start_fine_step'], w['output_steps_per_s'], w['K']) for w in case['windows']}
            require(actual == expected and len(actual) == len(case['windows']), 'All12 temporal configurations required')
            for w in case['windows']:
                start, rate, k = (w[n] for n in ('start_fine_step', 'output_steps_per_s', 'K'))
                require(w['target_fine_steps'] == [start + j * (240 // rate) for j in range(1, k + 1)]
                        and {start, *w['target_fine_steps']} <= states.keys(), 'Whole-output physical target times differ')
            for ordinal, state in enumerate(case['states']):
                previous = None if ordinal % 16 == 0 else case['states'][ordinal - 1]['fine_step']
                require(state['previous_fine_step'] == previous and set(state['arrays']) == set(FIELDS),
                        'Original bounded field decoding chain differs')
                for name, spec in state['arrays'].items():
                    member = f'states/{identifier}/{state["fine_step"]:05}/{name}.z'
                    require(spec['member'] == member and spec['xor_previous'] == (previous is not None)
                            and archive.getinfo(member).file_size == spec['compressed_bytes'], 'Captured field framing differs')
                    members.add(member)
            windows.extend((identifier, s) for s in starts)
        require(set(names) == members and len(windows) == 240 and state_count == 2125,
                'Complete original selected population required')
    return bank, windows


def inspect_scores(history, windows, probe_step):
    """Use recorded scores to independently check aggregation and selection."""
    trials = [('initial', history['initial_full_score'])]
    trials.extend((t['label'], t['score']) for t in history['trials'])
    starts_per_scene = Counter(identifier for identifier, _ in windows)
    for label, score in trials:
        require(score['windows'] == 240 and score['blocks'] == 2880 and score['physical_outputs'] == 7200
                and score['maximum_relative_error'] == max(score['worst_relative'].values()),
                'Recorded complete score dimensions/rank differ')
        rows = score['rows']
        require([(r['identifier'], r['start_fine_step']) for r in rows] == windows
                and all(r['blocks'] == 12 for r in rows), 'Recorded score omits/reorders training windows')
        weighted = math.fsum(r['objective'] / (24 * starts_per_scene[r['identifier']]) for r in rows)
        require(math.isclose(weighted, score['objective'], rel_tol=1e-12)
                and math.isclose(math.fsum(score['components'].values()), score['objective'], rel_tol=1e-12),
                'Independent balanced objective aggregation differs')
    worst = min(trials, key=lambda t: (t[1]['maximum_relative_error'], t[1]['objective']))
    mean = min(trials, key=lambda t: t[1]['objective'])
    require(worst[0] == history['selected_worst_trial'] and worst[1] == history['selected_worst_score']
            and mean[0] == history['best_mean_trial'] and mean[1] == history['best_mean_score'],
            'Independent recorded-score checkpoint selection differs')
    by_label = dict(trials)
    for index, iteration in enumerate(history['iterations'], 1):
        current, proposal = iteration['current'], iteration['proposal']
        probe = by_label[f'probe-{index}']
        norm = iteration['gradient_norm']
        coefficient = (probe['objective'] - current['objective'] + probe_step * norm) / probe_step**2
        candidate_step = min(probe_step, norm / (2 * coefficient)) if coefficient > 0 else probe_step
        require(math.isclose(coefficient, proposal['fitted_quadratic_coefficient'], rel_tol=1e-14)
                and math.isclose(candidate_step, proposal['proposed_step'], rel_tol=1e-14)
                and iteration['candidate'] == by_label[f'candidate-{index}']
                and iteration['accepted'] == (iteration['candidate']['objective'] < current['objective']),
                'Independent quadratic proposal/mean-loss acceptance differs')
    return worst[0], mean[0]


def run():
    require(not OUTPUT.exists(), 'Preserve original guided source review')
    index_path = PROJECT / 'notes/portfolio-index.json'
    index_raw = index_path.read_bytes()
    cards = {c['item_id']: c for c in read(index_path)['fits']}
    bank = windows = None
    rows, notes, initial_states = [], [], []
    for fid, kind, count, probe_step, selected in SPECS:
        pair = BASE / f'guided-{kind}-managed-pair-01'
        train = pair / 'train'
        receipt_path = train / 'receipt.json'
        receipt, history = read(receipt_path), read(train / 'history.json')
        require(cards[fid]['receipt_sha256'] == binding(receipt_path)['sha256']
                and not cards[fid].get('method_review_status'), 'Original unreviewed fit required')
        require(receipt['mode'] == 'train' and receipt['parameter_fits'] == 1
                and receipt['optimizer_updates'] == history['parameter_trial_updates'] == 8
                and receipt['accepted_steps'] == history['accepted_steps'] == 4
                and history['full_gradient_evaluations'] == 4 and history['full_forward_evaluations'] == 8
                and not receipt['cpu_model_forward_executed'] and receipt['execution']['device'] == 'cuda'
                and receipt['training_master_count'] == 24 and receipt['reserved52_references_opened'] == 0,
                'Original CUDA-only finite full-population fit differs')
        config_path = ASSETS / receipt['closure']['config_sha256']
        require(binding(config_path)['sha256'] == receipt['closure']['config_sha256'], 'Original config differs')
        config = read(config_path)
        require(config['widths'] == [4] and config['maximum_iterations'] == 4
                and config['optimizer'] == 'full_population_quadratic_ray'
                and config['probe_normalized_step'] == probe_step, 'Registered fit budget differs')
        sources, source_artifacts, bound_inputs = [], [], []
        for name, digest in receipt['closure']['sources'].items():
            path = ASSETS / digest
            require(binding(path)['sha256'] == digest, 'Complete captured entrypoint closure differs')
            source_artifacts.append(dict(binding(path), original_name=name))
            if name != 'custody_validator.py':
                folder = 'full-ai-next-priorities-v1' if name == 'managed_fit_artifacts.py' else 'full-ai-data-diversity-v1'
                current = ROOT / 'experiments' / folder / name
                require(binding(current)['sha256'] == digest, 'Reviewed named source changed')
                sources.append(binding(current))
        inputs = {}
        for slot, expected in receipt['closure']['inputs'].items():
            # Hash the shared 2.81GB payload once, in place, with bounded memory.
            if slot == 'part_00' and bank is not None:
                path = ASSETS / expected['sha256']
                require(expected == rows[0]['training_archive_identity'], 'Guided fits use different native banks')
            else:
                path, observed = asset(expected)
                if slot != 'part_00':
                    bound_inputs.append(dict(observed, slot=slot, bytes=expected['bytes']))
            inputs[slot] = path
        meta = read(inputs['manifest'])
        require(meta['archive'] == receipt['closure']['inputs']['part_00']
                and meta['model_source_sha256'] == receipt['closure']['inputs']['model_source']['sha256']
                and meta['qualified_index']['source_reference_rate_per_s'] == 240
                and meta['qualified_index']['independent_masters_per_split'] == {'train':24, 'development':12},
                'Captured qualified reference population differs')
        if bank is None:
            bank, windows = inspect_bank(inputs['part_00'])
        require(hashlib.sha256(json.dumps(bank, sort_keys=True).encode()).hexdigest() == meta['bank_semantic_sha256'],
                'Captured bank semantics differ')
        for identifier, case in bank['masters'].items():
            require(meta['qualified_index']['masters'][identifier]['split'] == 'train'
                    and case['native_receipt_sha256'] == meta['reference_eligibility']['masters'][identifier]['receipt_sha256'],
                    'Actual training scene differs from qualified reference')
        members = meta['source_manifest']['members']
        with zipfile.ZipFile(inputs['model_source']) as archive:
            require(len(archive.namelist()) == len(members) and set(archive.namelist()) == set(members),
                    'Captured source/asset membership differs')
            for name, digest in members.items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == digest, 'Captured source/asset member changed')
            trainer = archive.read(f'experiments/full-ai-data-diversity-v1/train_guided_{kind}_v1.py').decode()
            core = archive.read(f'experiments/full-ai-data-diversity-v1/guided_{kind}_fit_core_{"v3" if kind == "shared" else "v1"}.py').decode()
            require('rank = (score[\'maximum_relative_error\'], score[\'objective\'])' in trainer
                    and "weight = window['scene_weight'] * window['window_weight'] / 12" in core,
                    'Captured selection/weighting implementation differs')
        states = {}
        for name, inventory in receipt['checkpoint_inventories'].items():
            require(inventory == history['checkpoint_inventories'][name]
                    and sum(math.prod(v['shape']) for v in inventory.values()) == count,
                    'Original residual parameter inventory differs')
            with np.load(train / name, allow_pickle=False) as arrays:
                require(set(arrays.files) == {'w4.' + n for n in inventory}, 'Complete residual archive required')
                states[name] = {}
                for n, spec in inventory.items():
                    value = arrays['w4.' + n]
                    require(value.dtype.str == spec['dtype'] == '<f4' and list(value.shape) == spec['shape']
                            and np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == spec['sha256'],
                            'Original residual tensor bytes differ')
                    states[name][n] = value.copy()
        initial_states.append(states['initial-weights.npz'])
        for name, expected in receipt['outputs'].items():
            require(dict(sha256=binding(train / name)['sha256'], bytes=(train / name).stat().st_size) == expected,
                    'Original output artifact changed')
        probe = np.load(train / 'probe.npy', allow_pickle=False)
        require(probe.dtype.str == '<f8' and list(probe.shape) == history['probe_shape'] == [3, 667648]
                and np.isfinite(probe).all(), 'Original complete GPU probe shape/type differs')
        del probe
        worst_label, mean_label = inspect_scores(history, windows, probe_step)
        require(worst_label == selected and mean_label == 'candidate-4', 'Expected selected trial differs')
        restore = read(pair / 'restore/receipt.json')
        require(restore['typed_cpu_weight_restore_verified'] and restore['gpu_probe_copied_without_cpu_forward']
                and not restore['cpu_model_forward_executed'] and restore['optimizer_updates'] == 0,
                'Historical restore must be typed restoration without CPU inference')
        for name, expected in receipt['outputs'].items():
            restored = pair / 'restore' / name
            require(dict(sha256=binding(restored)['sha256'], bytes=restored.stat().st_size) == expected,
                    'Retained restore artifact differs from original CUDA output')
        first = history['initial_full_score']
        score, mean = history['selected_worst_score'], history['best_mean_score']
        independent = kind == 'pressure'
        row = dict(fit_item_id=fid, note_path=cards[fid]['card_path'], original_receipt=binding(receipt_path),
            history=binding(train/'history.json'), sources=sources, captured_source_artifacts=source_artifacts,
            original_configuration=binding(config_path), training_inputs=bound_inputs,
            training_archive_identity=receipt['closure']['inputs']['part_00'], training_archive_path=str(inputs['part_00'].relative_to(ROOT)),
            training_archive_rehashed=True, field_payloads_decoded=False, captured_source_members_verified=len(members),
            trained_residual_parameters=count, selected_worst_trial=worst_label, selected_mean_trial=mean_label,
            recorded_initial_score={k:first[k] for k in ('objective','worst_relative')},
            recorded_selected_score={k:score[k] for k in ('objective','worst_relative')},
            recorded_best_mean_score={k:mean[k] for k in ('objective','worst_relative')},
            method=f'A width-four F32 correction network with {count:,} fitted coefficients surrounds a frozen learned full-field guide. One shared spatial trunk per block encodes current/predicted endpoint fields, spacing, physical horizon, ordered prescribed controls and within-block event offsets. Endpoint and intermediate heads correct velocity/tracer. ' +
                ('Two additional bias-free four-coefficient heads predict independent pressure-RHS corrections; the restored learned pressure response converts these to reported pressure. Velocity projection pressure remains internal. This adds two learned correction-pressure responses per physical output.' if independent else
                 'Pressure correction is tied to the frozen learned projection of the velocity correction, with one correction-pressure response per physical output. The guide is evaluated without autograd; only the residual network is fitted.'),
            purpose=('Decouple reported pressure fitting from the solenoidal velocity correction while preserving the same frozen guide and inherited trunk initialization.' if independent else
                     'Fit small missing temporal corrections around a time-aware frozen guide rather than replacing all learned simulation operators or running more fine advances.'),
            experiment=f'All 24 independent training masters, 240 selected scene/start windows, 2,880 blocks and 7,200 physical outputs enter every complete score. Six grid resolutions (8/10/16/25/50/100 cells per metre), 16 shapes through 64³ and rectangular 128-cell axes, four force profiles and three motion types are present. Targets come from the qualified 240-step/s bank; every window covers 30/60/120 outputs/s crossed with K=1/2/3/4. Scenes, starts and temporal configurations are balanced, with 1/K weights within blocks. Four complete-gradient iterations each evaluate a normalized probe and quadratic candidate: eight recorded parameter trials, four mean-loss-accepted steps, not eight Adam updates. Probe step is {probe_step}. Future native states/RHS are labels only; guide-derived RMS sets correction units.',
            outcome=f'Retained initial objective {first["objective"]:.17g}; selected {worst_label} objective {score["objective"]:.17g}, worst pressure {score["worst_relative"]["pressure"]:.17g}, equation {score["worst_relative"]["equation"]:.17g}, velocity {score["worst_relative"]["velocity"]:.17g}, tracer {score["worst_relative"]["tracer"]:.17g}. Mean-best {mean_label} objective is {mean["objective"]:.17g}. ' +
                ('Worst-first selection retains probe-3 even though candidate-4 has lower average loss and much smaller worst pressure error. The selected probe was evaluated but was not the mean-loss-accepted continuation state. ' if independent else
                 'Selected, mean-best and last-evaluated checkpoints contain identical tensors; the small objective reduction leaves worst field errors essentially unchanged. ') +
                'These are original training-population scores, not fresh common-benchmark, held-out or latency results. The full-generator target was not met.',
            lesson='Rank and metric definitions must accompany every checkpoint. Minimizing the maximum across equation/pressure/velocity/tracer can prefer a slightly improved dominant equation error while worsening other fields and average loss. Original scores use frozen RMS denominator floors and native target RHS; they cannot be equated with same-request pressure accuracy of the near-exact F64 anchors. Small corrections still require the complete frozen guide and its cost; fitted residual parameter count is not total generator size.',
            reproduction='Retain the exact configuration, all five entrypoint source artifacts, captured model-source/asset ZIP, runtime and loss normalization, and the single lossless training archive. All source members and archived tensor states were checked in place; native field framing was checked but fields were not decoded. Reproduce the balanced complete scores, F64 gradient accumulation, F32 parameter trials and distinct worst/mean/last selections. Historical CPU restore constructs and checks typed weights and copies the saved GPU probe without neural inference. The original temporary Project root is absent; rebuilding its typed recipe bindings and a clean GPU-only replay/new-Rheon regeneration remain unqualified.')
        raw = (PROJECT / row['note_path']).read_bytes()
        original = verified_review_receipt(row)
        revised, suffix = revised_card(raw, row)
        require(raw.count(original) == revised.count(original) == 1 and revised.endswith(suffix), 'Original facts suffix changed')
        rows.append(row)
        notes.append(binding(PROJECT / row['note_path']))
    common = set(initial_states[0])
    require(common < set(initial_states[1]) and all(np.array_equal(initial_states[0][n], initial_states[1][n]) for n in common),
            'Inherited guided trunk initialization differs')
    require(index_path.read_bytes() == index_raw and all(binding(ROOT / p['path']) == p for p in notes)
            and 'torch' not in sys.modules, 'Read-only/no-neural invariant failed')
    result = dict(schema='rheon.captured-model-method-reviews.v1', created_unix=time.time(), rows=rows,
        source_reviewed_fits=2, original_note_bindings=notes, shared_initial_tensors_exact=True,
        training_population=dict(masters=24, selected_windows=240, retained_states=2125,
            blocks_per_complete_score=2880, physical_outputs_per_complete_score=7200,
            densities=dict(Counter(c['scene']['cells_per_metre'] for c in bank['masters'].values())),
            shapes=dict(Counter('x'.join(map(str,c['scene']['counts'])) for c in bank['masters'].values())),
            force_profiles=dict(Counter(c['scene']['forcing_program']['profile'] for c in bank['masters'].values()))),
        new_model_forwards=0, native_mutations=0, new_reference_generation=False, clean_reproduction_qualified=False,
        producer=binding(Path(__file__)), helpers=[binding(Path(__file__).with_name(n)) for n in
            ['prepare_method_review_revisions.py','review_water_learning_sequence.py']])
    OUTPUT.write_bytes(canonical(result) + b'\n')
    print('Two guided fits reviewed; complete archived sources, tensor states and independent score selection verified.')


if __name__ == '__main__':
    run()
