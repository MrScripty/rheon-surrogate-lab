"""Explain the two retained temporal readout fits without executing a model.

Check original small artifacts and captured source bindings. Native field archives
and the complete transitive source population remain outside this review; existing
full-generator results supply historical outcomes, not a fresh replay or latency.
"""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zipfile

from prepare_method_review_revisions import revised_card, verified_review_receipt


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/full-ai-data-diversity-v1'
PROJECT = ROOT / 'projects/rheon-model-portfolio-unified-04'
OUTPUT = ROOT / 'docs/plans/model-portfolio-migration-v1/reports/captured-macrostep-readout-method-reviews-01.json'
OVERVIEW = '''
## What the three/four-stage temporal readouts learned

The two macrostep readout fits learn only 9 or 12 global float64 coefficients,
not a new full-state network. They combine the current state and intermediate
states of a frozen three/four-internal-stage backbone into the next 1/60 s output.
The three coefficient groups are pressure, shared velocity and tracer.

Training uses all 148 scenes at two starts, but only 29 N8 scenes produce gradients;
119 N1 scenes contribute fixed losses. All 98 retained N8 evaluation rows fail the
diagnostic 1e-5 gate in each model, while all 920 unchanged N1 rows pass. The four-stage
arm has lower worst pressure/equation errors than three stages on those rows,
but neither establishes accurate reduced integration. K physical outputs still
advance sequentially; internal stages and physical future frames are different.
These are original retained GPU results, not new replay, latency or promotion.
'''


def read(path):
    return json.loads(path.read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path.read_bytes()))


def npy(raw):
    """Inspect the original simple typed arrays without numpy or model imports."""
    if raw[:8] != b'\x93NUMPY\x01\x00':
        raise ValueError('Expected retained NPY v1 representation')
    length = struct.unpack('<H', raw[8:10])[0]
    header = ast.literal_eval(raw[10:10 + length].decode('ascii'))
    payload = raw[10 + length:]
    if (header['fortran_order'] or header['descr'] not in ('<f8', '<i8')
            or len(payload) != 8 * math.prod(header['shape'])):
        raise ValueError('Original little-endian array shape/type/length differs')
    return header, payload


def inspect_weights(path, expected, inventory, stages):
    raw = path.read_bytes()
    if dict(sha256=sha(raw), bytes=len(raw)) != expected:
        raise ValueError('Recorded readout checkpoint archive differs')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['residual_weights.npy']:
            raise ValueError('Only the recorded readout tensor is supported')
        header, payload = npy(archive.read('residual_weights.npy'))
    actual = dict(dtype=header['descr'], shape=list(header['shape']), sha256=sha(payload))
    if inventory != {'residual_weights': actual} or actual['shape'] != [3, stages] or actual['dtype'] != '<f8':
        raise ValueError('Actual readout parameter inventory differs')
    values = struct.unpack('<' + 'd' * (3 * stages), payload)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Finite learned coefficients required')
    return values


def inspect_design(path, part):
    raw = path.read_bytes()
    if dict(sha256=sha(raw), bytes=len(raw)) != {k: part[k] for k in ('sha256', 'bytes')}:
        raise ValueError('Original supervised design archive differs')
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {n + '.npy' for n in part['arrays']}:
            raise ValueError('Original supervised design membership differs')
        for name, spec in part['arrays'].items():
            header, payload = npy(archive.read(name + '.npy'))
            if dict(dtype=header['descr'], shape=list(header['shape'])) != spec:
                raise ValueError('Original supervised design array differs')
            if name == 'scene_index' and struct.unpack('<148q', payload) != tuple(range(148)):
                raise ValueError('All original training scenes must be present')
            if name == 'teacher_substeps' and Counter(struct.unpack('<148q', payload)) != {1: 119, 8: 29}:
                raise ValueError('Active and fixed teacher populations differ')
            if name == 'starts' and struct.unpack('<2q', payload) != (32, 96):
                raise ValueError('Original two training windows differ')


def run():
    if OUTPUT.exists():
        raise ValueError('Fresh source review output required')
    manifest_path = BASE / 'macrostep-loss-view-01/manifest.json'
    manifest = read(manifest_path)
    progress_path = BASE / 'macrostep-loss-view-01/progress.json'
    progress = read(progress_path)
    if (manifest['status'] != 'verified_all148_train_macrostep_loss_view'
            or manifest['training_scenes'] != 148 or manifest['training_N8_scenes'] != 29
            or manifest['training_N1_scenes'] != 119 or manifest['development_or_reserved_used']
            or manifest['starts'] != [32, 96] or manifest['metric_to_parameter_group'] != [0, 0, 1, 2]
            or progress['manifest_sha256'] != binding(manifest_path)['sha256']
            or not progress['frozen_assets_unchanged'] or len(progress['rows']) != 592):
        raise ValueError('Original complete train-only supervised view required')
    active = {r['identifier']: r for r in progress['rows'] if r['trainable_readout_active']}
    fixed = {r['identifier']: r for r in progress['rows'] if not r['trainable_readout_active']}
    if len(active) != 29 or len(fixed) != 119 or set(active) & set(fixed):
        raise ValueError('Effective parameter-dependent population differs')
    generation_sources = []
    for name in ('prepare_macrostep_loss_view_v1.py', 'macrostep_loss_design_v1.py',
                 'macrostep_backbone_v1.py', 'macrostep_readout_v1.py'):
        path = ROOT / 'experiments/full-ai-data-diversity-v1' / name
        value = binding(path)
        if manifest['source_sha256'][value['path']] != value['sha256']:
            raise ValueError('Reviewed view-generation source differs')
        generation_sources.append(value)
    cards = {r['item_id']: r for r in read(PROJECT / 'notes/portfolio-index.json')['fits']}
    reviews = []
    for stages in (3, 4):
        project = BASE / f'macrostep-readout-s{stages}-project-01'
        pair = BASE / f'macrostep-readout-s{stages}-pair-01'
        train = pair / 'train'
        receipt, history = read(train / 'receipt.json'), read(train / 'history.json')
        config, definition = read(project / 'config.json'), read(project / 'project.json')
        if (receipt['status'] != 'complete' or receipt['mode'] != 'train'
                or receipt['parameters'] != 3 * stages or receipt['optimizer_updates'] != 160
                or config['stages'] != stages or config['optimizer_calls'] != 160
                or config['objective'] != 'equal_scene_window_four_metric_QR_residual_v1'
                or binding(project / 'config.json')['sha256'] != receipt['closure']['config_sha256']
                or not receipt['execution']['all_learning_and_forward_on_cuda']
                or receipt['execution']['dtype'] != 'float64'):
            raise ValueError('Original managed CUDA fit/configuration differs')
        sources = [binding(project / name) for name in sorted(receipt['closure']['sources'])]
        if len(sources) != 4 or any(r['sha256'] != receipt['closure']['sources'][Path(r['path']).name] for r in sources):
            raise ValueError('Captured four-source training closure differs')
        for declared in definition['recipes']:
            if binding(ROOT / declared['path'])['sha256'] != declared['sha256']:
                raise ValueError('Captured recipe declaration differs')
            for source in read(ROOT / declared['path'])['sources']:
                if binding(ROOT / source['path'])['sha256'] != source['sha256']:
                    raise ValueError('Captured recipe source differs')
        inputs = []
        for declared in definition['data']:
            path = ROOT / declared['path']
            value = dict(slot=declared['id'], **binding(path), bytes=path.stat().st_size)
            if (value['sha256'] != declared['sha256']
                    or {k: value[k] for k in ('sha256', 'bytes')} != receipt['closure']['inputs'][value['slot']]):
                raise ValueError('Static training input differs')
            inputs.append(value)
        if len(inputs) != 4:
            raise ValueError('All four original input slots required')
        part = next(p for p in manifest['parts'] if p['stages'] == stages)
        inspect_design(manifest_path.parent / part['path'], part)
        initial = inspect_weights(train / 'initial-weights.npz', receipt['outputs']['initial-weights.npz'], receipt['initial_inventory'], stages)
        final = inspect_weights(train / 'weights.npz', receipt['outputs']['weights.npz'], receipt['final_inventory'], stages)
        if any(initial) or (history['initial_inventory'], history['final_inventory']) != (receipt['initial_inventory'], receipt['final_inventory']):
            raise ValueError('Original zero initialization or history binding differs')
        if (history['optimizer_step_calls'] != 160 or history['actual_parameter_changed_steps'] != 160
                or history['closure_evaluations'] != (413 if stages == 3 else 398)
                or not history['train_only'] or history['training_scene_count'] != 148
                or history['history'][-1]['optimizer_call'] != 160):
            raise ValueError('Actual optimizer work differs')
        paired = read(pair / 'receipt.json')
        restored = read(pair / 'restore/receipt.json')
        if (paired['training_receipt_sha256'] != binding(train / 'receipt.json')['sha256']
                or paired['restore_receipt_sha256'] != binding(pair / 'restore/receipt.json')['sha256']
                or not all(paired['artifact_parity'].values())
                or not restored['typed_cpu_weight_restore_verified'] or restored['cpu_model_forward_executed']
                or restored['optimizer_updates'] != 0):
            raise ValueError('Original managed restore evidence differs')
        for name, expected in receipt['outputs'].items():
            for directory in (train, pair / 'restore'):
                raw = (directory / name).read_bytes()
                if dict(sha256=sha(raw), bytes=len(raw)) != expected:
                    raise ValueError('Original training/restore output artifact differs')
        for name in ('training_run_id', 'restore_run_id'):
            terminal = read(pair / ('terminal-' + paired[name] + '.json'))
            if terminal['state'] != 'completed' or terminal['custody_phase'] != 'drained':
                raise ValueError('Both original managed runs must be complete/drained')
        result_path = BASE / f'macrostep-student-s{stages}-blocks-01/results.json'
        result = read(result_path)
        identity = result['model_identity']
        q = 2. ** -40
        deployed = struct.pack('<' + 'd' * len(final), *(round(v / q) * q for v in final))
        if (identity['readout_weights_sha256'] != receipt['outputs']['weights.npz']['sha256']
                or identity['deployment_quantization'] != q
                or identity['deployed_readout_sha256'] != sha(deployed)
                or result['status'] != 'completed_macrostep_student_full_state_blocks'
                or result['latency_claim'] or result['cpu_model_forward']):
            raise ValueError('Retained complete-generator result selects another deployment')
        execution_sources = []
        for name in ('macrostep_student_v1.py', 'macrostep_backbone_v1.py', 'evaluate_macrostep_student_blocks_v1.py'):
            source = binding(ROOT / 'experiments/full-ai-data-diversity-v1' / name)
            if result['source_sha256'][source['path']] != source['sha256']:
                raise ValueError('Reviewed execution source differs from retained evaluation')
            execution_sources.append(source)
        regimes = []
        for teacher in (1, 8):
            rows = [r for r in result['rows'] if r['teacher_substeps'] == teacher]
            if any(r['metric_error'] or any(r['nonfinite_counts'].values()) for r in rows):
                raise ValueError('Finite complete retained field metrics required')
            regimes.append(dict(teacher_substeps=teacher, scored_outputs=len(rows),
                diagnostic_failures=sum(not r['all_accuracy_pass'] for r in rows),
                worst_relative={k: max(r['metrics']['relative'][k] for r in rows) for k in ('equation', 'pressure', 'velocity', 'tracer')}))
        if [(r['scored_outputs'], r['diagnostic_failures']) for r in regimes] != [(920, 0), (98, 98)]:
            raise ValueError('Historical active/bypass outcomes differ')
        fit = 'fit-' + binding(train / 'receipt.json')['sha256'][:16]
        review = dict(fit_item_id=fit, note_path=cards[fit]['card_path'],
            original_receipt=binding(train / 'receipt.json'), history=binding(train / 'history.json'),
            sources=sources, training_inputs=inputs, training_configuration=binding(project / 'config.json'),
            view_generation_sources=generation_sources, execution_sources=execution_sources,
            original_complete_generator_result=binding(result_path), original_reference_regimes=regimes,
            parameters=3 * stages, trainable_parameters=3 * stages, optimizer_updates=160,
            closure_evaluations=history['closure_evaluations'], active_gradient_scenes=29,
            fixed_bypass_scenes=119, active_scene_windows=58, fixed_scene_windows=238,
            raw_readout_coefficients=list(final), deployed_readout_sha256=sha(deployed),
            method=f'A {3 * stages}-parameter float64 affine temporal readout with three coefficient groups: pressure, shared vx/vy/vz, and tracer. Each group combines differences between the current/intermediate states and the last of {stages} frozen internal advances. Zero coefficients initially return the final backbone state. Pressure is gauge-fixed; velocity/tracer cast to float32. The frozen backbone supplies all spatial transport, forcing, viscosity and pressure work. This fit does not train that backbone.',
            purpose=f'Replace eight fine internal advances with {stages} larger uniform internal advances for each 1/60 s physical output, then learn a small global temporal combination to correct the endpoint. This is distinct from predicting four future physical frames from one distant endpoint.',
            experiment=f'CUDA double LBFGS with 160 outer optimizer calls, max_iter 1 per call, strong-Wolfe search, history 20, and {history["closure_evaluations"]} actual closure evaluations; all 160 calls change parameters. Optimize equally averaged scene/window/four-metric squared residuals from full-grid QR factors.148 train scenes at starts 32/96 contribute 296 windows, but only 29 N8 scenes/58windows depend on these coefficients;119 N1 scenes/238windows are fixed controls. Active grids span eight shapes through 64 cubed and 8/10/16/25/50/100 cells per metre. Final optimizer state is saved; no held-out model selection is performed.',
            outcome=f'First logged objective after optimizer call 1 is {history["history"][0]["objective"]:.17g}; final call 160 is {history["history"][-1]["objective"]:.17g}. This is a mean squared training objective, not worst pressure error. Original complete-generator GPU evaluation records 1,018 output rows: all 920 N1 bypass outputs pass and all 98 N8 outputs fail the diagnostic 1e-5 gate. N8 worst equation/pressure/velocity/tracer relative errors are {regimes[1]["worst_relative"]}. No new replay, independent confirmation, 600-output trajectory or qualified latency is established by this review.',
            lesson='A very small global affine correction can lower the compressed training loss without repairing reduced-integration field errors. Passing unchanged N1 outputs cannot establish accuracy on N8 cases. Three/four internal stages are not three/four future physical outputs: the runner repeats the endpoint computation sequentially for every K output, feeding its own generated state forward. Training factors describe F64 affine fields before final F32 casts, so deployed fields require separate scoring.',
            reproduction='Use the exact captured four-source trainer/configuration, static manifest/normalization/runtime/design shard and initial/final typed NPZ artifacts bound here. prepare_macrostep_loss_view_v1.py constructs full-grid QR loss factors from native training fields and the frozen managed backbone; no reference pressure solver runs during deployed prediction. macrostep_student_v1.py restores the learned readout then rounds every coefficient to 2^-40 before GPU evaluation. Small design/output artifacts, selected generation/execution sources and restore parity were freshly checked. The 592 native field archives and complete 1,698-member transitive generation closure were not reread. Clean retraining, new-Rheon regeneration and fresh complete-generator latency remain unqualified; CPU restore copies the original GPU probe without a model forward.')
        verified_review_receipt(review)
        changed, suffix = revised_card((PROJECT / review['note_path']).read_bytes(), review)
        if not changed.endswith(suffix):
            raise ValueError('Original retained card evidence changed')
        reviews.append(review)
    result = dict(schema='rheon.captured-model-method-reviews.v1', created_unix=time.time(),
        classification='authored_retrospective_interpretation_original_evidence_preserved',
        producer=binding(Path(__file__)), rows=reviews, overview_addition=OVERVIEW,
        training_view=binding(manifest_path), training_view_progress=binding(progress_path),
        active_shapes=sorted({tuple(r['counts_xyz']) for r in active.values()}),
        active_densities=dict(Counter(r['cells_per_metre'] for r in active.values())),
        native_field_archives_rehashed=0, new_neural_forwards=0, native_mutations=0,
        clean_reproduction_qualified=False, fresh_complete_generator_replay=False,
        scope='Exact small retained fit/design/output artifact and selected source review. Existing card consumer exercised. No captured source import, model forward, solver, SQL access or native mutation. Original evaluation outcomes remain historical; no performance or application drift acceptance.')
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status='two_macrostep_readout_source_reviews_verified', fits=[r['fit_item_id'] for r in reviews], native_adopted=False)))


if __name__ == '__main__':
    run()
