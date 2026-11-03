"""Account for parameter states outside the train-receipt output denominator.

One-time portfolio migration audit. Read JSON/text and hash one retained file
per byte identity; never deserialize parameters or execute models. Curated roles
below are source-review findings, not automatic model identities or acceptance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = 'runs/model-portfolio-migration-v1/discovery/'
NEXT = 'runs/full-ai-next-priorities-v1/'
DIVERSITY = 'runs/full-ai-data-diversity-v1/'
FAILED = ('runs/dependency-migration-v1/catalog-current-cold-04/workspace-runs/'
          '.external-2500fc7690c14049876250d8c3f5a981/')


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return digest


def context_notes():
    notes = {}

    def add(prefix, role, evidence, claim=None, note=None):
        if prefix in notes:
            raise ValueError('Duplicate curated context')
        notes[prefix] = dict(role=role, evidence=evidence, weight_claim=claim,
                             interpretation=note, benchmark_status='contract_review_pending')

    for prefix, suffix in (('08375747d6b8', ''), ('151037b35490', '-s2'), ('12bf7e491478', '-s3')):
        add(prefix, 'routed_parameter_assembly',
            [NEXT + 'a-head-calibration' + suffix + '-routed-assembly-01/assembly-receipt.json'],
            [0, ['outputs', 'weights.npz', 'sha256']],
            'Zero new optimizer updates; parent_components and per-tensor origins are recorded. Benchmark the assembled routing separately from its trained parents.')
    add('72a86d234f04', 'displacement_viscosity_parameter_assembly',
        [NEXT + 'b2-displacement-viscosity-assembly-01/receipt.json'], [0, ['weights_sha256']],
        'Zero-update assembly of exact displacement and viscosity fits; no new full-state training Run.')
    add('1b651a98bb8e', 'exported_fine_pressure_component',
        [NEXT + 'b1-fixed-viscosity-project-01/expert-assets/pressure/receipt.json'],
        [0, ['files', 'fine-weights.npz', 'sha256']],
        'Derived a-tail8 export; receipt claims no optimizer updates. Full parent parameter equivalence still needs source/array review.')
    add('19436a4d421a', 'untrained_residual_control',
        [DIVERSITY + 'internal-residual-qualification-01/results.json'],
        [0, ['initial_untrained_weights', 'sha256']],
        'Seed 1729 untrained network used for feature/gradient/graph controls; not a trained scientific candidate.')
    for prefix, path in (
            ('971dbdbf04bf', 'runs/pressure-methods-v1/learned-coarse-initialization-01/initialization.json'),
            ('35c1964791bf', 'runs/pressure-methods-v2/shape-head-initialization-01/initialization.json')):
        add(prefix, 'derived_initialization', [path], [0, ['weights_sha256']],
            'Receipt records source_weights_sha256 and zero optimizer updates. Added branches/parameters make this distinct from its source state.')
    for prefix, step in (('8e637f950264', 100), ('b3dfa72428aa', 500),
                         ('ded8e7ad6871', 1000), ('8fa85d9ba0f9', 2000)):
        add(prefix, 'unpublished_failed_fit_snapshot',
            [BASE + 'run-reconciliation-02.json', FAILED + 'custody/stdout.log',
             FAILED + 'custody/stderr.log',
             FAILED + 'source/runs/water-inference-budget-v1/stability-combined-single-pass/project/config.json'],
            note=f'Checkpoint-{step} of archived failed CUDA Run 2500fc7690c14049876250d8c3f5a981. Later gradient-norm failure does not erase these earlier states; final training receipt/publication is absent. Step attribution uses retained filename and progress log.')
    add('d41bbfd9dbb3', 'recovery_last_evaluated_parameter_state',
        [DIVERSITY + 'shared-recovery-entrypoint-qualification-01/output/receipt.json'],
        [0, ['outputs', 'last-evaluated-weights.npz', 'sha256']],
        'Recovery reports original failed Run fe69c8cbcbe14f3b974ea70c1d5acc0e, two parameter fits and 1920 updates; recovery itself performs zero new updates. Do not count recovery as a new fit.')
    add('e241d2621d3d', 'shared_initial_and_qualification_state',
        [DIVERSITY + 'actual-shared-trainer-qualification-01/output/receipt.json',
         DIVERSITY + 'shared-recovery-entrypoint-qualification-01/output/receipt.json'],
        [0, ['outputs', 'initial-weights.npz', 'sha256']],
        'Same bytes are named initial-weights and weights by zero-update qualification/recovery. Those names alone do not prove a trained state; retain all recorded roles.')
    add('d86302c2d870', 'exported_sparse_pressure_representation',
        [DIVERSITY + 'internal-residual-view-01/manifest.json',
         'experiments/full-ai-data-diversity-v1/prepare_internal_residual_view_v1.py'],
        [0, ['sparse_part', 'sha256']],
        'Preparation source exports restored pressure parameters into sparse arrays for the compact training view. No new fit is claimed; numerical parent equivalence remains a separate check.')
    add('e536905b0fdc', 'pretraining_package_conversion_probe',
        ['runs/pressure-spatial-coarse-v1/conversion-archive-01/package-conversion.json',
         'runs/pressure-spatial-coarse-v1/conversion-package-01/provenance.json'],
        [0, ['weights', 'sha256']],
        'Provenance explicitly says pretraining conversion probe, unchanged migration comparator, no training/promotion. Package receipt alone is not numerical qualification.')
    add('f97ffa3e083e', 'legacy_global_v1_fitted_checkpoint',
        ['experiments/final/freeze.json', 'experiments/final/README.md',
         'experiments/final/outcome.json'], [0, ['candidate_weights_sha256']],
        'Historical 335-parameter, 500-update candidate. Original final results are exposed; preserve no-post-test-tuning history. It is outside the later managed train-receipt schema.')
    add('25533a2fbaa5', 'legacy_modal_least_squares_fitted_checkpoint',
        ['experiments/trajectory-v2/freeze.json', 'scripts/trajectory_model.py',
         'experiments/trajectory-v2/frozen-source/trajectory_experiment.py'],
        [0, ['checkpoint_sha256']],
        '32-cubed modal response has 32767 supervised fitted gains. Closed-form least squares has zero optimizer steps but is training, not an untrained control.')
    add('fbacd5e53cd1', 'legacy_unfrozen_modal_checkpoint',
        ['scripts/trajectory_model.py', 'scripts/trajectory_experiment.py'],
        note='Earlier trajectory-v1 checkpoint and training.npz are retained. No adjacent final freeze/receipt was found. ZIP pickle opcodes identify modal schema and n=32 without deserialization; exact training-source/data binding still needs proof.')
    return notes


def run(output, summary):
    if output.exists() or summary.exists():
        raise ValueError('Fresh audit output paths required')
    paths = [ROOT / BASE / name for name in
             ('checkpoint-identities-01.json', 'model-worklist-05.json')]
    identities, worklist = (json.loads(p.read_text()) for p in paths)
    fit_outputs = {}
    for fit in worklist['fit_cards']:
        for checkpoint in fit['checkpoints']:
            fit_outputs.setdefault(checkpoint['sha256'], []).append(
                dict(fit_item_id=fit['item_id'], stage=checkpoint['stage'], name=checkpoint['name']))
    notes, used, rows = context_notes(), set(), []
    for group in identities['groups']:
        digest = group['sha256']
        path = next((ROOT / p for p in group['paths'] if (ROOT / p).is_file()), None)
        if path is None or sha(path) != digest or path.stat().st_size != group['bytes']:
            raise ValueError('Retained checkpoint identity unavailable or changed')
        row = dict(sha256=digest, bytes=group['bytes'], retained_path=str(path.relative_to(ROOT)),
                   recorded_alias_count=len(group['paths']), fit_outputs=fit_outputs.get(digest, []))
        if row['fit_outputs']:
            row.update(role='recorded_train_receipt_output', benchmark_status='contract_review_pending')
        else:
            matches = [p for p in notes if digest.startswith(p)]
            if len(matches) != 1:
                raise ValueError('Unmatched checkpoint requires explicit reviewed context')
            prefix = matches[0]
            used.add(prefix)
            row.update(notes[prefix])
            evidence = [ROOT / p for p in row['evidence']]
            if row['weight_claim']:
                index, selector = row['weight_claim']
                value = json.loads(evidence[index].read_text())
                for key in selector:
                    value = value[key]
                if value != digest:
                    raise ValueError('Context receipt contradicts checkpoint bytes')
            row['evidence'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in evidence]
            row['provenance_complete'] = False
        rows.append(row)
    if used != set(notes):
        raise ValueError('Curated context no longer corresponds to the input inventory')
    value = dict(schema='rheon.checkpoint-context-audit.v1', created_unix=time.time(),
                 status='all_known_byte_states_context_routed_not_complete_model_identity',
                 rows=rows, inputs=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths],
                 source_sha256=sha(Path(__file__)), inference_calls=0, payloads_deserialized=False,
                 complete_model_count_established=False, model_payloads_deleted=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + '\n')
    compact = {k: v for k, v in value.items() if k != 'rows'}
    compact.update(output=str(output), output_sha256=sha(output), byte_states=len(rows),
                   roles=dict(Counter(r['role'] for r in rows)),
                   non_train_receipt_contexts=[r for r in rows if not r['fit_outputs']],
                   limitations=['One current retained file verified per byte identity; aliases reuse the earlier copy audit.',
                                'A role is not a completed model card, source/data reproduction or applicable benchmark.',
                                '205 train-mode receipt drafts do not exhaust training attempts: failed/unpublished and least-squares fits remain represented separately.'])
    summary.write_text(json.dumps(compact, indent=2) + '\n')
    print(json.dumps({k: compact[k] for k in ('status', 'byte_states', 'roles')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.summary)
