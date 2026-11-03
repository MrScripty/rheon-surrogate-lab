"""Seal explicit selected240 reference-epoch eligibility for the original36scenes."""
import argparse
from collections import Counter
from pathlib import Path
import time

from support import ROOT, read, write, digest
from long_temporal_dataset_v1 import build_index
from long_temporal_dataset_v3 import qualified_training_index, LongTemporalDataset


def run(args):
    reg = read(args.protocol)
    final = ROOT / reg['eligibility_output']
    if args.output.exists() or final.exists():
        raise ValueError('Fresh explicit reference eligibility required')
    for name, expected in reg['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen full36 eligibility evidence changed: ' + name)
    index = build_index([ROOT / p for p in reg['audits']])
    reservation = ROOT / reg['reservation']
    expected_rows = {r['identifier']: r for r in read(reservation)['rows']}
    if set(index['masters']) != set(expected_rows) or len(index['masters']) != 36:
        raise ValueError('Original complete36scene reservation required')
    compatibility = {}
    for name in reg['compatibility_evidence']:
        result = read(ROOT / name)
        if result['status'] != 'passed_finite_reference_unchanged_GPU_model_compatibility' or result['failures']:
            raise ValueError('Every recovered reference needs unchanged GPU compatibility')
        identifiers = {r['identifier'] for r in result['rows']}
        if len(identifiers) != 1:
            raise ValueError('One explicitly identified recovery per compatibility receipt')
        compatibility[next(iter(identifiers))] = name
    rejected = {}
    for name in reg['original_rejected_receipts']:
        receipt = read(ROOT / name)
        if (receipt['status'] != 'reference_rejected' or not receipt['native_process']['reaped']
                or receipt['native_process']['returncode'] != 1):
            raise ValueError('Preserved original native rejection custody required')
        rejected[receipt['identifier']] = name
    if set(rejected) != set(compatibility) or len(rejected) != 4:
        raise ValueError('All four original rejections must be recovered, not omitted')
    masters = {}
    density = Counter()
    for identifier, master in index['masters'].items():
        receipt = read(ROOT / master['receipt'])
        if (receipt['actual_reference_updates'] != 4800 or len(receipt['frames']) != 4800
                or not all(f['frame']['advanced'] and f['frame']['quality']['qualified'] for f in receipt['frames'])
                or receipt['reference_dt_s'] != 1 / 240 or receipt['physical_duration_s'] != 20
                or not receipt['native_process']['reaped'] or receipt['native_process']['returncode'] != 0):
            raise ValueError('Full uninterrupted qualified native master required')
        epoch = receipt.get('reference_numerical_contract', 'original normal-only F32')
        spec = dict(receipt_sha256=master['receipt_sha256'], reference_numerical_contract=epoch,
            split=master['split'], physical_scenario_group_id=master['physical_scenario_group_id'])
        if identifier in compatibility:
            if epoch != 'experimental finite IEEE F32':
                raise ValueError('Recovered native reference arithmetic epoch untagged')
            spec.update(GPU_compatibility_evidence=compatibility[identifier],
                preserved_original_rejected_receipt=rejected[identifier])
        masters[identifier] = spec
        density[receipt['scene']['cells_per_metre']] += 1
    evidence = {name: digest(ROOT / name) for name in
        [*reg['audits'], *reg['compatibility_evidence'], *reg['original_rejected_receipts'],
         *reg['reference_policy_evidence']]}
    eligibility = dict(status='qualified_selected240_reference_epochs_for_full36_training',
        created_unix=time.time(), reservation_sha256=digest(reservation),
        audits_sha256=index['audits'], evidence_sha256=evidence, masters=masters,
        finite_f32_recovered_identifiers=sorted(compatibility),
        decision='Adopt32originalnormal-onlyF32masters and4explicitfiniteIEEEF32recoveries for training/development. OriginalsceneIDs,initialrecipes,controls,240clock and20sdurations are retained; every master passed native array/RHS/all12window/restart audits. Each changed-epoch recovery passed the unchanged accurate GPU baseline on all12cases at0,2880,4768.',
        physical_quality_gates_changed=False, float_representability_policy_changed=True,
        explicit_subnormal_flushing=False, temporal_convergence_proven=False,
        reference_pressure_semantics='Projection from final fine interval, with left-sampled240prescribed forcing; exact event metadata does not retroactively change these labels.',
        independent_masters_per_split=index['independent_masters_per_split'],
        density_counts=dict(density), source_sha256=reg['source_sha256'],
        new_parameter_fits=0, reserved52_references_opened=0, full_generator_target_pass=False)
    args.output.mkdir(parents=True)
    provisional = args.output / 'provisional-eligibility.json'
    write(provisional, eligibility)
    qualified_training_index([ROOT / p for p in reg['audits']], reservation, provisional)
    write(final, eligibility)
    ready = qualified_training_index([ROOT / p for p in reg['audits']], reservation, final)
    dataset = LongTemporalDataset(ready)
    if len(dataset.receipts) != 36 or not ready['training_ready_full36']:
        raise ValueError('Actual qualified interface did not admit complete36scene set')
    write(args.output / 'index.json', ready)
    write(args.output / 'results.json', dict(status='passed_complete36_selected240_reference_eligibility_and_training_interface',
        source_sha256=reg['source_sha256'], protocol_sha256=digest(args.protocol),
        eligibility=str(final.relative_to(ROOT)), eligibility_sha256=digest(final),
        index_sha256=digest(args.output / 'index.json'), independent_masters_per_split=ready['independent_masters_per_split'],
        density_counts=dict(density), original_reference_masters=32, explicit_finite_F32_recoveries=4,
        all12_temporal_configurations=True, complete_examples=len(ready['examples']),
        reference_updates=36 * 4800, training_ready_full36=True,
        temporal_convergence_proven=False, new_parameter_fits=0, cpu_model_forward=False,
        reserved52_references_opened=0, full_generator_target_pass=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
