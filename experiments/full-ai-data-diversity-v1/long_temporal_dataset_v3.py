"""Require complete reservation and explicit arithmetic-policy eligibility.

The earlier exploratory index remains usable for diagnostics. Training through
this interface requires all36 independent masters and the declared evidence
for every finite-F32 recovery; an audited candidate is not silently adopted.
"""
import json
from pathlib import Path

from long_temporal_dataset_v1 import ROOT, build_index, digest
from long_temporal_dataset_v2 import LongTemporalDataset as Parent
from reserve_long_scenes_v1 import instantiate


def qualified_training_index(audit_paths, reservation_path, eligibility_path):
    reservation_path = Path(reservation_path)
    eligibility_path = Path(eligibility_path)
    reservation = json.loads(reservation_path.read_text())
    eligibility = json.loads(eligibility_path.read_text())
    if (eligibility['status'] != 'qualified_selected240_reference_epochs_for_full36_training'
            or eligibility['reservation_sha256'] != digest(reservation_path)):
        raise ValueError('Explicit qualified reference-policy decision required')
    for name, expected in eligibility['evidence_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Reference-policy eligibility evidence changed: ' + name)
    index = build_index(audit_paths)
    rows = {r['identifier']: r for r in reservation['rows']}
    if (len(rows) != 36 or set(index['masters']) != set(rows)
            or index['independent_masters_per_split'] != {'train': 24, 'development': 12}
            or set(eligibility['masters']) != set(rows)
            or index['audits'] != eligibility['audits_sha256']):
        raise ValueError('Exact full36 reservation, audited population and splits required')
    finite_recoveries = set()
    for identifier, master in index['masters'].items():
        receipt_path = ROOT / master['receipt']
        receipt = json.loads(receipt_path.read_text())
        expected = eligibility['masters'][identifier]
        row = rows[identifier]
        if (master['split'] != row['split']
                or master['physical_scenario_group_id'] != row['physical_scenario_group_id']
                or expected['receipt_sha256'] != digest(receipt_path)
                or receipt['scene'] != instantiate(row)
                or receipt['reference_dt_s'] != 1 / 240
                or receipt['actual_reference_updates'] != 4800
                or receipt['physical_duration_s'] != 20):
            raise ValueError('Original physical scene, complete clock or receipt identity differs')
        epoch = receipt.get('reference_numerical_contract', 'original normal-only F32')
        if epoch != expected['reference_numerical_contract']:
            raise ValueError('Reference arithmetic epoch differs from eligibility decision')
        if epoch == 'experimental finite IEEE F32':
            finite_recoveries.add(identifier)
            evidence_path = ROOT / expected['GPU_compatibility_evidence']
            if str(evidence_path.relative_to(ROOT)) not in eligibility['evidence_sha256']:
                raise ValueError('Finite recovery compatibility evidence is unbound')
            evidence = json.loads(evidence_path.read_text())
            covered = {r['identifier'] for r in evidence['rows'] if r['all_accuracy_pass']}
            actual_cases = {(r['start_fine_step'], r['output_steps_per_s'], r['K'], r['lead'])
                            for r in evidence['rows'] if r['identifier'] == identifier}
            expected_cases = {(s, rate, k, lead) for s in (0, 2880, 4768)
                              for rate in (30, 60, 120) for k in (1, 2, 3, 4)
                              for lead in range(1, k + 1)}
            if (evidence['status'] != 'passed_finite_reference_unchanged_GPU_model_compatibility'
                    or evidence['failures'] or identifier not in covered
                    or actual_cases != expected_cases):
                raise ValueError('Recovered scene lacks unchanged GPU baseline compatibility')
        elif epoch != 'original normal-only F32':
            raise ValueError('Unqualified reference arithmetic contract')
    if finite_recoveries != set(eligibility['finite_f32_recovered_identifiers']):
        raise ValueError('Finite-F32 candidate population differs')
    index.update(schema='rheon.long-temporal-qualified-views.v3',
        training_ready_full36=True,
        reservation=str(reservation_path.resolve().relative_to(ROOT)),
        reservation_sha256=digest(reservation_path),
        reference_eligibility=str(eligibility_path.resolve().relative_to(ROOT)),
        reference_eligibility_sha256=digest(eligibility_path),
        reference_epochs={n: v['reference_numerical_contract']
                          for n, v in eligibility['masters'].items()})
    return index


class LongTemporalDataset(Parent):
    def __init__(self, index):
        if index.get('training_ready_full36') is not True:
            raise ValueError('Full36 training eligibility is not established')
        verified = qualified_training_index([ROOT / p for p in index['audits']],
            ROOT / index['reservation'], ROOT / index['reference_eligibility'])
        if index != verified:
            raise ValueError('Qualified training index changed')
        super().__init__(verified)
