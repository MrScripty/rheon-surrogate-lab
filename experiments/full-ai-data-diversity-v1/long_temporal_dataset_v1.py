"""Stream audited complete-state temporal views without copying field archives.

No learned forward runs on CPU. Example metadata owns scene identity and sampling
weights; model inputs contain only current fields, physical timing/grid metadata
and prescribed controls. Future states and matching fine RHS live in labels.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from temporal_control_packets_v1 import control_packet

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('pressure', 'vx', 'vy', 'vz', 'tracer')
SUPERVISION = ('rhs', 'native_input_residual', 'pressure_base')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(audit_paths):
    masters = {}
    audits = {}
    for audit_path in audit_paths:
        audit_path = Path(audit_path).resolve()
        audit = json.loads(audit_path.read_text())
        if audit['status'] != 'passed_completed_reference_labels_windows_and_late_replay':
            raise ValueError('Numerically audited completed master required')
        audits[str(audit_path.relative_to(ROOT))] = digest(audit_path)
        for case in audit['cases']:
            identifier = case['identifier']
            if identifier in masters:
                raise ValueError('Duplicate physical master across supplied audits')
            # Locate the exact bound terminal bank, not an unfinished writer.
            candidates = [ROOT / name for name in audit['source_sha256']
                          if name.endswith('/bank.json')]
            matching_banks = [p for p in candidates if digest(p) == audit['bank_sha256']]
            if len(matching_banks) != 1:
                raise ValueError('Exactly one immutable audited terminal bank required')
            bank = json.loads(matching_banks[0].read_text())
            records = [r for r in bank['completed'] if r['identifier'] == identifier]
            if len(records) != 1:
                raise ValueError('Audited case absent from terminal bank')
            record = records[0]
            receipt_path = ROOT / record['receipt']
            if digest(receipt_path) != record['receipt_sha256']:
                raise ValueError('Audited receipt changed')
            receipt = json.loads(receipt_path.read_text())
            if (receipt['split'] not in ('train', 'development')
                    or receipt['reference_rate_per_s'] != 240
                    or receipt['actual_reference_updates'] != 4800):
                raise ValueError('Complete nonreserved240/s master required')
            masters[identifier] = dict(receipt=record['receipt'],
                receipt_sha256=record['receipt_sha256'],
                physical_scenario_group_id=receipt['physical_scenario_group_id'],
                split=receipt['split'], windows=receipt['windows'])
    if not masters:
        raise ValueError('No audited master supplied')
    groups = {}
    for m in masters.values():
        g = m['physical_scenario_group_id']
        if g in groups:
            raise ValueError('Related realizations require explicit group-level weighting')
        groups[g] = m['split']
    split_counts = Counter(groups.values())
    examples = []
    for identifier, m in masters.items():
        starts = {w['start_fine_step'] for w in m['windows']}
        expected = {(s, rate, k) for s in starts for rate in (30, 60, 120) for k in (1, 2, 3, 4)}
        actual = {(w['start_fine_step'], w['output_steps_per_s'], w['K']) for w in m['windows']}
        if actual != expected or len(m['windows']) != len(expected):
            raise ValueError('Every selected start requires all12 cases')
        for w in m['windows']:
            rate, k, start = w['output_steps_per_s'], w['K'], w['start_fine_step']
            targets = [start + j * (240 // rate) for j in range(1, k + 1)]
            if targets != w['target_fine_steps']:
                raise ValueError('Targets do not spanKfull physical output intervals')
            examples.append(dict(identifier=identifier, split=m['split'],
                physical_scenario_group_id=m['physical_scenario_group_id'],
                start_fine_step=start, output_steps_per_s=rate, K=k,
                target_fine_steps=targets,
                example_weight=1 / (split_counts[m['split']] * len(starts) * 12),
                within_example_output_weight=1 / k))
    return dict(schema='rheon.long-temporal-views.v1', masters=masters, examples=examples,
        audits=audits, independent_masters_per_split=dict(split_counts),
        source_reference_rate_per_s=240, future_reference_fields_are_labels_only=True,
        additional_history_contract='Current nativeP,V,T is the Markov input for this new temporal model interface. Extra learned history is not supplied; old model histories are not implicitly qualified.',
        training_ready_full36=False, reserved52_references_opened=0)


class LongTemporalDataset:
    def __init__(self, index):
        self.index = deepcopy(index)
        self.receipts = {}
        for name, expected in index['audits'].items():
            if digest(ROOT / name) != expected:
                raise ValueError('Numerical audit changed')
        for identifier, m in index['masters'].items():
            path = ROOT / m['receipt']
            if digest(path) != m['receipt_sha256']:
                raise ValueError('Frozen receipt changed')
            receipt = json.loads(path.read_text())
            self.receipts[identifier] = (path, receipt,
                {s['fine_step']: s for s in receipt['states']})

    def _load(self, identifier, step):
        path, _, states = self.receipts[identifier]
        spec = states[step]
        source = path.parent / spec['path']
        if digest(source) != spec['sha256']:
            raise ValueError('Saved native state changed')
        with np.load(source, allow_pickle=False) as archive:
            values = {n: archive[n].copy() for n in archive.files}
        if set(values) != set((*FIELDS, *SUPERVISION)):
            raise ValueError('Complete native state and supervision required')
        for name, array in values.items():
            expected = spec['arrays'][name]
            if (array.dtype.str != expected['dtype'] or list(array.shape) != expected['shape']
                    or hashlib.sha256(array.tobytes()).hexdigest() != expected['sha256']
                    or not np.isfinite(array).all()):
                raise ValueError('Typed native label differs: ' + name)
        return values

    def inputs(self, example):
        _, receipt, _ = self.receipts[example['identifier']]
        start, rate, k = example['start_fine_step'], example['output_steps_per_s'], example['K']
        if rate not in (30, 60, 120) or k not in (1, 2, 3, 4):
            raise ValueError('Registered cadence/K required')
        state = self._load(example['identifier'], start)
        scene = receipt['scene']
        return dict(current={n: state[n] for n in FIELDS}, counts_xyz=tuple(scene['counts']),
            cell_size_m=1 / scene['cells_per_metre'], rho_kg_per_m3=1000.,
            start_time_s=start / 240, output_dt_s=1 / rate, K=k,
            block_duration_s=k / rate,
            lead_elapsed_s=np.arange(1, k + 1, dtype='<f8') / rate,
            normalized_phase=np.arange(1, k + 1, dtype='<f8') / k,
            reference_dt_s=1 / 240,
            known_force_packet=control_packet(scene, start, k * (240 // rate)))

    def labels(self, example):
        rate, k, start = example['output_steps_per_s'], example['K'], example['start_fine_step']
        expected = [start + j * (240 // rate) for j in range(1, k + 1)]
        if expected != example['target_fine_steps']:
            raise ValueError('Explicit target indices differ from physical lead times')
        outputs = [self._load(example['identifier'], step) for step in expected]
        return dict(outputs=[{n: a[n] for n in FIELDS} for a in outputs],
            endpoint={n: outputs[-1][n] for n in FIELDS},
            fine_pressure_supervision=[{n: a[n] for n in SUPERVISION} for a in outputs],
            pressure_rhs_dt_s=1 / 240,
            note='RHS matches the last fine pressure solve at each endpoint; it is not a coarse outputdt RHS and is never a model input.')
