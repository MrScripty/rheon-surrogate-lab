"""Freeze completed receipt records while their parent capture continues.

The completed subset is its own immutable index, not a declaration that the
parent controller finished. Native custody and terminal receipts are required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(args):
    if args.output.exists():
        raise ValueError('Fresh completed-subset snapshot required')
    raw = args.bank.read_bytes()
    observed = json.loads(raw)
    if observed['status'] not in ('running', 'completed', 'completed_with_reference_rejections', 'failed'):
        raise ValueError('Expected active or successfully terminal parent capture')
    if len(set(args.identifiers)) != len(args.identifiers):
        raise ValueError('Duplicate requested scenario')
    records = [r for r in observed['completed'] if r['identifier'] in args.identifiers]
    if len(records) != len(args.identifiers):
        raise ValueError('Every requested master must already have a terminal record')
    for record in records:
        path = ROOT / record['receipt']
        if digest(path) != record['receipt_sha256']:
            raise ValueError('Completed receipt identity differs')
        receipt = json.loads(path.read_text())
        custody = receipt['native_process']
        if (receipt['status'] != 'completed' or receipt['actual_reference_updates'] != 4800
                or receipt['physical_duration_s'] != 20 or len(receipt['frames']) != 4800
                or not custody['reaped'] or custody['returncode'] != 0
                or not custody['graceful_protocol_complete']
                or receipt['source_sha256'] != observed['source_sha256']
                or receipt['protocol_sha256'] != observed['protocol_sha256']):
            raise ValueError('Full terminal receipt and native custody required')
    args.output.mkdir(parents=True)
    (args.output / 'parent-bank-as-observed.json').write_bytes(raw)
    subset = dict(status='completed', active=None, completed=records, reference_rejections=[],
        scope='Only the explicitly selected completed master receipts are indexed here.',
        entire_parent_capture_complete=observed['status'] in ('completed', 'completed_with_reference_rejections'),
        parent_capture_terminal_failure=observed['status'] == 'failed',
        parent_bank=str(args.bank.resolve().relative_to(ROOT)),
        parent_bank_observation_sha256=hashlib.sha256(raw).hexdigest(),
        parent_status_when_observed=observed['status'], parent_active_when_observed=observed['active'],
        source_sha256=observed['source_sha256'], protocol_sha256=observed['protocol_sha256'],
        snapshot_source_sha256=digest(Path(__file__)), created_unix=time.time(),
        numerical_audit_required=True, actual_new_reference_updates=0,
        new_parameter_fits=0, learned_forwards=0, cpu_model_forward=False,
        reserved52_references_opened=0, full_generator_target_pass=False,
        timing_status='resource_contention_operational_diagnostic_only')
    write(args.output / 'bank.json', subset)
    print('Frozen completed subset:', ', '.join(args.identifiers), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--identifiers', nargs='+', required=True)
    run(parser.parse_args())
