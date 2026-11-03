"""Recover a terminal receipt omitted from its failed controller index.

Leave all original artifacts unchanged. This checks custody and storage integrity;
independent numerical auditing remains required before using these labels.
"""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/full-ai-data-diversity-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    original = BASE / 'long-reference-pilot-01'
    output = BASE / 'long-reference-pilot-recovery-01'
    if output.exists():
        raise ValueError('Fresh recovery output required')
    failed_path = original / 'bank.json'
    failed = json.loads(failed_path.read_text())
    if (failed['status'] != 'failed' or failed['active'] is not None
            or failed['completed'] or failed['reference_rejections']
            or 'is not in the subpath' not in failed['error']):
        raise ValueError('Expected terminal indexing failure only')
    path = original / 'long-v1-continuous-stationary-r0-cpm8/receipt.json'
    receipt = json.loads(path.read_text())
    custody = receipt['native_process']
    if (receipt['status'] != 'completed' or receipt['actual_reference_updates'] != 4800
            or receipt['physical_duration_s'] != 20 or len(receipt['frames']) != 4800
            or not custody['reaped'] or custody['returncode'] != 0
            or not custody['graceful_protocol_complete']
            or receipt['protocol_sha256'] != failed['protocol_sha256']
            or receipt['source_sha256'] != failed['source_sha256']):
        raise ValueError('Incomplete receipt, source binding or native custody')
    for i, packet in enumerate(receipt['frames'], 1):
        frame = packet['frame']
        if (frame['step'] != i or not frame['advanced']
                or not frame['quality']['qualified'] or frame['quality']['hard_stop']):
            raise ValueError('Rejected or skipped reference update')
    for state in receipt['states']:
        source = path.parent / state['path']
        if source.stat().st_size != state['bytes'] or digest(source) != state['sha256']:
            raise ValueError('Incomplete or changed saved reference state')
    record = dict(identifier=receipt['identifier'], split=receipt['split'],
        cells_per_metre=receipt['scene']['cells_per_metre'], counts_xyz=receipt['scene']['counts'],
        receipt=str(path.relative_to(ROOT)), receipt_sha256=digest(path),
        fine_updates=4800, stored_states=len(receipt['states']),
        retained_state_bytes=receipt['retained_state_bytes'],
        elapsed_seconds=receipt['elapsed_seconds'], quality_maxima=receipt['quality_maxima'])
    bank = dict(status='completed', active=None, completed=[record], reference_rejections=[],
        recovered_from_failed_index=str(failed_path.relative_to(ROOT)),
        failed_bank_sha256=digest(failed_path), source_sha256=failed['source_sha256'],
        protocol_sha256=failed['protocol_sha256'], recovery_source_sha256=digest(Path(__file__)),
        original_capture_tool_session=62759, original_capture_exit_code=1,
        original_capture_failure_preserved=True, actual_new_reference_updates=0,
        new_parameter_fits=0, learned_forwards=0, cpu_model_forward=False,
        reserved52_references_opened=0, full_generator_target_pass=False,
        timing_status='resource_contention_operational_diagnostic_only',
        numerical_audit_required=True, created_unix=time.time())
    output.mkdir()
    (output / 'bank.json').write_text(json.dumps(bank, indent=2) + '\n')
    print('Recovered terminal receipt index; independent numerical audit remains required.')


if __name__ == '__main__':
    run()
