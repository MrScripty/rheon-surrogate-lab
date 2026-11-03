"""Explicit root-triggered command for the sole structural-cutover pressure smoke.

This only invokes the qualified lab client. It implements no numerical worker,
runtime registration, catalog owner, retry or recovery path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
MIGRATION = ROOT / 'runs/dependency-migration-v1'
OPS = MIGRATION / 'ops'
PYTHON = MIGRATION / 'app-env/bin/python'
DRIVER = MIGRATION / 'lab/experiments/dependency-migration-v1/qualify_typed_pressure.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-authorized-smoke', action='store_true', required=True)
    parser.add_argument('--session-file', type=Path, default=OPS / 'migration-owner-session.private.txt')
    args = parser.parse_args()
    gate = OPS / 'm3-cutover.json'
    launch = OPS / 'm3-app-launch.json'
    prepared = MIGRATION / 'typed-pressure-live-source-01'
    output = MIGRATION / 'typed-pressure-live-01'
    launcher_receipt = OPS / 'typed-pressure-smoke-launcher.json'
    if any(path.exists() for path in (prepared, output, launcher_receipt)):
        raise ValueError('Prior smoke command/output exists; inspect durable state and do not replay.')
    value = json.loads(gate.read_bytes())
    if not (value.get('schema') == 'rheon-tuldok-migration-cutover-v1'
            and value.get('m3_service_switched') is True and value.get('m3_complete') is False
            and value.get('ordinary_admissions_open') is False
            and value.get('smoke_authorized') == {'kind': 'typed_pressure_one_update_cpu_restore', 'max_optimizer_updates': 1}
            and value.get('endpoint') == 'http://127.0.0.1:8091'
            and value.get('catalog_id') == 'a7da4a8de53541a2a0e2aad08005bb71'):
        raise ValueError('Exact root-issued structural cutover/single smoke authorization required.')
    launch_sha, gate_sha = sha(launch), sha(gate)
    common = [str(PYTHON), '-B', str(DRIVER), '--project-root', str(ROOT), '--project-alias', 'pressure',
        '--assembly', str(MIGRATION / 'typed-pressure-project-02/assembly.json'),
        '--tuldok-root', str(MIGRATION / 'Tuldok'), '--endpoint', value['endpoint'], '--catalog-id', value['catalog_id']]
    prepare_command = [*common, '--prepare-only', '--output', str(prepared)]
    started = time.monotonic()
    record = {'format': 'rheon_typed_pressure_smoke_launcher_v1', 'status': 'source_verification_started',
        'm3_sha256': gate_sha, 'app_launch_sha256': launch_sha, 'commands': [prepare_command],
        'recovery': 'Inspect qualification journal/durable objects; no automatic mutation retries.'}
    with launcher_receipt.open('x') as stream:
        json.dump(record, stream, indent=2)
    try:
        subprocess.run(prepare_command, check=True)
        receipt = prepared / 'receipt.json'
        execute_command = [*common, '--execute-after-m3', '--catalog-root',
            '/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres',
            '--session-file', str(args.session_file.resolve(strict=True)),
            '--prepared-receipt', str(receipt), '--prepared-sha256', sha(receipt),
            '--m3-receipt', str(gate), '--m3-sha256', gate_sha,
            '--app-launch-receipt', str(launch), '--app-launch-sha256', launch_sha, '--output', str(output)]
        record.update(status='qualified_client_dispatch', commands=[prepare_command, execute_command])
        launcher_receipt.write_text(json.dumps(record, indent=2) + '\n')
        subprocess.run(execute_command, check=True)
        record.update(status='machine_completed_browser_pending', receipt=str(output / 'receipt.json'))
    except BaseException:
        record['status'] = 'stopped_inspect_before_further_action'
        raise
    finally:
        record['elapsed_s'] = time.monotonic() - started
        launcher_receipt.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
