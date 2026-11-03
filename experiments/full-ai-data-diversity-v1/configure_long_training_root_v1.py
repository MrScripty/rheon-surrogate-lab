"""Add one prepared temporary-volume project root to the same idle local service.

Preserve the catalog, qualified application, endpoint and all existing roots.
Use the existing owner session, refuse active/uncertain work, and restore the
original command if the changed service cannot verify its original identity.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / 'runs/dependency-migration-v1/Tuldok-large-input-v3'
CATALOG = Path('/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres')
CATALOG_ID = 'a7da4a8de53541a2a0e2aad08005bb71'
STAGING_ROOT = Path('/tmp/rheon-shared-long-training-candidate-01')
OLD_PID = 3723727


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = ROOT / 'runs/full-ai-data-diversity-v1/managed-long-training-root-01'
    if output.exists() or not (STAGING_ROOT / 'fields.zip').is_file():
        raise ValueError('Fresh prepared root change required')
    argv = [v.decode() for v in Path(f'/proc/{OLD_PID}/cmdline').read_bytes().split(b'\0') if v]
    app_position = 2 if len(argv) > 2 and argv[1] == '-B' else 1
    if (len(argv) <= app_position or argv[app_position] != str(SERVICE / 'app.py') or str(CATALOG) not in argv
            or any(v.startswith('--') and v not in ('--data', '--host', '--port', '--project-root') for v in argv[app_position + 1:])):
        raise ValueError('Exact original local service command required')
    identity_before = Path(f'/proc/{OLD_PID}/stat').read_text().split(') ', 1)[1].split()[19]
    session = (ROOT / 'runs/tuldok-performance-v1/private/owner-session.txt').read_text().strip()
    headers = {'Authorization': 'Bearer ' + session}

    def identity(timeout=3):
        request = urllib.request.Request('http://127.0.0.1:8091/api/access/session', headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
        if result.get('kind') != 'owner' or result.get('catalog_id') != CATALOG_ID:
            raise ValueError('Original catalog owner identity required')
        return result

    identity()
    prefix = b'TULDOK_DATABASE_URL='
    setting = None
    with Path(f'/proc/{OLD_PID}/environ').open('rb') as stream:
        record = bytearray()
        while True:
            value = stream.read(1)
            if not value or value == b'\0':
                if record.startswith(prefix):
                    setting = bytes(record[len(prefix):]).decode()
                    break
                record.clear()
                if not value:
                    break
            else:
                record.extend(value)
    if setting is None:
        raise ValueError('Original configured local database unavailable')
    os.environ['TULDOK_DATABASE_URL'] = setting
    sys.path.insert(0, str(SERVICE))
    from metadata import Metadata
    db = Metadata(CATALOG, create=False)
    try:
        if db.dataset_id != CATALOG_ID:
            raise ValueError('Original catalog changed')
        active = db.execute("SELECT id FROM workspace_runs WHERE document::jsonb->>'state'=ANY(%s)",
            (['queued', 'running', 'cancelling', 'unknown'],)).fetchall()
        if active:
            raise ValueError('Active or uncertain jobs prevent service change')
    finally:
        db.close()
    if Path(f'/proc/{OLD_PID}/stat').read_text().split(') ', 1)[1].split()[19] != identity_before:
        raise ValueError('Service process identity changed')
    updated = [*argv, '--project-root', 'long-training=' + str(STAGING_ROOT)]
    output.mkdir(parents=True)
    intent = dict(status='prepared_idle_same_catalog_project_root_change',
        created_unix=time.time(), old_pid=OLD_PID, old_start_ticks=identity_before,
        original_command=argv, new_command=updated, active_or_uncertain_jobs=0,
        catalog_id=CATALOG_ID, service_source_sha256=sha(SERVICE / 'app.py'),
        source_sha256=sha(__file__), source_or_catalog_files_changed=False,
        original_project_roots_preserved=True, staging_root=str(STAGING_ROOT),
        alternate_credentials_read=False, training_runs_submitted=0)
    (output / 'intent.json').write_text(json.dumps(intent, indent=2) + '\n')
    log = (output / 'service.log').open('xb')
    os.chmod(output / 'service.log', 0o600)
    child = None
    try:
        os.kill(OLD_PID, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            process = Path(f'/proc/{OLD_PID}/stat')
            if not process.exists() or process.read_text().split(') ', 1)[1].split()[0] == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Original service did not cease; no replacement started')
        child = subprocess.Popen(updated, cwd=SERVICE, env=os.environ.copy(),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (output / 'active.json').write_text(json.dumps(dict(new_pid=child.pid, command=updated)) + '\n')
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError('Replacement service exited before identity verification')
            try:
                verified = identity(timeout=2)
                command = Path(f'/proc/{child.pid}/cmdline').read_bytes().split(b'\0')
                if ('long-training=' + str(STAGING_ROOT)).encode() not in command:
                    raise RuntimeError('New configured root absent from owned process')
                receipt = dict(intent)
                receipt.update(status='verified_idle_same_catalog_service_with_temporary_training_root',
                    new_pid=child.pid, owner_kind=verified['kind'], saved_session_still_valid=True,
                    active_jobs_interrupted=0, original_service_restarted=True)
                (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
                print('Verified same-catalog owner service with prepared temporary training root; no active jobs interrupted.', flush=True)
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(.5)
        raise TimeoutError('Replacement readiness deadline; restore original command')
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=15)
        # Restore only after confirming the original process is actually gone.
        process = Path(f'/proc/{OLD_PID}/stat')
        if not process.exists() or process.read_text().split(') ', 1)[1].split()[0] == 'Z':
            rollback = subprocess.Popen(argv, cwd=SERVICE, env=os.environ.copy(),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (output / 'rollback.json').write_text(json.dumps(dict(pid=rollback.pid,
                original_command_restored=True, error_type=type(error).__name__)) + '\n')
        raise
    finally:
        log.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Prepared local service root change did not complete:', type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
