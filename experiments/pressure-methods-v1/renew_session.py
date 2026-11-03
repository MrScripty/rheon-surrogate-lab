"""Renew the authorized existing local owner through CLI login and HTTP exchange.

Never print credentials or database connection settings. No scientific jobs are
created here. Run once; a failed exchange leaves its private login file intact.
"""
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'runs/tuldok-performance-v1'
CATALOG = 'a7da4a8de53541a2a0e2aad08005bb71'


def main():
    deployment = json.loads((BASE / 'deployment.json').read_text())
    process = Path('/proc') / str(deployment['pid'])
    assert process.joinpath('stat').read_text().split()[21] == deployment['start_ticks']
    arguments = process.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
    source = ROOT / deployment['deployment']
    assert str(source / 'app.py') in arguments
    data = Path(arguments[arguments.index('--data') + 1])
    assert json.loads((data / '.tuldok.json').read_text())['dataset_id'] == CATALOG
    private = BASE / 'private'
    destination = private / 'owner-session.txt'
    assert destination.is_file() and not destination.is_symlink()
    stamp = str(time.time_ns())
    login_file = private / ('owner-login-' + stamp + '.txt')
    selection = json.loads((ROOT / 'runs/dependency-migration-v1/ops/live-selection.private.json').read_text())
    environment = dict(os.environ, TULDOK_DATABASE_URL=selection['dsn'], PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([str(ROOT / 'runs/dependency-migration-v1/app-env/bin/python3.12'),
                    '-B', str(source / 'access.py'), 'login', '--data', str(data),
                    '--out', str(login_file)], env=environment, check=True, timeout=45,
                   stdout=subprocess.DEVNULL)
    endpoint = 'http://127.0.0.1:8091'
    request = urllib.request.Request(endpoint + '/api/access/session',
        data=json.dumps({'token': login_file.read_text().strip()}).encode(),
        headers={'Content-Type': 'application/json', 'Origin': endpoint}, method='POST')
    with urllib.request.urlopen(request, timeout=20) as response:
        identity = json.load(response)
        cookie = SimpleCookie()
        cookie.load(response.headers['Set-Cookie'])
        token = cookie['tuldok_session'].value
    assert identity['kind'] == 'owner' and identity['catalog_id'] == CATALOG
    request = urllib.request.Request(endpoint + '/api/access/session',
        headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=20) as response:
        verified = json.load(response)
    assert verified == identity
    temporary = private / ('owner-session-' + stamp + '.pending')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(token + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(private, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    login_file.unlink()
    receipt = dict(status='renewed_existing_owner', catalog_id=CATALOG,
                   actor_id=identity['actor_id'], expires_at=identity['expires_at'],
                   method='documented local CLI login, HTTP session exchange and bearer verification',
                   private_mode=oct(destination.stat().st_mode & 0o777),
                   scientific_jobs_submitted=0, credential_values_recorded=False)
    (BASE / ('owner-renewal-' + stamp + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
