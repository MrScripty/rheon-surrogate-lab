"""One-time verified local activation; never launch competing catalog owners."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import psycopg
from psycopg import sql

root = Path.cwd()
work = root / 'runs/tuldok-performance-v1'
migration = root / 'runs/dependency-migration-v1'
source = work / 'Tuldok'
old = migration / 'Tuldok-deployed/current-06'
new = migration / 'Tuldok-deployed/current-07'
output = work / 'deployment.json'
catalog = 'a7da4a8de53541a2a0e2aad08005bb71'
selection = json.loads((migration / 'ops/live-selection.private.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def snapshot():
    with psycopg.connect(selection['dsn'], connect_timeout=5) as db:
        db.execute('SET TRANSACTION READ ONLY')
        db.execute("SET LOCAL statement_timeout='5s'")
        rows = db.execute(sql.SQL('SELECT id,document FROM {}.workspace_runs').format(
            sql.Identifier('catalog_' + catalog))).fetchall()
    return {key: {'state': json.loads(raw)['state'],
                  'document_sha256': hashlib.sha256(raw.encode()).hexdigest()}
            for key, raw in rows}


assert not new.exists() and not output.exists(), 'Fresh activation required'
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'])
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert revision == 'eed2b84d21050943d211defffc3133609422950a'
assert sha(old / 'deployment-manifest.json') == '3c9aa3daa616649e986457adf9ef7902d68c890ffb0e3833d33b0183b093d1f4'
manifest = json.loads((old / 'deployment-manifest.json').read_text())
changed = []
for item in manifest['files']:
    assert sha(old / item['path']) == item['sha256'], item['path']
    if sha(source / item['path']) != item['sha256']:
        changed.append(item['path'])
assert changed == ['runs.py'], changed
saved = snapshot()
assert all(row['state'] not in ('queued', 'running', 'cancelling', 'unknown') for row in saved.values())
pid = 855603
proc = Path('/proc') / str(pid)
assert proc.joinpath('stat').read_text().split()[21] == '10183125'
command = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
assert str(old / 'app.py') in command
environment = dict(item.split('=', 1) for item in proc.joinpath('environ').read_bytes().decode().strip('\0').split('\0') if '=' in item)
new.mkdir()
files = []
for item in manifest['files']:
    origin, destination = source / item['path'], new / item['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(origin.read_bytes())
    destination.chmod(origin.stat().st_mode & 0o777)
    files.append(dict(path=item['path'], bytes=destination.stat().st_size,
                      sha256=sha(destination), mode=destination.stat().st_mode & 0o777))
write(new / 'deployment-manifest.json', dict(format='tuldok-qualified-deployment-v1',
    source_revision=revision, files=files, predecessor_manifest_sha256=sha(old / 'deployment-manifest.json'),
    qualification='docs/plans/request-latency-v1/reports/regression-suite.json'))
receipt = dict(format='tuldok-request-latency-deployment-v1', status='stop_intent',
    source_revision=revision, old_pid=pid, old_start_ticks='10183125', catalog_id=catalog,
    existing_runs=len(saved), active_runs=0, old_run_snapshot=saved, changed_product_files=changed,
    deployment=str(new.relative_to(root)), manifest_sha256=sha(new / 'deployment-manifest.json'),
    automatic_retry=False)
write(output, receipt)
started = time.monotonic()
os.kill(pid, signal.SIGINT)
for _ in range(120):
    if not proc.exists() or proc.joinpath('stat').read_text().split()[2] == 'Z':
        break
    time.sleep(.5)
else:
    raise RuntimeError('Previous service did not cease; replacement not started')
receipt['status'] = 'launch_intent'
write(output, receipt)
command[command.index(str(old / 'app.py'))] = str(new / 'app.py')
with (work / 'tuldok-current-07.log').open('xb') as log:
    process = subprocess.Popen(command, env=environment, cwd=root, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt.update(pid=process.pid, status='health_pending')
write(output, receipt)
for _ in range(90):
    assert process.poll() is None, 'Replacement exited; inspect private operational log'
    try:
        with urllib.request.urlopen('http://127.0.0.1:8091/', timeout=2) as response:
            assert response.status == 200
        break
    except OSError:
        time.sleep(.5)
else:
    raise RuntimeError('Replacement health wait expired')
assert snapshot() == saved, 'Existing Run documents changed'
sys.path.insert(0, str(source))
import agent_api
client = agent_api.AgentClient('http://127.0.0.1:8091',
    agent_api._private_file(work / 'private/owner-session.txt', 4096).decode().strip(), catalog, timeout=15)
identity = client._request('GET', '/api/access/session')
context = client._request('GET', '/api/workspace/context')
assert identity['status'] == context['status'] == 200
assert identity['result']['catalog_id'] == context['result']['dataset_id'] == catalog
assert identity['result']['kind'] == 'owner'
receipt.update(status='passed', start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
    elapsed_s=time.monotonic()-started, existing_run_documents_preserved=True,
    original_owner_session_preserved=True, capacity_environment={key: environment[key] for key in (
        'TULDOK_CPU_RUN_SLOTS', 'TULDOK_CPU_RUN_MEMORY_MIB', 'TULDOK_CUDA_RUN_MEMORY_MIB', 'TULDOK_RUN_MEMORY_RESERVE_MIB')})
write(output, receipt)
print(json.dumps({key: receipt[key] for key in ('status', 'pid', 'source_revision', 'existing_runs', 'elapsed_s')}), flush=True)
