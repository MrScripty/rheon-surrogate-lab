"""One root-issued supported CLI registration and verification; no retries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/dependency-migration-v1'
OPS = BASE / 'ops'
receipt = OPS / 'm3-runtime-registration.json'
assert not receipt.exists()
launch = json.loads((OPS / 'm3-app-launch.json').read_bytes())
assert launch['status'] == 'current_service_launched_smoke_pending'
assert launch['catalog_id'] == 'a7da4a8de53541a2a0e2aad08005bb71'
assert launch['deployment_manifest_sha256'] == '09400b61cee86dde0014a71ade0a1d8e4f80372684b53950cd9a1cae31873c5a'
manifest_sha = 'b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682'
prepared = json.loads((BASE / 'lab/docs/plans/rheon-tuldok-migration-v1/reports/torch-runtime-post-cutover-registration.json').read_bytes())
for command in ('register', 'verify'):
    argv = prepared[command + '_argv']
    assert argv[:3] == [str(BASE / 'app-env/bin/python'), '-B', str(BASE / 'Tuldok-deployed/current-03/torch_runtimes.py')]
    assert argv[3] == command
assert hashlib.sha256((BASE / 'runtime/current-manifest-v2.json').read_bytes()).hexdigest() == manifest_sha
env = {k: v for k, v in os.environ.items() if k in ('HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TMPDIR')}
env.update(PATH=str(BASE / 'app-env/bin') + ':/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1',
           TULDOK_DATABASE_URL=json.loads((OPS / 'live-selection.private.json').read_bytes())['dsn'])
record = {'schema': 'rheon-post-cutover-runtime-registration-v1', 'status': 'intent',
          'catalog_id': launch['catalog_id'], 'app_launch_sha256': hashlib.sha256((OPS / 'm3-app-launch.json').read_bytes()).hexdigest(),
          'runtime_manifest_sha256': manifest_sha, 'commands': [], 'optimizer_updates': 0,
          'automatic_retry': False, 'authentication': 'Supported CLI existing-owner login/exchange/bind/logout; legitimate authentication and registry audit writes expected.'}

def persist():
    temporary = receipt.with_suffix('.pending')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, receipt)

with receipt.open('x') as stream:
    json.dump(record, stream)
started = time.monotonic()
try:
    for command in ('register', 'verify'):
        argv = prepared[command + '_argv']
        record['status'] = command + '_dispatch_intent'
        record['commands'].append({'argv': argv, 'status': 'intent'})
        persist()
        before = time.monotonic()
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=120)
        log = OPS / ('m3-runtime-' + command + '.private.log')
        with os.fdopen(os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as stream:
            stream.write(result.stdout); stream.write(result.stderr)
        assert result.returncode == 0, 'Supported CLI failed; inspect private log and durable state before further action'
        value = json.loads(result.stdout)
        assert value['catalog_id'] == launch['catalog_id'] and value['manifest_sha256'] == manifest_sha
        assert value['profile'] == 'managed_torch_python_v1' and value['file_count'] == 25765
        record['commands'][-1].update(status='acknowledged', elapsed_s=time.monotonic() - before, result=value)
        persist()
    record['status'] = 'passed'
except BaseException:
    record['status'] = 'stopped_inspect_before_further_action'
    raise
finally:
    record['elapsed_s'] = time.monotonic() - started
    persist()
print(json.dumps({'status': record['status'], 'receipt': str(receipt), 'elapsed_s': record['elapsed_s']}))
