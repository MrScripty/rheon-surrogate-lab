"""One explicit replacement of the exact owned read-only migration publisher."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/dependency-migration-v1'
old = json.loads((BASE / 'receipts/live-migration/watcher.json').read_bytes())
receipt = BASE / 'receipts/live-migration/watcher-02.json'
assert not receipt.exists()
pid = old['pid']
proc = Path('/proc') / str(pid)
assert (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19] == old['linux_start_ticks']
assert [a.decode() for a in (proc / 'cmdline').read_bytes().split(b'\0') if a] == old['command']
assert str((proc / 'cwd').resolve()) == str(ROOT)
assert not any((t / 'children').read_text().strip() for t in (proc / 'task').iterdir())
frame = BASE / 'live-migration/frame.json'
before = json.loads(frame.read_bytes())
start = time.monotonic()
descriptor = os.pidfd_open(pid)
try:
    assert (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19] == old['linux_start_ticks']
    signal.pidfd_send_signal(descriptor, signal.SIGTERM)
finally:
    os.close(descriptor)
for _ in range(100):
    if not proc.exists() or (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
        break
    time.sleep(0.1)
else:
    raise RuntimeError('Publisher has not exited; no replacement started')

def limits():
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2097152, 2097152))

env = {k: v for k, v in os.environ.items() if k in ('HOME', 'PATH', 'LANG', 'LC_ALL', 'TMPDIR')}
env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
log = BASE / 'receipts/live-migration/watcher-02.stderr.log'
with log.open('xb') as stream:
    child = subprocess.Popen(old['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=stream, stderr=stream, preexec_fn=limits, start_new_session=True)
time.sleep(3)
assert child.poll() is None, 'Replacement exited; inspect retained log'
after = json.loads(frame.read_bytes())
assert before['source'] == after['source'] and before['pressure'] == after['pressure']
activity = json.loads((BASE / 'live-migration/activity.json').read_bytes())
assert 'Current catalog rehearsal:' in activity['message']
value = {**old, 'pid': child.pid, 'linux_start_ticks': (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19],
         'previous_pid': pid, 'restart_reason': 'Committed current catalog and live-smoke observation',
         'source_commit': subprocess.check_output(['git', '-C', str(BASE / 'lab'), 'rev-parse', 'HEAD'], text=True).strip(),
         'source_sha256': hashlib.sha256(Path(old['command'][2]).read_bytes()).hexdigest(),
         'actual_frame_source_and_pressure_unchanged': True, 'current_status_visible': True,
         'elapsed_s': time.monotonic() - start, 'log': str(log)}
with receipt.open('x') as stream:
    json.dump(value, stream, indent=2)
print(json.dumps({'status': 'publisher_replaced', 'pid': child.pid, 'receipt': str(receipt), 'elapsed_s': value['elapsed_s']}))
