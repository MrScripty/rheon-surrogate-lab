"""Timed delegation; scientific execution and managed Runs keep their owners."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


LAB = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(argv, directory, name, cancel=None):
    """One fresh attempt, with retained timing even on failure or interruption."""
    directory = Path(directory)
    argv = list(map(str, argv))
    record = {'argv': argv, 'started_at_utc': utc(), 'status': 'running'}
    path = directory/(name+'-timing.json')
    if path.exists():
        raise FileExistsError('attempt already exists; reconcile rather than replay: '+str(path))
    write(path, record)
    started = time.monotonic()
    child = None
    try:
        with (directory/(name+'.log')).open('xb') as stream:
            child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            record['pid'] = child.pid
            write(path, record)
            while True:
                if cancel is not None and cancel.is_set():
                    raise KeyboardInterrupt
                try:
                    code = child.wait(timeout=.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
        record.update(returncode=code, status='completed' if code == 0 else 'failed')
        if code:
            raise subprocess.CalledProcessError(code, argv)
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=140)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        raise
    finally:
        record.update(finished_at_utc=utc(), elapsed_seconds=time.monotonic()-started,
                      child_reaped=child is not None and child.poll() is not None)
        if child is not None:
            record['returncode'] = child.returncode
        write(path, record)
    return record
