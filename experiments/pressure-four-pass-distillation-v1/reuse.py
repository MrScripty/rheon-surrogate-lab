"""Reuse verified public downloads while an owned readback controller is paused.

This is local content reuse, never private Run access or remote execution. The
first missing destination is excluded because an HTTP download may be in flight.
"""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import time

from common import ROOT, read, sha, write


def reuse(projects, arm, output):
    if output.exists():
        raise ValueError('Fresh reuse receipt required')
    directory = projects / (arm + '-study-01')
    training_id = read(directory / 'active.json')['training_run_id']
    terminals = [p for p in directory.glob('terminal-*.json') if read(p)['id'] != training_id]
    if len(terminals) != 1:
        raise ValueError('Exact owner API CPU-restore terminal readback required')
    restore = read(terminals[0])
    if restore['state'] != 'completed' or restore['custody_phase'] != 'drained':
        raise ValueError('Completed drained restore required')
    pids = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = [v.decode() for v in (process / 'cmdline').read_bytes().split(b'\0') if v]
        except (OSError, UnicodeError):
            continue
        if (any(v.endswith(('/manage.py', '/recover.py')) for v in args) and '--output' in args
                and Path(args[args.index('--output') + 1]).resolve() == directory):
            pids.append(int(process.name))
    if len(pids) != 1:
        raise ValueError('One exact owned readback controller required')
    receipt = dict(status='intent', arm=arm, pid=pids[0], reused=[],
        executed_source_sha256=sha(__file__),
        restore_run_id=restore['id'], terminal_sha256=sha(terminals[0]), remote_mutations=0,
        scope='Only previously owner-downloaded train bytes, under matching public CPU restore size and whole SHA256; no restore-specific receipt substitution')
    write(output, receipt)
    handle = os.pidfd_open(pids[0])
    paused = False
    started = time.perf_counter()
    try:
        signal.pidfd_send_signal(handle, signal.SIGSTOP)
        paused = True
        for _ in range(100):
            state = (Path('/proc') / str(pids[0]) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state in ('T', 't'):
                break
            time.sleep(.001)
        else:
            raise RuntimeError('Owned controller did not confirm its pause')
        target = directory / 'restore'
        target.mkdir(exist_ok=True)
        missing = [item for item in restore['artifacts'] if not (target / item['name']).exists()]
        receipt['skipped_possibly_inflight'] = missing[0]['name'] if missing else None
        for item in missing[1:]:
            source = directory / 'train' / item['name']
            if source.is_symlink() or not source.is_file() or source.stat().st_size != item['bytes']:
                continue
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                continue
            destination = target / item['name']
            with destination.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if sha(destination) != item['sha256']:
                raise ValueError('Local reused bytes differ')
            receipt['reused'].append(dict(name=item['name'], sha256=item['sha256'], bytes=len(raw),
                source=str(source.relative_to(ROOT)), destination=str(destination.relative_to(ROOT))))
            write(output, receipt)
        receipt['status'] = 'passed'
    except BaseException:
        receipt['status'] = 'failed'
        raise
    finally:
        if paused:
            signal.pidfd_send_signal(handle, signal.SIGCONT)
            receipt['controller_resumed'] = True
        os.close(handle)
        receipt['elapsed_s'] = time.perf_counter() - started
        write(output, receipt)
    print('Verified readback reuse', arm, len(receipt['reused']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projects', type=Path, required=True)
    parser.add_argument('--arm', choices=['two-pass-control', 'four-pass-physics', 'four-pass-teacher'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reuse(args.projects.resolve(), args.arm, args.output.resolve())
