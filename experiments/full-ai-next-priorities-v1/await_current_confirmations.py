"""Observe current confirmation owners; never replay fits or measurements."""
import json
import time
from study_support import ROOT, HERE, BASE, DOCS, read, write, sha


def verify_sources():
    registration = read(DOCS / 'final-publication-registration-01.json')
    for relative, digest in registration['source_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Registered publication source changed: ' + relative)
    return registration


if __name__ == '__main__':
    registration = verify_sources()
    deadline = time.monotonic() + 100000
    pending = list(registration['owners'])
    while pending:
        remaining = []
        for name in pending:
            path = BASE / name / 'execution.json'
            try:
                state = read(path)
            except json.JSONDecodeError:
                remaining.append(name)
                continue
            if state['status'] == 'running':
                remaining.append(name)
            elif state['status'] != 'completed' or not state['owned_child_reaped'] or state.get('active') is not None:
                raise RuntimeError('Existing confirmation owner stopped; inspect retained evidence: ' + name)
            elif any(c['returncode'] != 0 or not c['reaped'] for c in state['completed']):
                raise ValueError('Confirmation owner retains failed work: ' + name)
        pending = remaining
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError('Existing confirmation observation deadline; no replay')
            time.sleep(5)
    verify_sources()
    write(BASE / 'final-publication-admission-01.json', dict(status='admitted',
        source_sha256=sha(__file__), owners={name: sha(BASE / name / 'execution.json') for name in registration['owners']},
        registration_sha256=sha(DOCS / 'final-publication-registration-01.json'),
        training_or_measurements_replayed=False))
    print('All existing confirmation owners complete and reaped; publication admitted', flush=True)
