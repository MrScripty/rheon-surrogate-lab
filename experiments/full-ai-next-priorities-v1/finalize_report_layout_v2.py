"""Apply the visual-only renderer after the existing final publication owner.

Preserve the prior report snapshot. Final status still requires every original
scientific gate; a failed prior study produces only an honest interim report.
"""
import json
import os
import shutil
import signal
import subprocess
import time
from study_support import ROOT, HERE, BASE, DOCS, read, write, sha


if __name__ == '__main__':
    registration_path = DOCS / 'report-layout-finalization-registration-02.json'
    def verify():
        registration = read(registration_path)
        for relative, digest in registration['source_sha256'].items():
            if sha(ROOT / relative) != digest:
                raise ValueError('Registered presentation source changed: ' + relative)
        return registration
    registration = verify()
    receipt_path = BASE / 'report-layout-finalization-receipt-02.json'
    if receipt_path.exists():
        raise ValueError('Fresh presentation receipt required')
    owner_path = BASE / 'final-publication-batch-01/execution.json'
    deadline = time.monotonic() + 110000
    def interrupted(*_):
        raise InterruptedError('Presentation finalization interrupted; no replay')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    while True:
        try:
            owner = read(owner_path)
        except json.JSONDecodeError:
            if time.monotonic() >= deadline:
                raise TimeoutError('Existing publication observation deadline')
            time.sleep(1)
            continue
        if owner['status'] != 'running':
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Existing publication remains live; never replay numerical work')
        time.sleep(5)
    if owner['status'] not in ('completed', 'stopped') or not owner['owned_child_reaped'] or owner.get('active') is not None:
        raise ValueError('Existing publication must be terminal and reaped')
    verify()
    receipt = dict(status='running', commands=[], prior_publication_owner_sha256=sha(owner_path),
        prior_publication_status=owner['status'], registration_sha256=sha(registration_path),
        source_sha256=sha(__file__), goal_status_updated=False, numerical_work_replayed=False)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    environment.pop('PYTHONPATH', None)
    def command(interpreter, script, *arguments):
        child = subprocess.Popen([str(interpreter), '-B', str(HERE / script), *arguments], cwd=ROOT, env=environment)
        try:
            code = child.wait(timeout=1800)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            receipt['commands'].append(dict(script=script, arguments=list(arguments), pid=child.pid,
                returncode=child.returncode, reaped=child.poll() is not None))
            write(receipt_path, receipt)
        if code:
            raise RuntimeError('Presentation command failed: ' + script)
    try:
        # Preserve the exact old report, including its source/hash provenance.
        previous_path = BASE / 'final-publication-receipt-01.json'
        previous = read(previous_path) if previous_path.exists() else {}
        final = owner['status'] == 'completed' and previous.get('status') == 'completed'
        if final:
            if previous['final_report_sha256'] != sha(DOCS / 'frontiers.json') or previous['completion_audit_sha256'] != sha(DOCS / 'completion-audit.json'):
                raise ValueError('Prior completed publication changed before snapshot')
        snapshot = BASE / 'report-layout-repair-01/prior-publication-snapshot-01'
        snapshot.mkdir()
        files = [DOCS / name for name in ('frontiers.json', 'frontiers.csv', 'accuracy.csv',
            'results.md', 'results.html', 'completion-audit.json', 'completion-audit.md')]
        files.extend(sorted((DOCS / 'figures').glob('*.png')))
        files.extend(sorted((DOCS / 'figures').glob('*.svg')))
        if previous_path.exists():
            files.append(previous_path)
        preserved = {}
        for path in files:
            if not path.exists():
                continue
            relative = path.relative_to(ROOT)
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if sha(destination) != sha(path):
                raise ValueError('Prior report snapshot differs')
            preserved[str(relative)] = dict(path=str(destination.relative_to(ROOT)), sha256=sha(destination))
        index = snapshot / 'snapshot.json'
        write(index, dict(prior_owner_sha256=sha(owner_path), prior_publication_completed=final, artifacts=preserved))
        receipt['prior_report_snapshot_sha256'] = sha(index)
        plot = ROOT / 'runs/full-ai-next-priorities-v1/plot-env-01/bin/python'
        command(plot, 'build_report_v2.py', *(['--final'] if final else []))
        control = ROOT / 'runs/dependency-migration-v1/app-env/bin/python3.12'
        command(control, 'audit_completion.py')
        audit = read(DOCS / 'completion-audit.json')
        if final and not audit['completion_proven']:
            raise ValueError('Final visual report did not retain all original requirement passes')
        receipt.update(status='completed_final' if final else 'completed_interim',
            completion_proven=audit['completion_proven'], final_report_sha256=sha(DOCS / 'frontiers.json'),
            completion_audit_sha256=sha(DOCS / 'completion-audit.json'),
            scientific_selection_and_thresholds_unchanged=True, independent_agent_review_required=True)
        print('Visual-only report finalization complete; completion proven:', audit['completion_proven'], flush=True)
    except BaseException as error:
        receipt.update(status='stopped', error=repr(error))
        raise
    finally:
        write(receipt_path, receipt)
