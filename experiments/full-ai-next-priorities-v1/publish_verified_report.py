"""Publish current evidence; final status requires a complete scientific audit."""
import os
import subprocess
import time
from await_current_confirmations import verify_sources
from study_support import ROOT, HERE, BASE, DOCS, read, write, sha


if __name__ == '__main__':
    registration = verify_sources()
    admission_path = BASE / 'final-publication-admission-01.json'
    admission = read(admission_path)
    for name, digest in admission['owners'].items():
        if sha(BASE / name / 'execution.json') != digest:
            raise ValueError('Admitted terminal owner changed')
    control = ROOT / 'runs/dependency-migration-v1/app-env/bin/python3.12'
    plot = ROOT / 'runs/full-ai-next-priorities-v1/plot-env-01/bin/python'
    receipt_path = BASE / 'final-publication-receipt-01.json'
    if receipt_path.exists():
        raise ValueError('Fresh publication receipt required')
    receipt = dict(status='running', commands=[], registration_sha256=sha(DOCS / 'final-publication-registration-01.json'),
        admission_sha256=sha(admission_path), source_sha256=sha(__file__), goal_status_updated=False)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    environment.pop('PYTHONPATH', None)
    def command(interpreter, script, *arguments):
        started = time.monotonic()
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
                returncode=child.returncode, reaped=child.poll() is not None, elapsed_s=time.monotonic()-started))
            write(receipt_path, receipt)
        if code:
            raise RuntimeError('Publication command failed: ' + script)
    try:
        command(control, 'refresh_work_state.py')
        command(control, 'audit_requirements.py')
        command(plot, 'build_report.py')
        command(control, 'audit_completion.py')
        readiness = read(DOCS / 'completion-audit.json')
        if not readiness['scientific_gates_passed'] or readiness['failed']:
            raise ValueError('Scientific target or evidence audit not complete; actual interim results retained')
        command(plot, 'build_report.py', '--final')
        command(control, 'audit_completion.py')
        final = read(DOCS / 'completion-audit.json')
        if not final['completion_proven'] or final['pending'] or final['failed']:
            raise ValueError('Final requirement-by-requirement audit did not prove completion')
        receipt.update(status='completed', completion_audit_sha256=sha(DOCS / 'completion-audit.json'),
            final_report_sha256=sha(DOCS / 'frontiers.json'),
            requirements_verified=[r['id'] for r in final['requirements']],
            thread_goal_requires_independent_agent_review=True)
        print('All original requirements verified and final report published; thread goal unchanged', flush=True)
    except BaseException as error:
        receipt.update(status='stopped', error=repr(error))
        raise
    finally:
        write(receipt_path, receipt)
