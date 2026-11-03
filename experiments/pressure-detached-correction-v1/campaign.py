"""Prepare and run the nine frozen fits sequentially without mutation retries."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DIRECTORY = ROOT / 'runs/pressure-detached-correction-v1'
REPORTS = ROOT / 'docs/plans/flow-reasoning-assessment/execution'
sys.path.insert(0, str(ROOT / 'experiments/pressure-four-pass-distillation-v1'))
from common import environment, read, sha, write


def process(command, log, env=None, timeout=4500):
    with log.open('xb') as stream:
        child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
            if code:
                raise RuntimeError('Command failed; inspect ' + str(log.relative_to(ROOT)))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def prepare():
    DIRECTORY.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    target = DIRECTORY / 'campaign.json'
    if target.exists():
        raise ValueError('Existing campaign must be inspected; do not prepare over it')
    runtime = read(ROOT / 'runs/dependency-migration-v1/runtime/current-manifest-v2.json')
    app_python = ROOT / 'runs/dependency-migration-v1/app-env/bin/python3.12'
    rows = []
    for initialization in ('C53', 'C44', 'C37'):
        for mode in ('U', 'A', 'D'):
            key = initialization.lower() + '-' + mode.lower()
            project = DIRECTORY / (key + '-project-' + ('02' if key == 'c53-u' else '01'))
            qualification = DIRECTORY / (key + '-qualification-01.json')
            if not (project / 'assembly.json').exists():
                process([app_python, '-B', HERE / 'prepare.py', '--initialization', initialization,
                         '--mode', mode, '--output', project], DIRECTORY / (key + '-preparation.log'), timeout=120)
            if not qualification.exists():
                process([runtime['python'], '-B', HERE / 'qualify.py', '--project', project,
                         '--output', qualification], DIRECTORY / (key + '-qualification.log'),
                         environment(runtime), timeout=180)
            checked = read(qualification)
            if (checked['status'] != 'passed' or checked['optimizer_updates'] != 0
                    or checked['project_sha256'] != sha(project / 'project.json')
                    or checked['source_witness_sha256'] != sha(project / 'source-witness.json')
                    or checked['detached_correction']['initialization'] != initialization):
                raise ValueError('Exact arm qualification required: ' + key)
            rows.append(dict(key=key, initialization=initialization, mode=mode,
                project=str(project.relative_to(ROOT)), config=str((DIRECTORY / (key + '-experiment.json')).relative_to(ROOT)),
                assembly_sha256=sha(project / 'assembly.json'), qualification=str(qualification.relative_to(ROOT)),
                qualification_sha256=sha(qualification), config_sha256=sha(DIRECTORY / (key + '-experiment.json')),
                project_id=read(project / 'project.json')['id']))
            write(REPORTS / (key + '-qualification.json'), checked)
            print('Ready', key, flush=True)
    write(target, dict(schema='rheon-detached-correction-campaign-v1', rows=rows,
        source_sha256=sha(__file__), runtime_manifest_sha256=sha(ROOT / 'runs/dependency-migration-v1/runtime/current-manifest-v2.json'),
        created_at=datetime.now(timezone.utc).isoformat(), optimizer_updates_per_fit=1600,
        fits=9, automatic_retry=False, model_promotion=False))
    write(REPORTS / 'prepared.json', read(target))


def run(session_file, *, resume_before_admission=False):
    campaign = read(DIRECTORY / 'campaign.json')
    execution_path = DIRECTORY / 'execution.json'
    controller_attempt = 1
    if execution_path.exists():
        if not resume_before_admission:
            raise ValueError('Execution already exists; reconcile journals before any continuation')
        previous = read(execution_path)
        controller = DIRECTORY / 'c53-u-controller-01'
        stopped = read(controller / 'execution.json')
        if (previous['status'] != 'stopped' or previous['completed'] or previous['active'] != 'c53-u'
                or stopped['status'] != 'failed' or not stopped['owned_controller_reaped']
                or list(DIRECTORY.glob('*-study-*'))
                or list(DIRECTORY.glob('**/mutations/*.json'))
                or 'A valid credential is required.' not in (controller / 'managed.log').read_text()):
            raise ValueError('Only the proven zero-mutation authentication failure can restart here')
        archived = DIRECTORY / 'execution-attempt-01.json'
        if archived.exists():
            raise ValueError('Authentication continuation was already attempted')
        write(archived, previous)
        write(REPORTS / 'execution-attempt-01.json', previous)
        controller_attempt = 2
    elif resume_before_admission:
        raise ValueError('No pre-admission failure to continue')
    for row in campaign['rows']:
        attempt = controller_attempt if row['key'] == 'c53-u' else 1
        if ((DIRECTORY / (row['key'] + f'-controller-{attempt:02d}')).exists()
                or (DIRECTORY / (row['key'] + '-study-01')).exists()):
            raise ValueError('Existing arm custody must be inspected: ' + row['key'])
        if (sha(ROOT / row['project'] / 'assembly.json') != row['assembly_sha256']
                or sha(ROOT / row['qualification']) != row['qualification_sha256']
                or sha(ROOT / row['config']) != row['config_sha256']):
            raise ValueError('Frozen campaign material changed')
    state = dict(status='running', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                 completed=[], active=None, automatic_retry=False, model_promotion=False,
                 source_sha256=sha(__file__), explicit_pre_admission_continuation=resume_before_admission)
    started = time.perf_counter()
    def publish():
        state['elapsed_s'] = time.perf_counter() - started
        write(execution_path, state)
        write(REPORTS / 'execution.json', state)
    def stop(*_):
        raise KeyboardInterrupt('Campaign interrupted; inspect managed custody before continuing')
    signal.signal(signal.SIGTERM, stop)
    publish()
    try:
        for row in campaign['rows']:
            state['active'] = row['key']
            publish()
            attempt = controller_attempt if row['key'] == 'c53-u' else 1
            controller = DIRECTORY / (row['key'] + f'-controller-{attempt:02d}')
            controller_log = row['key'] + ('-controller.log' if attempt == 1 else f'-controller-attempt-{attempt:02d}.log')
            process([sys.executable, '-B', HERE / 'run.py', '--config', ROOT / row['config'],
                '--assembly', ROOT / row['project'] / 'assembly.json', '--qualification', ROOT / row['qualification'],
                '--session-file', session_file, '--output', controller], DIRECTORY / controller_log)
            receipt_path = DIRECTORY / (row['key'] + '-study-01/receipt.json')
            receipt = read(receipt_path)
            if receipt['status'] != 'completed' or receipt['optimizer_updates'] != 1600 or not all(receipt['artifact_parity'].values()):
                raise ValueError('Managed train/restore parity failed')
            work = receipt['training_receipt']['fine_tuning_statistics']
            assert work['fundamental_bank']['optimizer_exposures'] == [200]*8
            assert work['rollout_feedback']['depth_exposures'] == [[50]*4 for _ in range(8)]
            assert work['rollout_feedback']['auxiliary_losses'] == (0 if row['mode'] == 'U' else 1600)
            history = read(receipt_path.parent / 'restore/history.json')
            assert history['optimizer_work']['network_forwards'] == (6400 if row['mode'] == 'U' else 8000)
            if row['key'] == 'c53-u':
                retained = read(ROOT / 'runs/pressure-fundamental-bank-fit-v1/shape-isolation-study-01/receipt.json')
                if receipt['weights_sha256'] != retained['weights_sha256']:
                    raise ValueError('C53 U did not reproduce retained C66; retain result and inspect before further fits')
            state['completed'].append(dict(key=row['key'], training_run_id=receipt['training_run_id'],
                restore_run_id=receipt['restore_run_id'], weights_sha256=receipt['weights_sha256'],
                managed_elapsed_s=receipt['elapsed_s'], artifact_parity=receipt['artifact_parity']))
            write(REPORTS / (row['key'] + '-training.json'), dict(
                status='completed', initialization=row['initialization'], mode=row['mode'],
                receipt=str(receipt_path.relative_to(ROOT)), receipt_sha256=sha(receipt_path),
                **state['completed'][-1], optimizer_work=history['optimizer_work'],
                rollout_feedback=work['rollout_feedback']))
            print('Completed', row['key'], flush=True)
            publish()
        state.update(status='training_completed', active=None)
    except BaseException as error:
        state.update(status='stopped', error=str(error), remote_custody='Inspect durable arm journals; no automatic resubmission.')
        raise
    finally:
        publish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run'])
    parser.add_argument('--session-file', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        if args.session_file is None:
            parser.error('--session-file is required for managed admission')
        run(args.session_file.resolve())
