"""Run an isolated native Tuldok controller with task-owned PostgreSQL.

Uses ordinary local owner bootstrap. No existing service configuration, tokens
or databases are read. Credentials stay in the ignored private control tree.
SIGTERM/SIGINT drain the owned application and stop the owned database.
"""
import argparse
import hashlib
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import urllib.request
from urllib.parse import urlsplit
import uuid


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/model-portfolio-migration-v1/sources/Tuldok'
PYTHON = ROOT / 'runs/dependency-migration-v1/app-env/bin/python'
PG = Path('/usr/lib/postgresql/16/bin')


def private_write(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(text)


def private_replace(path, text):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    private_write(temporary,text)
    temporary.replace(path)


def require_prior_boot(previous, cluster):
    """Permit restart only when a host reboot proves the old owners are gone.

    PostgreSQL owns crash recovery. This does not acknowledge an interrupted
    project operation or replay its mutation.
    """
    boot_seconds = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                        if line.startswith('btime '))
    if previous.get('ready_unix', time.time()) >= boot_seconds:
        raise ValueError('Recovery requires a controller from an earlier host boot')
    if any(Path('/proc', str(previous[key])).exists() for key in ('launcher_pid', 'app_pid')):
        raise ValueError('Prior process identifier exists; reconcile before recovery')
    stopped = subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), 'status'],
                             capture_output=True, timeout=15)
    if stopped.returncode != 3:
        raise ValueError('Recovery requires the original PostgreSQL cluster to be stopped')
    return dict(host_boot_unix=boot_seconds,
                host_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                prior_processes_absent=True, postgres_status_returncode=stopped.returncode,
                native_mutation_replayed=False)


def run(output, cold_audit=False, root_binding=None, *, source=SOURCE, resume=False, keep_open=False,
        recover_after_reboot=False):
    if (not source.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1/sources')
            or not (source / 'app.py').is_file()):
        raise ValueError('Explicit task-owned native application source required')
    bindings = root_binding if type(root_binding) is list else [root_binding or 'lab='+str(ROOT)]
    configured=[];seen=set()
    for binding in bindings:
        alias,separator,directory=binding.partition('=')
        resolved=Path(directory).resolve()
        if (not separator or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',alias)
                or not Path(directory).is_absolute() or not resolved.is_dir() or alias in seen):
            raise ValueError('Native roots must be distinct ALIAS=absolute-existing-directory selections')
        seen.add(alias);configured.append(dict(alias=alias,directory=str(resolved)))
    if cold_audit and len(configured)!=1:
        raise ValueError('Cold audit requires one explicit confined project root')
    alias=configured[0]['alias'];root_directory=Path(configured[0]['directory'])
    if not output.is_relative_to(ROOT/'runs/model-portfolio-migration-v1') or os.geteuid()==0:
        raise ValueError('Task-owned controller path and ordinary local user required')
    if recover_after_reboot and not resume:
        raise ValueError('Reboot recovery requires explicit resume of the existing controller')
    previous=None
    recovery=None
    if resume:
        previous=json.loads((output/'controller.json').read_text())
        if (cold_audit or previous.get('cold_open_audit') or previous['status']!='stopped'
                or not previous['app_stopped'] or not previous['postgres_stopped']):
            if not recover_after_reboot or cold_audit or previous.get('cold_open_audit'):
                raise ValueError('Resume requires a clean stop or explicit prior-boot recovery')
            recovery=require_prior_boot(previous,output/'private/postgres-data')
        if previous.get('root_bindings',[previous['root_binding']])!=configured:
            raise ValueError('Resume requires the same root and a cleanly stopped ordinary controller')
        for key in ('launcher_pid','app_pid'):
            if Path('/proc',str(previous[key])).exists():
                raise ValueError('Prior process identifier exists; reconcile it before resuming')
        endpoint=urlsplit(previous['origin'])
        if endpoint.scheme!='http' or endpoint.hostname!='127.0.0.1' or endpoint.port is None:
            raise ValueError('Resume requires the original loopback endpoint')
        port=endpoint.port
    elif output.exists():
        raise ValueError('Fresh task-owned controller directory and ordinary local user required')
    else:
        output.mkdir(parents=True)
        port=0
    private = output / 'private'
    if resume:
        if private.is_symlink() or (private/'postgres-data'/'PG_VERSION').read_text().strip()!='16':
            raise ValueError('Original task-owned private PostgreSQL16 cluster required')
    else:private.mkdir(mode=0o700)
    cluster = private / 'postgres-data'
    socket = Path(tempfile.mkdtemp(prefix='rheon-portfolio-native-', dir='/tmp'))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment['TULDOK_DATABASE_URL'] = f'dbname=postgres host={socket} port=56974'
    (private_replace if resume else private_write)(private / 'database.json', json.dumps(dict(dsn=environment['TULDOK_DATABASE_URL'])))
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop);signal.signal(signal.SIGINT, stop)
    app = None
    state = dict(status='starting', launcher_pid=os.getpid(), source=str(source),
                 source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                 source_worktree_dirty=bool(subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()),
                 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 control=str(private / 'control'), live_catalog_accessed=False,
                 live_credentials_read=False, app_stopped=False, postgres_stopped=False,
                 root_binding=dict(alias=alias, directory=str(root_directory)),root_bindings=configured)
    state.update(keep_open=keep_open,resumed=resume)
    if recovery is not None:
        receipt=output/('reboot-recovery-'+uuid.uuid4().hex+'.json')
        private_write(receipt,json.dumps(dict(previous_controller=previous,recovery=recovery),indent=2)+'\n')
        state['reboot_recovery_receipt']=str(receipt)
    if previous:
        state['previous_catalog_id']=previous['catalog_id']
        state['restart_count']=previous.get('restart_count',0)+1
    app_environment = dict(environment)
    if cold_audit:
        audit_dir = private / 'cold-audit';audit_dir.mkdir()
        audit_log = audit_dir / 'events.jsonl'
        armed = audit_dir / 'armed'
        # Verification only. The normal app and portable project contain no
        # audit hook. Arm after startup and before the explicit cold Open.
        code = '''import json, os, sys
_root = ROOT_VALUE
_project = PROJECT_VALUE
_original_catalog = ORIGINAL_CATALOG_VALUE
_allowed = ALLOWED_VALUE
_armed = ARMED_VALUE
_log = LOG_VALUE
def _record(value):
    with open(_log, 'a') as stream: stream.write(json.dumps(value)+'\\n')
_record({'event':'audit_installed','pid':os.getpid()})
def _guard(event, args):
    if event not in ('subprocess.Popen','os.system','os.exec','os.posix_spawn','import','open','exec'): return
    if not os.path.exists(_armed): return
    reason = None
    if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn'): reason='process_execution'
    if event=='import' and args and (args[0]=='torch' or args[0].startswith('torch.')): reason='neural_runtime_import'
    if event in ('open','exec'):
        value = args[0] if event=='open' else getattr(args[0], 'co_filename', '')
        if isinstance(value, str) and os.path.isabs(value):
            path=os.path.normpath(value)
            if path==_original_catalog or path.startswith(_original_catalog+'/'): reason='original_catalog_dependency'
            if event=='exec' and (path==_project or path.startswith(_project+'/')): reason='portable_script_execution'
            if path.startswith(_root+'/') and not any(path==a or path.startswith(a+'/') for a in _allowed): reason='external_lab_dependency'
    if reason:
        _record({'event':event,'blocked':reason,'pid':os.getpid()})
        raise AssertionError('Cold Open must remain inert and self-contained: '+reason)
sys.addaudithook(_guard)
'''
        values = dict(ROOT_VALUE=str(ROOT), PROJECT_VALUE=str(root_directory),
                      ORIGINAL_CATALOG_VALUE='/media/jeremy/OrangeCream/Linux Software/repos/owned/creative-media/Tuldok/data-postgres',
                      ALLOWED_VALUE=[str(source), str(PYTHON.parent.parent), str(private),
                                     str(root_directory), str(ROOT)],
                      ARMED_VALUE=str(armed), LOG_VALUE=str(audit_log))
        # ROOT itself permits the anchored root directory open, not all of its
        # children; the hook handles that one path separately below.
        values['ALLOWED_VALUE'].remove(str(ROOT))
        code = code.replace("path.startswith(_root+'/')", "path!=_root and path.startswith(_root+'/')")
        for key, value in values.items():code = code.replace(key, repr(value))
        (audit_dir / 'sitecustomize.py').write_text(code)
        app_environment['PYTHONPATH'] = str(audit_dir)
        state['cold_open_audit'] = dict(armed_file=str(armed), log=str(audit_log), verification_only=True,
                                      forbidden_original_catalog=values['ORIGINAL_CATALOG_VALUE'],
                                      stack_signal_diagnostics=False)
    def save():
        path = output / 'controller.json'
        pending = path.with_suffix('.pending')
        pending.write_text(json.dumps(state, indent=2) + '\n');pending.replace(path)
    save()
    try:
        with (output / 'postgres-setup.log').open('a' if resume else 'w') as log:
            if not resume:subprocess.run([str(PG / 'initdb'), '-D', str(cluster), '--auth-local=trust',
                            '--auth-host=reject', '--no-locale', '-E', 'UTF8'],
                           stdout=log, stderr=subprocess.STDOUT, env=environment, check=True, timeout=90)
            subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-l', str(output / 'postgres.log'),
                            '-o', f"-h '' -k {socket} -p 56974 -c max_connections=20 -c shared_buffers=32MB",
                            '-w', '-t', '45', 'start'], stdout=log, stderr=subprocess.STDOUT,
                           env=environment, check=True, timeout=60)
        log_path=output/'app.log'
        log_offset=log_path.stat().st_size if resume else 0
        with log_path.open('a' if resume else 'w') as log:
            app = subprocess.Popen([str(PYTHON), '-B', '-u', str(source / 'app.py'),
                                    '--data', state['control'], '--host', '127.0.0.1', '--port', str(port),
                                    *[argument for binding in configured for argument in ('--project-files-root',binding['alias']+'='+binding['directory'])]],
                                   cwd=source, env=app_environment, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not stopping:
                if app.poll() is not None:
                    raise RuntimeError('Native controller exited during startup; inspect app.log')
                with log_path.open('rb') as observed:
                    observed.seek(log_offset);current_log=observed.read().decode('utf-8',errors='replace')
                match = re.search(r'Tuldok: (http://127\.0\.0\.1:\d+)', current_log)
                if match:
                    state['origin'] = match[1];break
                time.sleep(.2)
            if 'origin' not in state:
                raise TimeoutError('Native controller startup deadline')
            state['app_pid'] = app.pid
            owner_login=private/('owner-login-resume-'+uuid.uuid4().hex+'.txt' if resume else 'owner-login.txt')
            subprocess.run([str(PYTHON), '-B', str(source / 'access.py'), 'login' if resume else 'bootstrap',
                            '--data', state['control'], '--out', str(owner_login)],
                           cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
            request = urllib.request.Request(state['origin'] + '/api/access/session',
                data=json.dumps(dict(token=owner_login.read_text().strip())).encode(),
                headers={'Content-Type': 'application/json', 'Origin': state['origin']})
            with urllib.request.urlopen(request, timeout=30) as response:
                cookies = SimpleCookie()
                for value in response.headers.get_all('Set-Cookie', []):cookies.load(value)
                identity = json.loads(response.read())
            if identity.get('kind') != 'owner' or 'tuldok_session' not in cookies:
                raise ValueError('Fresh native installation owner exchange failed')
            if resume and (identity['catalog_id']!=previous['catalog_id'] or state['origin']!=previous['origin']):
                raise ValueError('Resumed controller changed its installation identity or endpoint')
            (private_replace if resume else private_write)(private / 'owner-session.txt', cookies['tuldok_session'].value + '\n')
            browser_login=private/('browser-login-resume-'+uuid.uuid4().hex+'.txt' if resume else 'browser-login.txt')
            subprocess.run([str(PYTHON), '-B', str(source / 'access.py'), 'login',
                            '--data', state['control'], '--out', str(browser_login)],
                           cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
            if resume:private_replace(private/'browser-login.txt',browser_login.read_text())
            state.update(status='ready', catalog_id=identity['catalog_id'], ready_unix=time.time())
            save();print(json.dumps(state), flush=True)
            deadline = None if keep_open else time.monotonic() + 7200
            while not stopping and (deadline is None or time.monotonic() < deadline):
                if app.poll() is not None:
                    raise RuntimeError('Native controller stopped; inspect app.log')
                time.sleep(.5)
            state['status'] = 'stopping'
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if app is not None and app.poll() is None:
            app.send_signal(signal.SIGINT)
            try:app.wait(timeout=30)
            except subprocess.TimeoutExpired:
                app.terminate()
                try:app.wait(timeout=10)
                except subprocess.TimeoutExpired:app.kill();app.wait(timeout=10)
        state['app_stopped'] = app is None or app.poll() is not None
        state['app_returncode'] = app.returncode if app is not None else None
        stopped = subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-m', 'fast',
                                  '-w', '-t', '30', 'stop'], capture_output=True, text=True, timeout=45)
        state['postgres_stopped'] = stopped.returncode == 0
        (output / 'postgres-stop.log').write_text(stopped.stdout + stopped.stderr)
        if state['postgres_stopped']:socket.rmdir()
        if state['status'] == 'stopping':state['status'] = 'stopped'
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cold-audit', action='store_true')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--root', action='append', help='Explicit project root, ALIAS=absolute-directory; repeat for additional roots')
    parser.add_argument('--resume',action='store_true',help='Resume this cleanly stopped task-owned UI installation at the same port')
    parser.add_argument('--recover-after-reboot',action='store_true',help='With resume, verify prior-boot owners are gone and let PostgreSQL recover its existing cluster; never replay project operations')
    parser.add_argument('--keep-open',action='store_true',help='Keep serving until explicitly stopped')
    args = parser.parse_args()
    run(args.output.resolve(), args.cold_audit, args.root, source=args.source.resolve(),resume=args.resume,
        keep_open=args.keep_open,recover_after_reboot=args.recover_after_reboot)
