"""Qualify native project SQL lifecycle on a task-owned disposable PostgreSQL.

No live service settings or credentials are read. The upstream tests use their
own explicit fixture owners and cannot establish lab migration acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/model-portfolio-migration-v1/sources/Tuldok'
PYTHON = ROOT / 'runs/dependency-migration-v1/app-env/bin/python'
PG = Path('/usr/lib/postgresql/16/bin')
TESTS = [
    'test_project_workspace',
    'test_project_extraction.LegacyCapture.test_immutable_model_bytes_and_original_publication_audit_survive',
    'test_project_extraction.LegacyCapture.test_two_study_lineage_reservation_candidate_and_run_science_are_captured',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, report, *, source=SOURCE, tests=None, cold_role=False):
    tests = TESTS if tests is None else tests
    if (not source.is_relative_to(ROOT / 'runs/model-portfolio-migration-v1/sources')
            or not (source / 'app.py').is_file() or not tests):
        raise ValueError('Explicit task-owned application source and nonempty tests required')
    if output.exists() or report.exists():
        raise ValueError('Fresh native qualification output paths required')
    if os.geteuid() == 0:
        raise ValueError('PostgreSQL fixture must run as the ordinary workspace user')
    output.mkdir(parents=True)
    cluster = output / 'postgres-data'
    socket = Path(tempfile.mkdtemp(prefix='rheon-portfolio-pg-', dir='/tmp'))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Unix socket only, inside a private 0700 directory. No TCP listener or
    # existing server connection, password or service configuration is involved.
    environment['TULDOK_DATABASE_URL'] = f'dbname=postgres host={socket} port=56973'
    environment['PYTHONPATH'] = str(source)
    environment['TULDOK_QA_OUTPUT_ROOT'] = str(output / 'qa')
    started = time.monotonic()
    source_files = ['backup_metadata.py', 'project_transfer.py',
                    'project_workspace.py', 'project_store.py', 'project_projection.py',
                    'project_extraction.py', 'project_history.py', 'project_material.py', 'projects.py', 'torch_runtime.py',
                    'runtime_snapshots.py', 'recipe_process.py', 'recipe_contract.py', 'runs.py',
                    'project_artifacts.py', 'project_files.py',
                    'agent_api.py', 'project_dispatch.py', 'tuldok_client.py',
                    'static/project-files.js', 'static/project-files.html',
                    'tests/test_project_workspace.py', 'tests/test_project_extraction.py',
                    'tests/test_runtime_snapshots.py', 'tests/test_project_study_execution.py']
    for name in ('project_composition.py', 'project_session.py', 'static/experiment-graph.js', 'static/project-progression.js',
                 'project_measurements.py', 'retained_measurements.py',
                 'project_sources.py', 'project_fork.py', 'project_fork_composition.py', 'app.py',
                 'static/retained-chart.js', 'static/retained-analysis.js',
                 'static/retained-measurement-maxima.vl.json',
                 'static/project-measurements.js', 'static/project-measurements.css',
                 'static/workspaces.html'):
        if (source / name).is_file():
            source_files.append(name)
    for test in tests:
        path = 'tests/' + test.split('.')[0] + '.py'
        if path not in source_files:
            source_files.append(path)
    value = dict(schema='rheon.native-project-SQL-fixture.v1', created_unix=time.time(),
                 status='running', tests=tests, source_path=str(source.relative_to(ROOT)), source_sha256={
                     name: sha(source / name) for name in source_files},
                 runner_sha256=sha(Path(__file__)),
                 source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                 source_tree=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
                 source_worktree_dirty=bool(subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()),
                 live_catalog_accessed=False, live_credentials_read=False,
                 lab_project_created=False, lab_models_imported=False,
                 native_UI_tested=False, full_migration_qualified=False,
                 postgres_stopped=False)
    def save():
        report.write_text(json.dumps(value, indent=2) + '\n')
    save()
    try:
        with (output / 'postgres-setup.log').open('w') as log:
            subprocess.run([str(PG / 'initdb'), '-D', str(cluster),
                            '--auth-local=trust', '--auth-host=reject', '--no-locale', '-E', 'UTF8'],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
            subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster), '-l', str(output / 'postgres.log'),
                            '-o', f"-h '' -k {socket} -p 56973 -c max_connections=20 -c shared_buffers=32MB",
                            '-w', '-t', '45', 'start'], env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=60)
        if cold_role:
            import psycopg
            from psycopg import sql
            suffix = uuid.uuid4().hex[:12]
            role, database = 'rheon_cold_' + suffix, 'rheon_cold_' + suffix
            with psycopg.connect(environment['TULDOK_DATABASE_URL'], autocommit=True) as connection:
                connection.execute(sql.SQL('CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS').format(sql.Identifier(role)))
                connection.execute(sql.SQL('CREATE DATABASE {} OWNER {}').format(sql.Identifier(database), sql.Identifier(role)))
                connection.execute('REVOKE CONNECT ON DATABASE postgres FROM PUBLIC')
                connection.execute(sql.SQL('REVOKE ALL ON DATABASE {} FROM PUBLIC').format(sql.Identifier(database)))
                denied = not connection.execute("SELECT has_database_privilege(%s,'postgres','CONNECT')", (role,)).fetchone()[0]
                if not denied:
                    raise RuntimeError('Cold fixture role can connect to the source database')
            environment['TULDOK_PROJECT_FILES_COLD_DATABASE_URL'] = f'user={role} dbname={database} host={socket} port=56973'
            value.update(cold_role_provisioned=True, cold_role_source_CONNECT_denied=denied)
            save()
        with (output / 'tests.log').open('w') as log:
            result = subprocess.run([str(PYTHON), '-B', '-m', 'unittest', '-v', *tests],
                                    cwd=source / 'tests', env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=900)
        text = (output / 'tests.log').read_text()
        value.update(test_returncode=result.returncode, tests_log_sha256=sha(output / 'tests.log'),
                     status='passed_disposable_native_project_SQL_fixture' if result.returncode == 0 and 'skipped=' not in text
                     else 'failed_or_incomplete_fixture', elapsed_seconds=time.monotonic() - started)
        if result.returncode:
            raise RuntimeError('Native SQL fixture failed; inspect retained test log')
        if value['source_sha256'] != {name: sha(source / name) for name in source_files}:
            raise RuntimeError('Observed source changed during native qualification')
        value['source_stable_through_tests'] = True
    except BaseException as error:
        value.update(status='failed_or_incomplete_fixture', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if cluster.exists():
            stopped = subprocess.run([str(PG / 'pg_ctl'), '-D', str(cluster),
                                      '-m', 'fast', '-w', '-t', '30', 'stop'],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45)
            value['postgres_stopped'] = stopped.returncode == 0 or 'no server running' in stopped.stdout
            (output / 'postgres-stop.log').write_text(stopped.stdout)
        if value['postgres_stopped']:
            shutil.rmtree(socket)
        save()
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--tests', nargs='+')
    parser.add_argument('--cold-role', action='store_true')
    args = parser.parse_args()
    run(args.output.resolve(), args.report.resolve(), source=args.source.resolve(), tests=args.tests,
        cold_role=args.cold_role)
