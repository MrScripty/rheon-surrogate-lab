"""Local gate negatives; no original endpoint, browser, owner session or DB."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('live_browser', HERE / 'live_browser.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


def documents():
    cutover = {'schema': 'rheon-tuldok-migration-cutover-v1', 'm3_service_switched': True, 'm3_complete': False,
        'ordinary_admissions_open': False, 'smoke_authorized': {
            'kind': 'typed_pressure_one_update_cpu_restore', 'max_optimizer_updates': 1},
        'endpoint': owner.ENDPOINT, 'catalog_id': owner.CATALOG}
    launch = {'status': 'current_service_launched_smoke_pending', 'endpoint': owner.ENDPOINT,
        'catalog_id': owner.CATALOG, 'endpoint_owned_by_selected_pid': True,
        'structural_service_switch_only': True, 'deployment_manifest_sha256': 'a' * 64,
        'app_sha256': 'b' * 64, 'app_source_sha256': 'c' * 64}
    machine = {'format': 'rheon_typed_pressure_qualification_v1',
        'status': 'machine_qualified_browser_pending', 'endpoint': owner.ENDPOINT,
        'catalog_id': owner.CATALOG, 'optimizer_updates_executed': 1,
        'restore_optimizer_updates': 0, 'ordinary_restore_bound_profile': False,
        'scientific_acceptance': 'not_asserted', 'model_promotion': False,
        'probe': {'status': 'exact', 'fresh_process': True, 'cuda_initialized': False,
                  'optimizer_updates': 0, 'field_api_cli_exact': True},
        'model': {'catalog_id': owner.CATALOG},
        'trial': {'material': {'catalog_id': owner.CATALOG, 'sha256': 'd' * 64}},
        'composition_sha256': 'd' * 64, 'training_run_id': 'train', 'restore_run_id': 'restore',
        'browser': {'run_id': 'train', 'checkpoint_sha256': 'e' * 64},
        'checkpoint_document_sha256': 'e' * 64,
        'source_verification': {'tuldok_source': {'content_sha256': 'c' * 64}}}
    provenance = {'identity_relation': 'new_current_project', 'project_id': 'f' * 32,
                  'current_manifest_sha256': '1' * 64}
    return [cutover, launch, machine, provenance]


class Gates(unittest.TestCase):
    def test_exact_cookie_origin_headers_against_deployed_handler(self):
        # Execute only the actual isolated predicate, without importing app/DB owners.
        source = HERE.parent / 'Tuldok-deployed/current-03/app.py'
        parsed = ast.parse(source.read_text())
        predicate = next(node for node in ast.walk(parsed)
                         if isinstance(node, ast.FunctionDef) and node.name == '_same_origin')
        namespace = {'access': SimpleNamespace(AccessError=RuntimeError)}
        exec(compile(ast.Module(body=[predicate], type_ignores=[]), str(source), 'exec'), namespace)
        guard = namespace['_same_origin']
        host = owner.ENDPOINT.removeprefix('http://')
        with self.assertRaises(RuntimeError):
            guard(SimpleNamespace(headers={'Host': host}), True)
        result = subprocess.run(['/home/jeremy/.nvm/versions/node/v24.12.0/bin/node', '-e',
            "const h=require(process.argv[1]).readHeaders;process.stdout.write(JSON.stringify(h({method:'POST',operation:'runs.model_evidence',version:2,sha256:'a'.repeat(64)})))",
            str(HERE / 'live_readback.cjs')], capture_output=True, text=True, check=True, timeout=5)
        headers = json.loads(result.stdout)
        self.assertEqual(headers['Origin'], owner.ENDPOINT)
        self.assertEqual(headers['Tuldok-Operation-Contract'], 'runs.model_evidence/2/' + 'a' * 64)
        guard(SimpleNamespace(headers={'Host': host, **headers}), True)
        with self.assertRaises(RuntimeError):
            guard(SimpleNamespace(headers={'Host': host, **headers, 'Origin': 'http://127.0.0.1:8100'}), True)

    def test_selected_node_imports_actual_playwright_under_launcher_environment(self):
        # Import only: no browser, endpoint, owner session or catalog access.
        selected = Path('/home/jeremy/.nvm/versions/node/v24.12.0/bin/node')
        with patch.dict(os.environ, {'PATH': '/usr/bin:/bin'}):
            with self.assertRaisesRegex(ValueError, 'Node 20 or later'):
                owner.inspect_node(Path('/usr/bin/node'))
            identity = owner.inspect_node(selected)
        self.assertEqual(identity['version'], '24.12.0')
        self.assertEqual(identity['sha256'], owner.sha(selected))
        result = subprocess.run([identity['path'], '-e',
            'require("/home/jeremy/.npm/_npx/9833c18b2d85bc59/node_modules/playwright");'
            'process.stdout.write(process.versions.node)'],
            env={'PATH': '/usr/bin:/bin', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0)
        self.assertGreaterEqual(int(result.stdout.split('.')[0]), 20)

    def check(self, docs):
        with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')):
            owner.validate_documents(*docs, 'a' * 64, 'b' * 64)

    def test_local_positive_never_opens_network(self):
        self.check(documents())

    def test_each_uncompleted_or_mismatched_authority_fails_locally(self):
        cases = [(0, ('m3_service_switched',), False), (0, ('ordinary_admissions_open',), True),
            (0, ('schema',), 'schema8'), (0, ('smoke_authorized', 'max_optimizer_updates'), 2),
            (1, ('deployment_manifest_sha256',), '0' * 64), (1, ('endpoint_owned_by_selected_pid',), False),
            (1, ('endpoint',), 'http://127.0.0.1:8123'), (2, ('status',), 'prepared'),
            (2, ('optimizer_updates_executed',), 2), (2, ('restore_optimizer_updates',), 1),
            (2, ('probe', 'fresh_process'), False), (2, ('probe', 'cuda_initialized'), True),
            (2, ('probe', 'field_api_cli_exact'), False), (2, ('model_promotion',), True),
            (2, ('scientific_acceptance',), 'accepted'), (2, ('ordinary_restore_bound_profile',), True),
            (2, ('trial', 'material', 'catalog_id'), 'another'),
            (2, ('browser', 'run_id'), 'restore'), (2, ('composition_sha256',), '0' * 64),
            (2, ('source_verification', 'tuldok_source', 'content_sha256'), '0' * 64),
            (3, ('identity_relation',), 'historical_alias')]
        for index, keys, value in cases:
            with self.subTest(keys=keys, value=value):
                docs = copy.deepcopy(documents()); target = docs[index]
                for key in keys[:-1]: target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError): self.check(docs)

    def test_gate_bytes_and_nonordinary_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'gate.json'; file.write_text('{"status":"pending"}')
            digest = owner.sha(file)
            self.assertEqual(owner.checked(file, digest)['status'], 'pending')
            file.write_text('{"status":"complete"}')
            with self.assertRaises(ValueError): owner.checked(file, digest)
            link = Path(directory) / 'alias.json'; link.symlink_to(file)
            with self.assertRaises(ValueError): owner.checked(link, owner.sha(file))

    def test_current_selected_docs_are_bounded_unpinned_external_references(self):
        root = owner.ROOT
        provenance = json.loads((owner.MIGRATION / 'current-project-04/provenance.json').read_bytes())
        manifest = root / provenance['manifest_path']
        self.assertEqual(owner.sha(manifest), provenance['current_manifest_sha256'])
        project = json.loads(manifest.read_bytes())
        for identifier in ('dependency-migration-plan', 'dependency-migration-results',
                           'dependency-migration-scientific-parity'):
            ref = next(row for row in project['evidence'] if row['id'] == identifier)
            self.assertNotIn('sha256', ref)
            self.assertLessEqual((root / ref['path']).stat().st_size, 65536)

    def test_complete_local_preparation_uses_real_current_artifact_contracts_only(self):
        docs = documents()
        application = owner.MIGRATION / 'Tuldok-deployed/current-03'
        docs[1].update(deployment=str(application),
            deployment_manifest_sha256=owner.sha(application / 'deployment-manifest.json'),
            app_sha256=owner.sha(application / 'app.py'), project_roots={'current-lab': str(owner.ROOT)})
        docs[2]['model'].update(id='1' * 32)
        docs[2]['trial'].update(id='2' * 32, study_id='3' * 32)
        docs[2].update(training_run_id='4' * 32, restore_run_id='5' * 32)
        docs[2]['browser'].update(run_id='4' * 32, member_ordinal=0, field_id='full_pressure')
        provenance = owner.MIGRATION / 'current-project-04/provenance.json'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = SimpleNamespace(application=application, current_project_provenance=provenance,
                provenance_sha256=owner.sha(provenance), project_alias='current-lab',
                import_reference='cubic-fields', existing_record=None, allow_project_import=True,
                execute_after_m3=False, owner_session_file=base / 'must-not-open-private-authority',
                output=base / 'must-not-create')
            for name, document in zip(('cutover', 'launch', 'machine'), docs[:3]):
                file = base / (name + '.json')
                if name == 'machine':
                    document.update(m3={'path': str(args.cutover_receipt), 'sha256': args.cutover_sha256},
                        app_launch={'path': str(args.launch_receipt), 'sha256': args.launch_sha256})
                file.write_text(json.dumps(document))
                setattr(args, name + '_receipt', file)
                setattr(args, name + '_sha256', owner.sha(file))
            with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden')), \
                 patch.object(owner.subprocess, 'Popen', side_effect=AssertionError('process forbidden')):
                config = owner.prepare(args)
            self.assertFalse(config['root_trigger'])
            self.assertEqual(config['provenance']['project_id'], json.loads(provenance.read_bytes())['project_id'])
            self.assertEqual(len(config['requests']), 7)
            self.assertTrue(all(row['path'].startswith('/api/workspace/') for row in config['requests'].values()))
            self.assertFalse(args.owner_session_file.exists())
            self.assertFalse(args.output.exists())

    def test_interrupt_and_failed_wait_kill_and_reap_real_inert_owned_child(self):
        for exception in (KeyboardInterrupt(), RuntimeError('private text must never be logged'),
                          subprocess.TimeoutExpired('inert child', 0)):
            with self.subTest(kind=type(exception).__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / 'import-intent.json').write_text('{"possible_dispatch":true}')
                child = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c',
                    'import time; time.sleep(60)'], start_new_session=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                real_wait = child.wait
                calls = 0

                def interrupted_wait(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise exception
                    return real_wait(*args, **kwargs)

                try:
                    with patch.object(owner.subprocess, 'Popen', return_value=child), \
                         patch.object(child, 'wait', side_effect=interrupted_wait):
                        receipt = owner.run_browser(['inert-offline-fixture'], output)
                    self.assertEqual(receipt['status'], 'stopped_inspect_receipt_before_retry')
                    self.assertEqual(receipt['error_kind'], type(exception).__name__)
                    self.assertTrue(receipt['owned_process_group_cleanup_attempted'])
                    self.assertTrue(receipt['owned_child_reaped'])
                    self.assertTrue(receipt['import_dispatch_may_have_committed'])
                    self.assertFalse(receipt['automatic_retry'])
                    self.assertEqual(receipt['returncode'], -9)
                    self.assertIsNone(receipt['cleanup_error_kind'])
                    self.assertEqual(json.loads((output / 'launcher-receipt.json').read_bytes()), receipt)
                    self.assertNotIn('private text', (output / 'launcher-receipt.json').read_text())
                    with self.assertRaises(ProcessLookupError): os.killpg(child.pid, 0)
                    with self.assertRaises(ChildProcessError): os.waitpid(child.pid, os.WNOHANG)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, 9)
                        real_wait(timeout=15)

    def test_launch_failure_keeps_bounded_receipt_without_possible_import(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(owner.subprocess, 'Popen', side_effect=OSError('private launch detail')):
                receipt = owner.run_browser(['inert-offline-fixture'], output)
            self.assertEqual(receipt['error_kind'], 'OSError')
            self.assertTrue(receipt['owned_process_group_cleanup_attempted'])
            self.assertFalse(receipt['owned_child_reaped'])
            self.assertFalse(receipt['import_dispatch_may_have_committed'])
            self.assertFalse(receipt['automatic_retry'])
            self.assertNotIn('private launch detail', (output / 'launcher-receipt.json').read_text())

    def test_detached_descendant_reaped_on_interruption_wait_failure_and_nonzero(self):
        original_popen = subprocess.Popen
        original_signals = [signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)]
        original_subreaper = owner.child_subreaper()
        for kind in ('KeyboardInterrupt', 'TimeoutExpired', 'RuntimeError', 'NonzeroExit'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                output = Path(directory); marker = output / 'detached-pid'
                (output / 'import-intent.json').write_text('{"possible_dispatch":true}')
                source = ('import subprocess,sys,pathlib,time,os; '
                    'child=subprocess.Popen([sys.executable,"-I","-S","-B","-c",'
                    '"import time; time.sleep(60)"],start_new_session=True); '
                    'pathlib.Path(sys.argv[1]).write_text(str(child.pid)); '
                    + ('time.sleep(0.1); os._exit(3)' if kind == 'NonzeroExit' else 'time.sleep(60)'))
                handles = []

                def launch(command, **kwargs):
                    child = original_popen([sys.executable, '-I', '-S', '-B', '-c', source, str(marker)], **kwargs)
                    handles.append(child)
                    if kind != 'NonzeroExit':
                        actual_wait = child.wait
                        first = True

                        def fail_once(*args, **wait_kwargs):
                            nonlocal first
                            if first:
                                first = False
                                deadline = time.monotonic() + 5
                                while not marker.exists() and time.monotonic() < deadline:
                                    time.sleep(0.01)
                                self.assertTrue(marker.exists())
                                if kind == 'KeyboardInterrupt': raise KeyboardInterrupt()
                                if kind == 'TimeoutExpired': raise subprocess.TimeoutExpired('inert', 0)
                                raise RuntimeError('private wait failure')
                            return actual_wait(*args, **wait_kwargs)
                        child.wait = fail_once
                    return child

                with patch.object(owner.subprocess, 'Popen', side_effect=launch):
                    receipt = owner.run_browser(['inert-fake-node'], output)
                detached = int(marker.read_text())
                self.assertEqual(receipt['status'], 'stopped_inspect_receipt_before_retry')
                self.assertEqual(receipt['error_kind'], kind)
                self.assertTrue(receipt['owned_child_reaped'])
                self.assertIsNone(receipt['cleanup_error_kind'])
                self.assertIn(detached, [row['pid'] for row in receipt['detached_descendants_reaped']])
                self.assertFalse(Path('/proc', str(detached)).exists())
                with self.assertRaises(ChildProcessError): os.waitpid(detached, os.WNOHANG)
                with self.assertRaises(ChildProcessError): os.waitpid(handles[0].pid, os.WNOHANG)
                self.assertEqual(owner.child_subreaper(), original_subreaper)
                self.assertEqual([signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)], original_signals)

    def test_partial_launch_before_handle_assignment_cleans_new_owned_child(self):
        original_popen = subprocess.Popen
        original_signals = [signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)]
        original_subreaper = owner.child_subreaper()
        for kind in ('KeyboardInterrupt', 'SIGTERM', 'RuntimeError'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                output = Path(directory); handles = []

                def interrupted_constructor(command, **kwargs):
                    child = original_popen([sys.executable, '-I', '-S', '-B', '-c',
                        'import time; time.sleep(60)'], **kwargs)
                    handles.append(child)
                    if kind == 'KeyboardInterrupt': raise KeyboardInterrupt()
                    if kind == 'SIGTERM': os.kill(os.getpid(), signal.SIGTERM)
                    raise RuntimeError('private partial launch')

                try:
                    with patch.object(owner.subprocess, 'Popen', side_effect=interrupted_constructor):
                        receipt = owner.run_browser(['inert-partial-launch'], output)
                    self.assertEqual(receipt['status'], 'stopped_inspect_receipt_before_retry')
                    self.assertEqual(receipt['error_kind'], 'InterruptedError' if kind == 'SIGTERM' else kind)
                    self.assertIsNone(receipt['returncode'])
                    self.assertIn(handles[0].pid, [row['pid'] for row in receipt['detached_descendants_reaped']])
                    self.assertIsNone(receipt['cleanup_error_kind'])
                    self.assertFalse(Path('/proc', str(handles[0].pid)).exists())
                    with self.assertRaises(ChildProcessError): os.waitpid(handles[0].pid, os.WNOHANG)
                    self.assertEqual(owner.child_subreaper(), original_subreaper)
                    self.assertEqual([signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)], original_signals)
                finally:
                    for child in handles:
                        if child.poll() is None:
                            os.killpg(child.pid, 9)
                            child.wait(timeout=15)


if __name__ == '__main__': unittest.main()
