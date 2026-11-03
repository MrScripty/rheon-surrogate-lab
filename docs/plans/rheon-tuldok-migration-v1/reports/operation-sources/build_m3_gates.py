"""Build root's exact M3 gate manifest from explicit, completed receipt bindings.

Reads local evidence and source bytes only: no catalog, endpoint, credentials,
numerical imports or execution. Missing/pending evidence never creates a gate.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import m3_orchestrator as m3

REVISION = '824665aefdc68a458b5510a0b33d6cae1eb4887d'
DEPLOYMENT = m3.BASE / 'Tuldok-deployed/current-03'
DEPLOYMENT_SHA = '09400b61cee86dde0014a71ade0a1d8e4f80372684b53950cd9a1cae31873c5a'
SCHEMA = m3.OPS / 'schema-rehearsal03-receipt.json'
SCHEMA_SHA = '928ed4be9c0008ed42c82ca1d04f0f62aa41f512376596438032616716446e45'
APPLICATION_SHA = '3243fb0fd5410594d4ff0454f35d8be0df83395229e8cd73b7fadd34ecb62376'
RUNTIME_SHA = 'b4329b27c27622d1fdf2b6c5712d6420c2d3e5fedf01c1450fc1a236cdbc9682'


def eq(path, value):
    return {'path': path.split('.') if isinstance(path, str) else path, 'equals': value}


def length(path, value):
    return {'path': path.split('.'), 'length': value}


def required_checks():
    passed = [eq('status', 'passed')]
    cold = passed + [eq('schema', 'rheon-cold-current-inspection-v1'),
        eq('old_catalog_backup_and_converter_unavailable', True),
        eq('read_only_database_counts_unchanged', True), eq('database_default_transaction_read_only', True),
        eq('workers_or_application_constructed', False)]
    checks = {
        'q2_independent': passed + [eq('schema', 'rheon-independent-q2-final-review-v1'),
            eq('attempt', '04'), eq('scope', 'isolated-rehearsal'), eq('accepted', True),
            eq('limits.production_or_service_cutover_qualified', False)],
        'q2_cold': cold, 'q2_restored_cold': cold,
        'q2_roundtrip': passed + [eq('schema', 'rheon-completed-current-roundtrip-v1'),
            eq('target_version', 10), eq('all_current_rows_sequences_and_managed_files_equal', True),
            eq('application_started', False)],
        'q6_browser': passed + [eq('schema', 'rheon-isolated-browser-inspection-v1'),
            eq('application', str(DEPLOYMENT)), eq('deployment_manifest_sha256', DEPLOYMENT_SHA),
            eq('production_endpoint_selected', False), eq('run_admission_or_training_performed', False),
            eq('activity_authority', 'external_observation'), eq('activity_available', True),
            eq('historical_source_run_state', 'completed'),
            eq('historical_representation', 'external_archived_evidence'), length('browser_page_errors', 0)],
        'q6_server': passed + [eq('schema', 'rheon-isolated-browser-server-v1'),
            eq('source', str(DEPLOYMENT)), eq('current_application_sha256', APPLICATION_SHA),
            eq('deployment_manifest.sha256', DEPLOYMENT_SHA), eq('application_bytes_pre_post_identical', True),
            eq('production_service_changed', False), eq('server_exited', True),
            eq('fresh_login_or_credential_minted', False)],
        'deployment_review': [eq('schema', 'rheon-independent-deployment-review-v1'),
            eq('status', 'passed_read_only_payload_identity_review'), eq('artifact', str(DEPLOYMENT)),
            eq('manifest_sha256', DEPLOYMENT_SHA), eq('exact_files_no_extra', True),
            eq('all_payload_source_bytes_equal', True), eq('all_top_level_python_source_artifact_equal', True)],
        'q3_pipeline': passed + [eq('observed_optimizer_updates.pipeline', 1),
            eq('production_accessed', False), eq('runtime.manifest_sha256', RUNTIME_SHA),
            eq(['weight_artifact_hash_parity', 'weights.npz'], True),
            eq(['weight_artifact_hash_parity', 'checkpoint-000.npz'], True),
            eq(['weight_artifact_hash_parity', 'checkpoint-001.npz'], True)],
        'q3_cuda_training': [eq('status', 'complete'), eq('mode', 'train'), eq('optimizer_updates', 1),
            eq('diagnostic_population', 'train_only'), eq('heldout_evaluations', 0),
            eq('execution.device', 'cuda:0'), eq('execution.model_dtype', 'float32'),
            eq('execution.parameters_gradients_adam_moments_cuda', True)],
        'q3_cpu_restore': [eq('status', 'complete'), eq('mode', 'restore'), eq('optimizer_updates', 0),
            eq('diagnostic_population', 'train_only'), eq('heldout_evaluations', 0),
            eq('execution.device', 'cpu'), eq('execution.cuda_initialized', False)],
        'q8_lifecycle': passed + [eq('production_accessed', False), eq('observed_optimizer_updates.lifecycle', 0),
            eq('runtime.manifest_sha256', RUNTIME_SHA), length('cases', 5),
            eq('cases.hash_cancel.custody_receipt.go_received', True),
            eq('cases.hash_cancel.custody_receipt.leader_returncode', None)],
        'q7_full74': passed + [eq('schema', 'rheon-migration-independent-full-cohort-v1'),
            eq('result.passed', True), eq('result.cases', 74), length('case_receipts', 74),
            eq('native_controls', 74), eq('optimizer_updates', 0),
            eq('all_call_sequences_exact_to_reproduced_old', True),
            eq('all_pressure_maxima_exact_to_reproduced_old', True),
            eq('result.same_rhs_and_trajectory_metrics_distinct', True)]}
    for case in ('train', 'restore'):
        checks['q3_pipeline'] += [eq('cases.' + case + '.state', 'completed'),
            eq('cases.' + case + '.custody_phase', 'drained'),
            eq('cases.' + case + '.custody_receipt.drained', True),
            eq('cases.' + case + '.custody_receipt.leader_returncode', 0)]
    for case in ('cancel', 'deadline', 'hash_cancel', 'shutdown', 'startup_failure'):
        checks['q8_lifecycle'] += [eq('cases.' + case + '.custody_phase', 'drained'),
            eq('cases.' + case + '.custody_receipt.drained', True),
            eq('cases.' + case + '.custody_receipt.echild', True)]
    return checks


def evaluate(document, checks):
    for check in checks:
        value = document
        for key in check['path']:
            value = value[key]
        m3.require(set(check) in ({'path', 'equals'}, {'path', 'length'}), 'Unknown pass predicate')
        valid = m3.canonical(value) == m3.canonical(check['equals']) if 'equals' in check else len(value) == check['length']
        m3.require(valid, 'Receipt PASS predicate failed: ' + str(check['path']))


def bound(row):
    m3.require(type(row) is dict and {'path', 'sha256'} <= set(row), 'Explicit receipt/artifact path and hash required')
    selected = Path(row['path'])
    m3.require(selected.is_absolute() and not selected.is_symlink(), 'Explicit absolute nonsymlink evidence path required')
    path = selected.resolve(strict=True)
    m3.require(path.is_relative_to(m3.BASE) and m3.sha(path) == row['sha256'],
               'Selected local evidence/artifact changed')
    return path


def verify_payload():
    manifest = DEPLOYMENT / 'deployment-manifest.json'
    m3.require(m3.sha(manifest) == DEPLOYMENT_SHA, 'Current03 manifest changed')
    declared = m3.read(manifest)['files']
    for row in declared:
        path = DEPLOYMENT / row['path']
        source = m3.SOURCE / row['path']
        m3.require(path.resolve(strict=True).is_relative_to(DEPLOYMENT) and not path.is_symlink()
                   and path.stat().st_size == row['bytes'] and m3.sha(path) == row['sha256']
                   and m3.stat.S_IMODE(path.stat().st_mode) == row['mode'], 'Current03 payload changed')
        m3.require(source.resolve(strict=True).is_relative_to(m3.SOURCE) and not source.is_symlink()
                   and m3.sha(source) == row['sha256'], 'Current03/source payload differs')
    actual = {p.relative_to(DEPLOYMENT).as_posix() for p in DEPLOYMENT.rglob('*') if p.is_file()}
    m3.require(actual == {row['path'] for row in declared} | {'deployment-manifest.json'}, 'Unexpected current03 payload file')
    return len(declared)


def build(selection):
    m3.require(selection['format'] == 'rheon_m3_gate_selection_v1', 'Unknown root selection')
    witness = m3.source_witness()
    m3.require(witness['revision'] == REVISION and witness == selection['tuldok_source'], 'Final824 source identity differs')
    baseline = required_checks()
    m3.require(set(baseline) <= set(selection['receipts']), 'Required actual completed receipt selections missing')
    result = {'format': 'rheon_m3_root_gates_v1', 'tuldok_source': witness, 'receipts': {}, 'artifacts': {}}
    documents = {}
    for name, row in selection['receipts'].items():
        path = bound(row)
        m3.require(row.get('checks'), 'Explicit PASS predicates required: ' + name)
        declared = {m3.canonical(check) for check in row['checks']}
        m3.require(all(m3.canonical(check) in declared for check in baseline.get(name, [])),
                   'Required PASS predicate omitted: ' + name)
        value = m3.read(path)
        evaluate(value, row['checks'])
        documents[name] = value
        result['receipts'][name] = dict(row, path=str(path))
    for name, row in selection['artifacts'].items():
        result['artifacts'][name] = dict(row, path=str(bound(row)))
    artifacts = result['artifacts']
    m3.require({'schema_rehearsal', 'deployment_manifest', 'rheon_binary'} <= set(artifacts), 'Exact artifact selections missing')
    m3.require(artifacts['schema_rehearsal'] == {'path': str(SCHEMA), 'sha256': SCHEMA_SHA}, 'Root-selected schema03 differs')
    m3.require(artifacts['deployment_manifest'] == {'path': str(DEPLOYMENT / 'deployment-manifest.json'), 'sha256': DEPLOYMENT_SHA}, 'Selected clean03 differs')
    verify_payload()
    schema = m3.read(SCHEMA)
    m3.require(schema['status'] == 'passed' and schema['schema_only'] is True
               and Path(schema['source_backup']).resolve() == m3.BACKUP.resolve()
               and schema['source_backup_receipt_sha256'] == m3.sha(m3.BACKUP / 'backup-receipt.json'), 'Schema03 does not bind backup02')
    final = documents['q2_independent']
    bound({'path': final['prepared']['path'], 'sha256': final['prepared']['file_sha256']})
    m3.require(final['current_application_sha256'] == APPLICATION_SHA
               and final['deployment']['path'] == str(DEPLOYMENT)
               and final['deployment']['manifest_sha256'] == DEPLOYMENT_SHA
               and final['deployment']['independent_review_sha256'] == result['receipts']['deployment_review']['sha256']
               and final['source']['backup_receipt_sha256'] == schema['source_backup_receipt_sha256'], 'Combined Q2 source/deployment differs')
    for proof, name in (('cold_current', 'q2_cold'), ('restored_cold', 'q2_restored_cold'),
                        ('completed_roundtrip', 'q2_roundtrip'), ('browser', 'q6_browser')):
        claimed = final['proofs'][proof]
        m3.require(Path(claimed['path']).resolve() == Path(result['receipts'][name]['path'])
                   and claimed['sha256'] == result['receipts'][name]['sha256'], 'Combined Q2 proof selection differs: ' + proof)
    for proof in final['proofs'].values():
        bound(proof)
    m3.require(documents['q2_roundtrip']['conversion_sha256'] == final['prepared']['canonical_sha256'], 'Roundtrip conversion differs')
    m3.require(documents['q2_restored_cold']['catalog'] == documents['q2_roundtrip']['isolated_destination']
               == documents['q6_server']['catalog'], 'Restored cold/browser clone differs')
    m3.require(Path(documents['q6_server']['browser_receipt']).resolve() == Path(result['receipts']['q6_browser']['path']), 'Browser/server receipt differs')
    train, restore = documents['q3_cuda_training'], documents['q3_cpu_restore']
    for case, name in (('train', 'q3_cuda_training'), ('restore', 'q3_cpu_restore')):
        saved = documents['q3_pipeline']['cases'][case]
        artifacts = [row for row in saved['artifacts'] if row['name'] == 'receipt.json']
        m3.require(len(artifacts) == 1 and artifacts[0]['sha256'] == result['receipts'][name]['sha256'],
                   'Q3 nested receipt is not the actual published Run artifact')
        for filename, measured in saved['prospective_admission']['host_files'].items():
            m3.require(m3.sha(m3.SOURCE / filename) == measured, 'Q3 host source changed: ' + filename)
    # A8 predates the separately qualified Projects parent-stat correction.
    # Bind its unchanged managed execution owners without claiming whole-source
    # identity for that earlier fixture. Q3 above covers current Projects bytes.
    for saved in documents['q8_lifecycle']['cases'].values():
        for filename in ('recipe_process.py', 'recipe_contract.py', 'torch_runtime.py'):
            m3.require(saved['prospective_admission']['host_files'][filename] == m3.sha(m3.SOURCE / filename),
                       'A8 execution owner changed: ' + filename)
    m3.require(train['prediction_inventory'] == restore['prediction_inventory'], 'Q3 exact CPU prediction parity differs')
    for name in ('weights.npz', 'checkpoint-000.npz', 'checkpoint-001.npz'):
        m3.require(train['outputs'][name]['sha256'] == restore['outputs'][name]['sha256'], 'Q3 checkpoint parity differs')
    for row in documents['q7_full74']['case_receipts']:
        bound(row)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--selection-sha256', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--write-gates', action='store_true')
    args = parser.parse_args(argv)
    m3.require(m3.sha(args.selection) == args.selection_sha256, 'Root selection changed')
    result = build(m3.read(args.selection))
    output = m3.OPS / 'm3-gates.json'
    if args.write_gates:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(m3.canonical(result)); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'status': 'all_actual_gates_verified', 'gate_written': args.write_gates,
                      'output': str(output) if args.write_gates else None,
                      'manifest_sha256': m3.digest(result), 'receipt_count': len(result['receipts'])}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        sys.stderr.write('M3 gate builder stopped (' + type(error).__name__ + '); no PASS evidence is inferred.\n')
        sys.exit(1)
