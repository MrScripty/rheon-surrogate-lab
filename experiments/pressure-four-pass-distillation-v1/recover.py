"""Reconcile one known restore acknowledgement, then perform read-only collection.

Never submits, approves or retrains. The exact Run ID comes from durable request
lookup; current owner API readback must match every admitted request field.
The original unresolved journal is retained before local reconciliation.
"""
import argparse
from pathlib import Path
import sys
import time

from common import ROOT, read, sha, write


def recover(args):
    started = time.perf_counter()
    config = read(args.config)
    source = ROOT / config['source_lab'] / 'experiments'
    sys.path[:0] = [str(args.tuldok), str(source / 'pressure-throughput-feedback-v1'),
                   str(source / 'dependency-migration-v1')]
    import agent_api
    import run_study
    from qualify_typed_pressure import OwnerHTTP
    directory = args.output
    journal_path = directory / 'mutations/13-admit-restore.json'
    journal = read(journal_path)
    if journal['status'] != 'unresolved' or journal['operation'] != 'runs.submit_external':
        raise ValueError('One unresolved restore submission required; do not replay a managed command')
    owner = OwnerHTTP(agent_api, args.endpoint, args.catalog_id, args.session_file)
    owner.transport.timeout = 600
    run = owner.call('runs.get', dict(run_id=args.run_id))
    body = journal['arguments']
    if not (run['request_key'] == body['request_key'] and run['graph']['id'] == body['graph_id']
            and run['graph_revision'] == body['graph_revision']
            and run['material_sha256'] == body['expected_digest']
            and run['approval_id'] == body['approval_id'] and run['actor_id'] == owner.identity['actor_id']
            and run['dataset_id'] == args.catalog_id and run['state'] == 'completed'
            and run['custody_phase'] == 'drained'):
        raise ValueError('Exact completed owner-approved restore request not established; leave journal unresolved')
    recovery = directory / 'recovery'
    recovery.mkdir(exist_ok=False)
    original_sha = sha(journal_path)
    (recovery / 'original-unresolved.json').write_bytes(journal_path.read_bytes())
    write(recovery / 'readback.json', dict(status='exact_completed_request_confirmed',
        original_journal_sha256=original_sha, run=run, remote_mutations=0,
        verification='Current owner API matches request key, graph ID/revision, material digest, approval, actor and catalog; completed and drained'))
    journal.update(status='acknowledged', result=run,
        reconciliation=dict(original_journal_sha256=original_sha, evidence='recovery/readback.json',
                            mode='read_only_exact_completed_request', remote_mutations=0))
    write(journal_path, journal)
    original_wait = run_study.await_drained
    def wait(client, run_id, timeout=2100):
        result = original_wait(client, run_id, timeout)
        terminal = directory / ('terminal-' + run_id + '.json')
        if not terminal.exists():
            write(terminal, result)
        return result
    run_study.await_drained = wait
    # The recovery branch of execute only consumes acknowledged existing IDs.
    # Reuse the already negotiated current owner, without another login/session.
    run_study.OwnerHTTP = lambda *_: owner
    args.root = ROOT
    args.collect_existing = True
    run_study.execute(args)
    write(recovery / 'receipt.json', dict(status='completed', elapsed_s=time.perf_counter() - started,
        remote_mutations=0, retraining_updates=0, restore_run_id=args.run_id,
        original_journal_sha256=original_sha, completed_study_receipt_sha256=sha(directory / 'receipt.json')))
    print('Recovered exact managed restore without remote mutation', args.run_id, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'assembly', 'tuldok', 'session-file', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ['endpoint', 'catalog-id', 'project-alias', 'run-id']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    recover(args)
